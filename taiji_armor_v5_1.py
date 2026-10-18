"""XuanShu H3 太极装甲 v5.1 — 末梦秋花v5.1：预测自适应版。

v5.1 = v5（极简截断漂移版）+ v3s1 预测性机制融合：
- 保留 v5：l1 delta + AdaCorrection alpha 半缓存半计算（alpha>0.5 才整步复用）
        + EMA 趋势补偿 + FeatFix 轻量增益 + IEC 误差反馈动态扣减阈值
        + phase_aware 阶段感知 + max_consecutive 截断 + end_step 末尾保护
        + chunk_size 分块 delta（batch>1 保守）
- 引入 v3s1：_fit_log_curve 多项式拟合（warmup 期对 log(delta)~step polyfit，
        外推后续步阈值）+ 置信度衰减（三段式概率作为 alpha 源）
        + VRAM 水位门（显存超限时阈值×1.5 保守化）
- 仍砍掉：五策略择优、护体真元 fallback、error_budget 熔断、freq_decouple

阈值链路：
  thresh = max(base_threshold × safety × phase_mult,
               poly_pred_delta × safety × phase_mult)      # 预测优先，下限保底
  thresh ×= 1.5                                            # VRAM 水位门
  eff_thresh = max(thresh - iec_pen, base_threshold × 0.2) # IEC 动态扣减
  alpha = confidence(delta, eff_thresh)                    # 置信度衰减三段式
  should_cache = alpha > 0.5                               # 半缓存半计算
"""

import time
import torch
from typing import Optional

import numpy as np


# ============================================================
# 状态
# ============================================================
class TaijiArmorV5_1State:
    def __init__(self):
        self.step_idx = 0
        self.prev_input: Optional[torch.Tensor] = None   # CPU float
        self.prev_output: Optional[torch.Tensor] = None  # CPU float
        self.output_trend: Optional[torch.Tensor] = None
        self.consecutive_cache = 0
        self.iec_error_buffer = []
        self.current_phase = 'mid'
        self.cache_count = 0
        self.real_count = 0
        # v3s1 预测性状态
        self.delta_history = []      # [(step, delta)]
        self.poly_coeffs = None      # np.polyfit 系数
        self.poly_thresh_used = 0
        self.base_thresh_used = 0
        self.peak_vram_mb = 0
        self.reported = False

    def reset(self):
        self.prev_input = None
        self.prev_output = None
        self.output_trend = None
        self.consecutive_cache = 0
        self.iec_error_buffer = []
        self.cache_count = 0
        self.real_count = 0
        self.delta_history = []
        self.poly_coeffs = None
        self.poly_thresh_used = 0
        self.base_thresh_used = 0
        self.peak_vram_mb = 0


# ============================================================
# 工具函数
# ============================================================
def read_vram_mb() -> int:
    try:
        return int(torch.cuda.memory_reserved() // (1024 * 1024))
    except Exception:
        return 0


# ============================================================
# 调度器
# ============================================================
class TaijiArmorV5_1Scheduler:
    def __init__(
        self,
        total_steps: int,
        warmup_steps: int = 2,
        safety_margin: float = 1.6,
        correction_strength: float = 0.2,
        end_step: int = -2,
        max_consecutive: int = 3,
        base_threshold: float = 0.12,
        adac_min_alpha: float = 0.3,
        iec_enabled: bool = True,
        iec_gamma: float = 0.6,
        chunk_size: int = 2,
        # v3s1 预测性参数
        poly_calibrate: bool = True,
        poly_degree: int = 2,
        vram_thresh_mb: int = 7500,
        confidence_mode: bool = True,
        verbose: bool = True,
    ):
        self.total_steps = int(total_steps)
        self.warmup_steps = max(0, int(warmup_steps))
        self.safety_margin = float(safety_margin)
        self.correction_strength = float(correction_strength)
        self.end_step = int(end_step)
        self.max_consecutive = max(1, int(max_consecutive))
        self.base_threshold = float(base_threshold)
        self.adac_min_alpha = float(adac_min_alpha)
        self.iec_enabled = bool(iec_enabled)
        self.iec_gamma = float(iec_gamma)
        self.chunk_size = max(1, int(chunk_size))
        self.poly_calibrate = bool(poly_calibrate)
        self.poly_degree = max(1, int(poly_degree))
        self.vram_thresh_mb = int(vram_thresh_mb)
        self.confidence_mode = bool(confidence_mode)
        self.verbose = verbose

    # ---------- 阶段感知（v5） ----------
    def _phase(self, step: int) -> str:
        p = step / max(1, self.total_steps)
        if p < 0.3:
            return 'early'
        elif p < 0.75:
            return 'mid'
        return 'late'

    def _phase_mult(self, phase: str) -> float:
        return {'early': 1.4, 'mid': 1.0, 'late': 1.2}.get(phase, 1.0)

    # ---------- delta（v5 chunk 分块） ----------
    def _delta(self, cur: torch.Tensor, prev: torch.Tensor) -> float:
        b = cur.shape[0]
        if b <= 1 or b <= self.chunk_size:
            return (cur - prev).abs().mean().item()
        blocks = []
        for i in range(0, b, self.chunk_size):
            blocks.append((cur[i:i + self.chunk_size] - prev[i:i + self.chunk_size]).abs().mean().item())
        return max(blocks)

    # ---------- VRAM 水位门（v3s1） ----------
    def _vram_gate(self, state) -> bool:
        vram = read_vram_mb()
        if vram > state.peak_vram_mb:
            state.peak_vram_mb = vram
        pressure = vram > self.vram_thresh_mb
        if self.verbose and pressure:
            print("[TaijiArmor v5.1] VRAM水位: %dMB > %dMB → 阈值×1.5" % (vram, self.vram_thresh_mb))
        return pressure

    # ---------- 多项式拟合 / 外推（v3s1 _fit_log_curve） ----------
    def _fit_log_curve(self, state) -> None:
        """warmup 期对 log(delta)~step 做多项式拟合，外推后续步阈值。"""
        if len(state.delta_history) < 2:
            return
        steps = np.array([d[0] for d in state.delta_history], dtype=np.float64)
        deltas = np.array([d[1] for d in state.delta_history], dtype=np.float64)
        log_deltas = np.log(np.maximum(deltas, 1e-8))
        degree = min(self.poly_degree, len(steps) - 1)
        try:
            coeffs = np.polyfit(steps, log_deltas, degree)
            state.poly_coeffs = coeffs.tolist()
            if self.verbose:
                preds = []
                for s in range(int(steps.min()), int(steps.max()) + 4):
                    preds.append("s%d→%.4f" % (s, float(np.exp(np.polyval(coeffs, s)))))
                print("[TaijiArmor v5.1] 对数线性拟合(deg=%d, n=%d): %s" % (
                    degree, len(steps), " ".join(preds)))
        except Exception as e:
            if self.verbose:
                print("[TaijiArmor v5.1] 对数拟合失败: %s" % e)

    def _predict_delta(self, step: int, state) -> Optional[float]:
        if state.poly_coeffs is None:
            return None
        try:
            log_val = float(np.polyval(state.poly_coeffs, step))
            return max(float(np.exp(log_val)), 0.001)
        except Exception:
            return None

    # ---------- 置信度衰减 → alpha（v3s1 _compute_confidence） ----------
    def _compute_alpha(self, delta: float, thresh: float) -> float:
        """三段式置信度：delta 远低于阈值 → alpha=1（整步复用）；
        接近阈值 → 线性衰减（半缓存半计算的过渡带）；超阈值 → 0（全算）。"""
        if not self.confidence_mode:
            return 1.0 if delta < thresh else 0.0
        if delta < thresh * 0.5:
            return 1.0
        elif delta < thresh:
            return (thresh - delta) / (thresh * 0.5)
        return 0.0

    # ---------- IEC 误差反馈（v5，缓存步真实误差） ----------
    def _iec_penalty(self, state) -> float:
        if not self.iec_enabled or len(state.iec_error_buffer) < 2:
            return 0.0
        errors = torch.tensor([e for _, e in state.iec_error_buffer], dtype=torch.float32)
        w = torch.exp(torch.linspace(-2.0, 0.0, len(errors)))
        w = w / w.sum()
        return (errors * w).sum().item() * self.iec_gamma

    # ---------- 决策 ----------
    def should_reuse(self, state, x: torch.Tensor):
        step = state.step_idx

        # 形状变化重置
        if state.prev_input is not None and tuple(x.shape) != tuple(state.prev_input.shape):
            if self.verbose:
                print("[V5.1] 形状变化 → 重置")
            state.reset()
            return (False, 0.0, 0.0, 0.0, "shape_reset")

        if state.prev_output is None or state.prev_input is None:
            return (False, 0.0, 0.0, 0.0, "no_ref")

        state.current_phase = self._phase(step)

        # delta（始终计算并记录，供拟合/报告）
        cur_cpu = x.detach().float().cpu()
        delta = self._delta(cur_cpu, state.prev_input)
        state.delta_history.append((step, delta))

        # 曲线拟合：样本≥2 即拟合一次（warmup 期为主），成功后固定外推
        if self.poly_calibrate and state.poly_coeffs is None and len(state.delta_history) >= 2:
            self._fit_log_curve(state)

        # warmup：全算
        if step < self.warmup_steps:
            return (False, delta, 0.0, 0.0, "warmup(%d<%d)" % (step, self.warmup_steps))

        # end_step 保护
        eff_end = self.end_step if self.end_step >= 0 else self.total_steps + self.end_step
        if step >= eff_end:
            return (False, delta, 0.0, 0.0, "end_protect")

        # === 阈值计算：预测外推优先，基础阈值保底 ===
        base = max(self.base_threshold, 1e-6)
        thresh = base * self.safety_margin * self._phase_mult(state.current_phase)
        src = "base"
        if self.poly_calibrate and state.poly_coeffs is not None:
            pred = self._predict_delta(step, state)
            if pred is not None and pred > 0:
                poly_thresh = pred * self.safety_margin * self._phase_mult(state.current_phase)
                thresh = max(poly_thresh, base * self.safety_margin)
                state.poly_thresh_used += 1
                src = "poly"
            else:
                state.base_thresh_used += 1
        else:
            state.base_thresh_used += 1

        # VRAM 水位门：显存压力 → 阈值×1.5
        if self._vram_gate(state):
            thresh *= 1.5

        # IEC 反馈：历史误差动态扣减阈值（下限保底 base×0.2）
        iec_pen = self._iec_penalty(state)
        eff_thresh = max(thresh - iec_pen, base * 0.2)

        # AdaCorrection alpha：置信度衰减生成 alpha，alpha>0.5 才整步复用
        alpha = self._compute_alpha(delta, eff_thresh)
        should_cache = alpha > 0.5

        # max_consecutive 截断
        if should_cache and state.consecutive_cache >= self.max_consecutive:
            should_cache = False
            alpha = 0.0
            reason = "max_consecutive(%d)" % state.consecutive_cache
        else:
            reason = ("cache(a=%.2f)" % alpha) if should_cache \
                else "compute(d=%.4f>=t=%.4f)" % (delta, eff_thresh)

        return (should_cache, delta, eff_thresh, alpha, reason)

    # ---------- 补偿（v5：EMA 趋势 + FeatFix） ----------
    def compensate(self, base_output, state, alpha):
        out = base_output
        # EMA 趋势
        if self.correction_strength > 0 and state.output_trend is not None and state.step_idx > 1:
            trend = state.output_trend.to(base_output.device).to(base_output.dtype)
            strength = self.correction_strength * alpha
            with torch.no_grad():
                out_std = base_output.std().clamp(min=1e-6)
                trend_mag = trend.abs().mean().clamp(min=1e-6)
                max_str = (2.0 * out_std / trend_mag).item()
                strength = min(strength, max_str)
            comp = out + trend * strength
            if torch.isfinite(comp).all():
                out = comp
        # FeatFix 轻量增益
        if alpha > 0.01:
            gain = 1.0 + 0.15 * alpha
            if abs(gain - 1.0) > 0.005:
                out = out * gain
        return out

    # ---------- 计算/缓存后状态更新（v5） ----------
    def update_after_compute(self, state, output_cpu, was_cached):
        if self.correction_strength > 0 and state.prev_output is not None:
            change = output_cpu - state.prev_output
            if state.output_trend is None:
                state.output_trend = change.clone()
            else:
                state.output_trend = 0.7 * state.output_trend + 0.3 * change
        # IEC：仅缓存步记录真实误差
        if was_cached and state.prev_output is not None:
            err = (output_cpu - state.prev_output).abs().mean().item()
            state.iec_error_buffer.append((state.step_idx, err))
            if len(state.iec_error_buffer) > 6:
                state.iec_error_buffer.pop(0)
        state.prev_input = state._cur_input_ref
        state.prev_output = output_cpu
        state.step_idx += 1


# ============================================================
# Wrapper
# ============================================================
def make_v5_1_wrapper(total_steps, warmup_steps, safety_margin, correction_strength,
                      end_step, max_consecutive, base_threshold, adac_min_alpha,
                      iec_enabled, iec_gamma, chunk_size,
                      poly_calibrate, poly_degree, vram_thresh_mb, confidence_mode,
                      verbose):
    state = TaijiArmorV5_1State()
    sched = TaijiArmorV5_1Scheduler(
        total_steps, warmup_steps, safety_margin, correction_strength,
        end_step, max_consecutive, base_threshold, adac_min_alpha,
        iec_enabled, iec_gamma, chunk_size,
        poly_calibrate, poly_degree, vram_thresh_mb, confidence_mode, verbose
    )

    def _maybe_report():
        if state.step_idx >= max(0, total_steps - 1) and not state.reported:
            state.reported = True
            tot = state.cache_count + state.real_count
            rate = (state.cache_count / tot * 100) if tot else 0.0
            iec_str = ""
            if iec_enabled and state.iec_error_buffer:
                iec_str = " iec_err=%s" % (
                    ", ".join("%.4f" % e for _, e in state.iec_error_buffer))
            print(("[XuanShuTaijiArmor v5.1] steps=%d warmup=%d "
                   "cache=%d real=%d hit_rate=%.1f%% peak_vram=%dMB "
                   "poly_used=%d base_used=%d%s") % (
                   total_steps, warmup_steps, state.cache_count, state.real_count,
                   rate, state.peak_vram_mb, state.poly_thresh_used,
                   state.base_thresh_used, iec_str))

    def wrapper(apply_model, args):
        x = args["input"]
        t0 = time.perf_counter()

        reuse, delta, thresh, alpha, reason = sched.should_reuse(state, x)

        if reuse:
            state.cache_count += 1
            state.consecutive_cache += 1
            base_out = state.prev_output.to(x.device).to(x.dtype)
            output = sched.compensate(base_out, state, alpha)
            state._cur_input_ref = x.detach().float().cpu()
            sched.update_after_compute(state, output.detach().float().cpu(), True)
            if verbose:
                print("[V5.1] step %d/%d CACHE d=%.5f t=%.5f a=%.2f %s" % (
                    state.step_idx - 1, total_steps, delta, thresh, alpha, reason))
            _maybe_report()
            return output

        # 全算
        output = apply_model(x, args["timestep"], **args["c"])
        state.real_count += 1
        state.consecutive_cache = 0
        state._cur_input_ref = x.detach().float().cpu()
        out_cpu = output.detach().float().cpu()
        sched.update_after_compute(state, out_cpu, False)
        if verbose:
            dt = time.perf_counter() - t0
            print("[V5.1] step %d/%d COMPUTE %.1fs %s" % (
                state.step_idx - 1, total_steps, dt, reason))
        _maybe_report()
        return output

    return state, wrapper


# ============================================================
# ComfyUI 节点
# ============================================================
class MiniMaxH3TaijiArmorV5_1:
    """玄枢太极装甲 v5.1（末梦秋花v5.1）：预测自适应版。

    v5 极简截断漂移 + v3s1 预测性：
    _fit_log_curve 多项式外推后续步阈值 + 置信度衰减 alpha（>0.5 整步复用）
    + VRAM 水位门 + IEC 误差反馈动态扣减。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 6, "min": 1, "max": 200,
                    "tooltip": "总去噪步数"}),
                "warmup_steps": ("INT", {"default": 2, "min": 0, "max": 100,
                    "tooltip": "探针步数：前N步全算并收集delta供曲线拟合"}),
                "safety_margin": ("FLOAT", {"default": 1.6, "min": 0.5, "max": 4.0, "step": 0.1,
                    "tooltip": "安全边际：阈值×此值，越大越保守"}),
                "correction_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "EMA趋势补偿强度"}),
                "end_step": ("INT", {"default": -2, "min": -20, "max": 100,
                    "tooltip": "末尾保护步数，负数倒数"}),
                "max_consecutive": ("INT", {"default": 3, "min": 1, "max": 10,
                    "tooltip": "连续缓存上限：截断漂移累积"}),
                "base_threshold": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 0.5, "step": 0.01,
                    "tooltip": "缓存判定基准阈值（预测外推的下限保底）"}),
                "iec_gamma": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "IEC误差反馈增益：历史误差动态扣减阈值"}),
                "chunk_size": ("INT", {"default": 2, "min": 1, "max": 8,
                    "tooltip": "batch分块delta（保守取最大块）"}),
                # === v3s1 预测性参数 ===
                "poly_calibrate": ("BOOLEAN", {"default": True,
                    "tooltip": "预测性外推：warmup期对log(delta)~step拟合多项式，外推后续步阈值"}),
                "poly_degree": ("INT", {"default": 2, "min": 1, "max": 5,
                    "tooltip": "拟合阶数：2=二次（推荐），1=线性"}),
                "vram_thresh_mb": ("INT", {"default": 7500, "min": 1000, "max": 20000,
                    "tooltip": "显存水位阈值(MB)：超限自动阈值×1.5保守化"}),
                "confidence_mode": ("BOOLEAN", {"default": True,
                    "tooltip": "置信度衰减：三段式概率作为alpha（半缓存半计算过渡带），关闭=二值判定"}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "末梦秋花v5.1（预测自适应）"

    def apply(self, model, total_steps, warmup_steps, safety_margin, correction_strength,
              end_step, max_consecutive, base_threshold, iec_gamma, chunk_size,
              poly_calibrate, poly_degree, vram_thresh_mb, confidence_mode, verbose):
        state, wrapper = make_v5_1_wrapper(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            safety_margin=safety_margin,
            correction_strength=correction_strength,
            end_step=end_step,
            max_consecutive=max_consecutive,
            base_threshold=base_threshold,
            adac_min_alpha=0.3,
            iec_enabled=True,
            iec_gamma=iec_gamma,
            chunk_size=chunk_size,
            poly_calibrate=poly_calibrate,
            poly_degree=poly_degree,
            vram_thresh_mb=vram_thresh_mb,
            confidence_mode=confidence_mode,
            verbose=verbose,
        )
        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        patched.model_options["xuanshu_taiji_armor_v5_1"] = {"state": state}
        for k in ["xuanshu_teacache_guard", "xuanshu_taiji_armor",
                  "xuanshu_taiji_armor_v4", "xuanshu_taiji_armor_v4_1",
                  "xuanshu_taiji_armor_v4_final", "xuanshu_taiji_armor_v4s",
                  "xuanshu_taiji_armor_v3s1", "xuanshu_taiji_armor_v5"]:
            patched.model_options.pop(k, None)
        return (patched,)


NODE_CLASS_MAPPINGS_V5_1 = {"MiniMaxH3TaijiArmorV5_1": MiniMaxH3TaijiArmorV5_1}
NODE_DISPLAY_NAME_MAPPINGS_V5_1 = {"MiniMaxH3TaijiArmorV5_1": "末梦秋花v5.1(预测自适应)"}
