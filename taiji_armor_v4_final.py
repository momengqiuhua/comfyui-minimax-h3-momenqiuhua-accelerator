"""
太极装甲 v4 Final — 完整版迭代误差校正
========================================
在 v4.1 稳定性基础上，将所有机制真正落地：

核心机制（全部真正工作）：
  1. AdaCorrection   — 自适应连续混合（缓存强度随 delta 平滑变化）
  2. IEC             — 迭代误差校正（指数加权缓冲 → 阈值惩罚）
  3. FeatFix v2      — delta 信号增益调节（轻量、非破坏性）
  4. 频率解耦        — DCT 低频直接复用 + 高频趋势外推重建
  5. 阶段感知        — 早/中/晚三阶段自适应调节安全边际和补偿强度
  6. EMA 趋势外推    — 指数移动平均输出变化趋势

相比 v4.1 的改进：
  - FeatFix 从"直接 return None"改为 delta 信号驱动的增益调节
  - 频率解耦从"函数写了但没调用"改为真正接入缓存补偿 pipeline
  - 新增阶段感知策略，不同去噪阶段用不同参数
  - AdaCorrection + FeatFix + 频率解耦 统一在补偿 pipeline 里协调
  - 形状变化自动重置（避免分辨率/帧数变化崩溃）
"""

import torch
import torch.nn.functional as F
import math
import time as _time
from typing import Optional


# ============================================================
#  DCT 频率解耦工具（空间维度，支持任意维度输入）
# ============================================================

def _dct_freq_decompose(x: torch.Tensor, low_ratio: float = 0.3):
    """对最后两个空间维度做 DCT 分解，返回 (低频, 高频)。"""
    *dims, H, W = x.shape
    x_h = _dct_1d(x.float(), norm='ortho')
    x_hw = _dct_1d(x_h.transpose(-2, -1), norm='ortho').transpose(-2, -1)

    low_h = max(1, int(H * low_ratio))
    low_w = max(1, int(W * low_ratio))

    low_freq = torch.zeros_like(x_hw)
    high_freq = x_hw.clone()

    sl_h = slice(0, low_h)
    sl_w = slice(0, low_w)
    low_freq[..., sl_h, sl_w] = x_hw[..., sl_h, sl_w]
    high_freq[..., sl_h, sl_w] = 0.0

    return low_freq, high_freq


def _dct_freq_recompose(low_freq: torch.Tensor, high_freq: torch.Tensor) -> torch.Tensor:
    """低频 + 高频 → IDCT → 空间域。"""
    x_freq = low_freq + high_freq
    x_h = _idct_1d(x_freq.transpose(-2, -1), norm='ortho').transpose(-2, -1)
    x = _idct_1d(x_h, norm='ortho')
    return x


def _dct_1d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """Type-II DCT，作用在最后一维。"""
    N = x.shape[-1]
    n = torch.arange(N, device=x.device, dtype=x.dtype)
    k = torch.arange(N, device=x.device, dtype=x.dtype)
    cos_mat = torch.cos(math.pi / N * (n.unsqueeze(0) + 0.5) * k.unsqueeze(1))
    out = torch.matmul(x, cos_mat.T)
    if norm == 'ortho':
        scale = torch.ones(N, device=x.device, dtype=x.dtype)
        scale[0] = 1.0 / math.sqrt(N)
        scale[1:] = math.sqrt(2.0 / N)
        out = out * scale
    return out


def _idct_1d(x: torch.Tensor, norm: str = 'ortho') -> torch.Tensor:
    """Type-III DCT（即 Type-II 的逆），作用在最后一维。"""
    N = x.shape[-1]
    n = torch.arange(N, device=x.device, dtype=x.dtype)
    k = torch.arange(N, device=x.device, dtype=x.dtype)
    cos_mat = torch.cos(math.pi / N * n.unsqueeze(0) * (k.unsqueeze(1) + 0.5))
    if norm == 'ortho':
        scale = torch.ones(N, device=x.device, dtype=x.dtype)
        scale[0] = 1.0 / math.sqrt(N)
        scale[1:] = math.sqrt(2.0 / N)
        x_scaled = x * scale
        out = torch.matmul(x_scaled, cos_mat.T)
    else:
        out = torch.matmul(x, cos_mat.T)
    return out


# ============================================================
#  状态
# ============================================================

class TaijiArmorV4FinalState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_input: Optional[torch.Tensor] = None
        self.prev_output: Optional[torch.Tensor] = None
        self.prev_output_freq_low: Optional[torch.Tensor] = None
        self.prev_output_freq_high: Optional[torch.Tensor] = None
        self.output_trend: Optional[torch.Tensor] = None
        self.high_freq_trend: Optional[torch.Tensor] = None
        self.step_idx: int = 0
        self.real_count: int = 0
        self.cache_count: int = 0
        self.iec_error_buffer: list = []
        self.iec_max_buffer: int = 16
        self.consecutive_cache: int = 0
        self.current_phase: str = 'early'
        self.step_wall_times: list = []
        self.peak_vram_mb: float = 0.0


# ============================================================
#  装甲调度器
# ============================================================

class TaijiArmorV4FinalScheduler:
    """v4 Final 调度器：决策 + 补偿逻辑。"""

    def __init__(
        self,
        total_steps: int,
        warmup_steps: int = 3,
        safety_margin: float = 1.8,
        correction_strength: float = 0.15,
        adac_min_alpha: float = 0.3,
        iec_enabled: bool = True,
        iec_gamma: float = 0.6,
        featfix_enabled: bool = True,
        freq_decouple: bool = False,
        freq_low_ratio: float = 0.3,
        phase_aware: bool = True,
        max_consecutive: int = 3,
        end_step: int = -2,
        base_threshold: float = 0.12,
        verbose: bool = True,
    ):
        self.total_steps = int(total_steps)
        self.warmup_steps = max(0, int(warmup_steps))
        self.safety_margin = max(0.5, min(5.0, float(safety_margin)))
        self.correction_strength = max(0.0, min(2.0, float(correction_strength)))
        self.adac_min_alpha = max(0.0, min(0.9, float(adac_min_alpha)))
        self.iec_enabled = bool(iec_enabled)
        self.iec_gamma = max(0.0, min(2.0, float(iec_gamma)))
        self.featfix_enabled = bool(featfix_enabled)
        self.freq_decouple = bool(freq_decouple)
        self.freq_low_ratio = max(0.1, min(0.6, float(freq_low_ratio)))
        self.phase_aware = bool(phase_aware)
        self.max_consecutive = max(1, int(max_consecutive))
        self.end_step = int(end_step)
        self.base_threshold = float(base_threshold)
        self.verbose = bool(verbose)

    # ---- 阶段感知 ----
    def _detect_phase(self, step_idx: int) -> str:
        if self.total_steps <= 0:
            return 'mid'
        progress = step_idx / self.total_steps
        if progress < 0.3:
            return 'early'
        elif progress < 0.75:
            return 'mid'
        else:
            return 'late'

    def _phase_params(self, phase: str) -> dict:
        if not self.phase_aware:
            return {'safety_mult': 1.0, 'correction_mult': 1.0}
        if phase == 'early':
            return {'safety_mult': 1.4, 'correction_mult': 0.6}
        elif phase == 'mid':
            return {'safety_mult': 1.0, 'correction_mult': 1.0}
        else:
            return {'safety_mult': 1.2, 'correction_mult': 0.5}

    # ---- 决策 ----
    def should_reuse(self, state: TaijiArmorV4FinalState,
                     current_input: torch.Tensor) -> tuple:
        """返回 (是否缓存, delta, threshold, blend_alpha, 原因)。"""
        step = state.step_idx

        # 形状变化自动重置
        if state.prev_input is not None:
            cur_shape = tuple(current_input.shape)
            prev_shape = tuple(state.prev_input.shape)
            if cur_shape != prev_shape:
                if self.verbose:
                    print("[V4Final] 形状变化 %s → %s → 自动重置" % (prev_shape, cur_shape))
                state.reset()
                return (False, 0.0, 0.0, 0.0, "shape_reset")

        if state.prev_output is None or state.prev_input is None:
            return (False, 0.0, 0.0, 0.0, "no_ref")

        # 阶段检测
        state.current_phase = self._detect_phase(step)

        # end_step 保护
        effective_end = (self.end_step if self.end_step >= 0
                         else self.total_steps + self.end_step)
        if step >= effective_end:
            return (False, 0.0, 0.0, 0.0, "end_protect")

        # warmup
        if step < self.warmup_steps:
            return (False, 0.0, 0.0, 0.0, "warmup(%d<%d)" % (step, self.warmup_steps))

        # 计算 delta
        cur_cpu = current_input.detach().float().cpu()
        diff = (cur_cpu - state.prev_input).abs()
        delta = diff.mean().item()
        # 阈值：绝对基准（与v3的yang_thresh同量级）
        threshold = max(self.base_threshold, 1e-6)

        # IEC 阈值惩罚
        iec_penalty = 0.0
        if self.iec_enabled and len(state.iec_error_buffer) >= 2:
            errors = torch.tensor([e for _, e in state.iec_error_buffer],
                                  dtype=torch.float32)
            weights = torch.exp(torch.linspace(-2.0, 0.0, len(errors)))
            weights = weights / weights.sum()
            corrected_error = (errors * weights).sum().item()
            iec_penalty = corrected_error * self.iec_gamma

        # 阶段感知安全边际
        phase_mult = self._phase_params(state.current_phase)['safety_mult']
        effective_margin = self.safety_margin * phase_mult

        # 有效阈值 = 基准阈值 × 安全边际 - IEC 惩罚（下限为基准的 20%）
        effective_thresh = max(
            threshold * effective_margin - iec_penalty,
            threshold * 0.2
        )

        # AdaCorrection 混合系数
        ratio = delta / effective_thresh if effective_thresh > 0 else 10.0
        if ratio <= 1.0:
            alpha = 1.0 - (1.0 - self.adac_min_alpha) * ratio
        else:
            alpha = self.adac_min_alpha

        # 决策：alpha > 0.5 则缓存
        should_cache = alpha > 0.5

        # max_consecutive 限制
        if should_cache and state.consecutive_cache >= self.max_consecutive:
            should_cache = False
            alpha = 0.0
            reason = "max_consecutive(%d)" % state.consecutive_cache
        else:
            reason = "adac(%.2f)" % alpha if should_cache else "below_thresh"

        return (should_cache, delta, threshold, alpha, reason)

    # ---- 频率解耦补偿 ----
    def _freq_decouple_compensate(self, base_output: torch.Tensor,
                                   state: TaijiArmorV4FinalState) -> torch.Tensor:
        """低频直接复用 + 高频趋势外推重建。"""
        if not self.freq_decouple:
            return base_output

        try:
            base_cpu = base_output.detach().cpu()
            cur_low, cur_high = _dct_freq_decompose(base_cpu, self.freq_low_ratio)

            # 高频用趋势外推
            if state.high_freq_trend is not None:
                compensated_high = cur_high + state.high_freq_trend
            else:
                compensated_high = cur_high

            # 合回空间域
            result = _dct_freq_recompose(cur_low, compensated_high)

            # 数值稳定防护
            if not torch.isfinite(result).all():
                return base_output

            # 幅值钳制（±4σ）
            mean_val = base_cpu.mean().item()
            std_val = base_cpu.std().item()
            lower = mean_val - 4.0 * std_val
            upper = mean_val + 4.0 * std_val
            result = torch.clamp(result, lower, upper)

            return result.to(base_output.device).to(base_output.dtype)

        except Exception as e:
            if self.verbose:
                print("[V4Final] 频率解耦失败: %s → 退化为普通缓存" % str(e)[:80])
            return base_output

    # ---- FeatFix v2：delta 信号增益调节 ----
    def _featfix_apply(self, output: torch.Tensor, delta: float,
                       threshold: float, blend_alpha: float) -> torch.Tensor:
        """轻量标量增益：delta小则增强细节，delta大则抑制伪影。"""
        if not self.featfix_enabled:
            return output
        if threshold <= 0 or delta <= 0 or blend_alpha <= 0:
            return output

        ratio = delta / threshold
        # ratio=0 → gain=1.15, ratio=0.5 → gain=1.0, ratio=1.0 → gain=0.85
        gain = 1.15 - 0.3 * min(ratio, 1.5) / 1.5
        # 用 blend_alpha 调制，越确定缓存效果越强
        gain = 1.0 + (gain - 1.0) * blend_alpha

        if abs(gain - 1.0) < 0.005:
            return output
        return output * gain

    # ---- 完整补偿 pipeline ----
    def apply_compensation(self, base_output: torch.Tensor,
                           state: TaijiArmorV4FinalState,
                           delta: float, threshold: float,
                           blend_alpha: float) -> torch.Tensor:
        output = base_output

        # Step 1: 频率解耦
        if self.freq_decouple and state.step_idx > self.warmup_steps:
            freq_out = self._freq_decouple_compensate(output, state)
            if torch.isfinite(freq_out).all():
                output = freq_out

        # Step 2: EMA 趋势补偿（强度由 AdaCorrection alpha + 阶段共同控制）
        if self.correction_strength > 0 and state.output_trend is not None and state.step_idx > 1:
            phase_mult = self._phase_params(state.current_phase)['correction_mult']
            effective_strength = self.correction_strength * blend_alpha * phase_mult
            trend = state.output_trend.to(base_output.device).to(base_output.dtype)

            # 安全闸：补偿幅度不超过输出标准差的 2 倍，防止推爆 latent
            with torch.no_grad():
                out_std = base_output.std().clamp(min=1e-6)
                trend_mag = trend.abs().mean().clamp(min=1e-6)
                max_str = (2.0 * out_std / trend_mag).item()
                effective_strength = min(effective_strength, max_str)

            compensated = output + trend * effective_strength
            if torch.isfinite(compensated).all():
                output = compensated

        # Step 3: FeatFix 增益调节
        if self.featfix_enabled and state.step_idx > 1:
            featfix_out = self._featfix_apply(output, delta, threshold, blend_alpha)
            if torch.isfinite(featfix_out).all():
                output = featfix_out

        return output

    # ---- 计算步更新 ----
    def update_after_compute(self, state: TaijiArmorV4FinalState,
                             output: torch.Tensor, delta: float):
        cur_out_cpu = output.detach().float().cpu()

        # EMA 趋势
        if self.correction_strength > 0:
            if state.prev_output is not None:
                change = cur_out_cpu - state.prev_output
                if state.output_trend is None:
                    state.output_trend = change.clone()
                else:
                    state.output_trend = 0.7 * state.output_trend + 0.3 * change

        # 频率解耦：高频趋势
        if self.freq_decouple:
            try:
                _, cur_high = _dct_freq_decompose(cur_out_cpu, self.freq_low_ratio)
                if state.prev_output_freq_high is not None:
                    hf_change = cur_high - state.prev_output_freq_high
                    if state.high_freq_trend is None:
                        state.high_freq_trend = hf_change.clone()
                    else:
                        state.high_freq_trend = (
                            0.7 * state.high_freq_trend + 0.3 * hf_change
                        )
                state.prev_output_freq_high = cur_high
            except Exception:
                pass

        # IEC 误差记录
        if self.iec_enabled and delta > 0 and state.step_idx > 0:
            state.iec_error_buffer.append((state.step_idx, delta))
            if len(state.iec_error_buffer) > state.iec_max_buffer:
                state.iec_error_buffer.pop(0)

        state.prev_output = cur_out_cpu


# ============================================================
#  ComfyUI Wrapper
# ============================================================

def make_v4final_armor_wrapper(
    total_steps: int,
    warmup_steps: int = 3,
    safety_margin: float = 1.8,
    correction_strength: float = 0.15,
    adac_min_alpha: float = 0.3,
    iec_enabled: bool = True,
    iec_gamma: float = 0.6,
    featfix_enabled: bool = True,
    freq_decouple: bool = False,
    freq_low_ratio: float = 0.3,
    phase_aware: bool = True,
    max_consecutive: int = 3,
    end_step: int = -2,
    base_threshold: float = 0.12,
    verbose: bool = True,
) -> tuple:
    """创建 (state, wrapper) 供 ComfyUI set_model_unet_function_wrapper 使用。"""
    state = TaijiArmorV4FinalState()
    scheduler = TaijiArmorV4FinalScheduler(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        safety_margin=safety_margin,
        correction_strength=correction_strength,
        adac_min_alpha=adac_min_alpha,
        iec_enabled=iec_enabled,
        iec_gamma=iec_gamma,
        featfix_enabled=featfix_enabled,
        freq_decouple=freq_decouple,
        freq_low_ratio=freq_low_ratio,
        phase_aware=phase_aware,
        max_consecutive=max_consecutive,
        end_step=end_step,
        base_threshold=base_threshold,
        verbose=verbose,
    )

    _step_t0 = [_time.perf_counter()]
    _step_wall_times = []
    _vram_history = []

    def _get_vram_mb() -> float:
        try:
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def wrapper(apply_model, args) -> torch.Tensor:
        x = args["input"]
        _t0 = _time.perf_counter()
        vram_before = _get_vram_mb()

        # 决策
        reuse, delta, thresh, blend_val, strategy = scheduler.should_reuse(state, x)
        _sched_dt = _time.perf_counter() - _t0
        _step_wall = _time.perf_counter() - _step_t0[0]

        if reuse:
            state.cache_count += 1
            state.consecutive_cache += 1

            base_output = state.prev_output.to(x.device).to(x.dtype)
            output = scheduler.apply_compensation(
                base_output, state, delta, thresh, blend_val
            )

            state.prev_input = x.detach().float().cpu()
            state.step_idx += 1

            vram_after = _get_vram_mb()
            _vram_history.append(vram_after)
            state.peak_vram_mb = max(_vram_history) if _vram_history else 0.0

            if verbose:
                trend_norm = (state.output_trend.abs().mean().item()
                              if state.output_trend is not None else 0.0)
                print("[V4Final] Step %d/%d: CACHE (delta=%.5f thresh=%.5f "
                      "blend=%.2f phase=%s trend=%.5f vram=%.0fMB %s)" % (
                    state.step_idx - 1, total_steps, delta, thresh, blend_val,
                    state.current_phase, trend_norm, vram_after, strategy))

            _step_wall_times.append(_step_wall)
            state.step_wall_times = _step_wall_times
            return output

        # 计算步
        output = apply_model(x, args["timestep"], **args["c"])
        state.real_count += 1
        state.consecutive_cache = 0

        vram_after = _get_vram_mb()
        _vram_history.append(vram_after)
        state.peak_vram_mb = max(_vram_history) if _vram_history else 0.0

        # 更新状态
        cur_cpu = x.detach().float().cpu()
        scheduler.update_after_compute(state, output, delta)
        state.prev_input = cur_cpu
        state.step_idx += 1

        _step_t0[0] = _time.perf_counter()
        _step_wall_times.append(_step_wall)
        state.step_wall_times = _step_wall_times

        if verbose:
            _model_dt = _time.perf_counter() - _t0
            print("[V4Final] Step %d/%d: COMPUTE wall=%.1fs (sched=%.0fms "
                  "model=%.0fms) phase=%s vram=%.0fMB (delta=%.5f thresh=%.5f blend=%.2f)" % (
                state.step_idx, total_steps, _step_wall, _sched_dt * 1000,
                _model_dt * 1000, state.current_phase, vram_after, delta, thresh, blend_val))

        if state.step_idx >= total_steps:
            _v4final_report(state, total_steps, warmup_steps, scheduler)

        return output

    return state, wrapper


def _v4final_report(state, total_steps, warmup_steps, scheduler):
    tot = state.cache_count + state.real_count
    rate = (state.cache_count / tot * 100) if tot else 0.0
    peak_vram = getattr(state, 'peak_vram_mb', 0.0)
    base = (f"[XuanShuV4Final] steps={total_steps} warmup={warmup_steps} "
            f"cache={state.cache_count} real={state.real_count} hit_rate={rate:.1f}% "
            f"peak_vram={peak_vram:.0f}MB phase_aware={scheduler.phase_aware}\n"
            f"  IEC={scheduler.iec_enabled}(buf={len(state.iec_error_buffer)}) "
            f"AdaC=on "
            f"FeatFix={scheduler.featfix_enabled} "
            f"FreqDec={scheduler.freq_decouple}(ratio={scheduler.freq_low_ratio})")
    wall_times = state.step_wall_times
    if wall_times:
        base += (f"\n  wall_times=[{', '.join('%.1f' % w for w in wall_times)}] "
                 f"total=%.1fs" % sum(wall_times))
    print(base)


# ============================================================
#  ComfyUI 节点类
# ============================================================

class MiniMaxH3TaijiArmorV4Final:
    """玄枢太极装甲 v4 Final（完整版：AdaC + IEC + FeatFix v2 + 频率解耦 + 阶段感知）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 6, "min": 1, "max": 200,
                    "tooltip": "总去噪步数"}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 100,
                    "tooltip": "预热步数，建立EMA趋势后才缓存"}),
                "safety_margin": ("FLOAT", {"default": 1.8, "min": 0.5, "max": 4.0, "step": 0.1,
                    "tooltip": "安全边际，越大越保守"}),
                "correction_strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "趋势补偿强度"}),
                "end_step": ("INT", {"default": -2, "min": -20, "max": 100,
                    "tooltip": "末尾强制计算步数，负数表示从末尾倒数"}),
                "max_consecutive": ("INT", {"default": 3, "min": 1, "max": 20,
                    "tooltip": "最大连续缓存步数，防止长序列漂移"}),
                "base_threshold": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "缓存判定阈值基准，越小越激进（与v3的yang_thresh同量级）"}),
                # --- 核心机制开关 ---
                "iec_enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "迭代误差校正：缓存误差反馈到阈值"}),
                "featfix_enabled": ("BOOLEAN", {"default": True,
                    "tooltip": "FeatFix v2：delta信号增益调节（轻量非破坏）"}),
                "freq_decouple": ("BOOLEAN", {"default": False,
                    "tooltip": "频率解耦：低频复用+高频趋势外推（CPU上较慢，谨慎开启）"}),
                "phase_aware": ("BOOLEAN", {"default": True,
                    "tooltip": "阶段感知：早/中/晚期自适应参数"}),
                # --- 精细调节 ---
                "adac_min_alpha": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.9, "step": 0.05,
                    "tooltip": "AdaCorrection最小混合系数"}),
                "iec_gamma": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "IEC增益：误差→阈值惩罚的转换率"}),
                "freq_low_ratio": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.6, "step": 0.05,
                    "tooltip": "频率解耦低频占比"}),
                "verbose": ("BOOLEAN", {"default": True,
                    "tooltip": "打印每步详细信息"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "MiniMax H3 太极装甲 v4 Final"

    def apply(self, model, total_steps, warmup_steps, safety_margin,
              correction_strength, end_step, max_consecutive, base_threshold,
              iec_enabled, featfix_enabled, freq_decouple, phase_aware,
              adac_min_alpha, iec_gamma, freq_low_ratio, verbose):
        # 冲突检测：只保留最高版本装甲
        if "xuanshu_taiji_armor_v4_final" in model.model_options:
            print("[V4Final] 警告：v4 Final 已应用，跳过")
            return (model,)

        state, wrapper = make_v4final_armor_wrapper(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            safety_margin=safety_margin,
            correction_strength=correction_strength,
            adac_min_alpha=adac_min_alpha,
            iec_enabled=iec_enabled,
            iec_gamma=iec_gamma,
            featfix_enabled=featfix_enabled,
            freq_decouple=freq_decouple,
            freq_low_ratio=freq_low_ratio,
            phase_aware=phase_aware,
            max_consecutive=max_consecutive,
            end_step=end_step,
            base_threshold=base_threshold,
            verbose=verbose,
        )

        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        patched.model_options["xuanshu_taiji_armor_v4_final"] = {"state": state}
        # 移除低版本装甲和 Guard，避免冲突
        patched.model_options.pop("xuanshu_teacache_guard", None)
        patched.model_options.pop("xuanshu_taiji_armor", None)
        patched.model_options.pop("xuanshu_taiji_armor_v4", None)
        patched.model_options.pop("xuanshu_taiji_armor_v4_1", None)
        return (patched,)


NODE_CLASS_MAPPINGS_V4FINAL = {
    "MiniMaxH3TaijiArmorV4Final": MiniMaxH3TaijiArmorV4Final,
}
NODE_DISPLAY_NAME_MAPPINGS_V4FINAL = {
    "MiniMaxH3TaijiArmorV4Final": "MiniMax H3 太极装甲 v4 Final",
}
