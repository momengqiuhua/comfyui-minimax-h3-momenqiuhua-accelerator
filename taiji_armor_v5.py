"""XuanShu H3 太极装甲 v5 — 末梦秋花v5：极简截断漂移版。

综合 v3s1 + v4 Final，砍掉冗余：
- 保留：l1 delta + AdaCorrection alpha 半跳半算 + EMA趋势补偿 + FeatFix轻量增益
        + IEC误差反馈 + phase_aware阶段感知 + max_consecutive截断漂移 + end_step末尾保护
        + chunk_size分块delta（batch>1保守）
- 砍掉：五策略择优、numpy多项式拟合、VRAM水位门、护体真元fallback、
        置信度衰减、error_budget熔断、freq_decouple频率解耦
"""

import time
import torch
from typing import Optional


# ============================================================
# 状态
# ============================================================
class TaijiArmorV5State:
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

    def reset(self):
        self.prev_input = None
        self.prev_output = None
        self.output_trend = None
        self.consecutive_cache = 0
        self.iec_error_buffer = []


# ============================================================
# 调度器
# ============================================================
class TaijiArmorV5Scheduler:
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
        self.verbose = verbose

    def _phase(self, step: int) -> str:
        p = step / max(1, self.total_steps)
        if p < 0.3: return 'early'
        elif p < 0.75: return 'mid'
        return 'late'

    def _phase_mult(self, phase: str) -> float:
        return {'early': 1.4, 'mid': 1.0, 'late': 1.2}.get(phase, 1.0)

    def _delta(self, cur: torch.Tensor, prev: torch.Tensor) -> float:
        b = cur.shape[0]
        if b <= 1 or b <= self.chunk_size:
            return (cur - prev).abs().mean().item()
        blocks = []
        for i in range(0, b, self.chunk_size):
            blocks.append((cur[i:i+self.chunk_size] - prev[i:i+self.chunk_size]).abs().mean().item())
        return max(blocks)

    def _iec_penalty(self, state) -> float:
        if not self.iec_enabled or len(state.iec_error_buffer) < 2:
            return 0.0
        errors = torch.tensor([e for _, e in state.iec_error_buffer], dtype=torch.float32)
        w = torch.exp(torch.linspace(-2.0, 0.0, len(errors)))
        w = w / w.sum()
        return (errors * w).sum().item() * self.iec_gamma

    def should_reuse(self, state, x: torch.Tensor):
        step = state.step_idx

        # 形状变化重置
        if state.prev_input is not None and tuple(x.shape) != tuple(state.prev_input.shape):
            if self.verbose:
                print(f"[V5] 形状变化 → 重置")
            state.reset()
            return (False, 0.0, 0.0, 0.0, "shape_reset")

        if state.prev_output is None or state.prev_input is None:
            return (False, 0.0, 0.0, 0.0, "no_ref")

        state.current_phase = self._phase(step)

        # end_step 保护
        eff_end = self.end_step if self.end_step >= 0 else self.total_steps + self.end_step
        if step >= eff_end:
            return (False, 0.0, 0.0, 0.0, "end_protect")

        # warmup
        if step < self.warmup_steps:
            return (False, 0.0, 0.0, 0.0, f"warmup({step}<{self.warmup_steps})")

        # delta
        cur_cpu = x.detach().float().cpu()
        delta = self._delta(cur_cpu, state.prev_input)

        # 有效阈值
        thresh = max(self.base_threshold, 1e-6)
        iec_pen = self._iec_penalty(state)
        eff_thresh = max(
            thresh * self.safety_margin * self._phase_mult(state.current_phase) - iec_pen,
            thresh * 0.2
        )

        # AdaCorrection alpha
        ratio = delta / eff_thresh if eff_thresh > 0 else 10.0
        if ratio <= 1.0:
            alpha = 1.0 - (1.0 - self.adac_min_alpha) * ratio
        else:
            alpha = self.adac_min_alpha

        should_cache = alpha > 0.5

        # max_consecutive 截断
        if should_cache and state.consecutive_cache >= self.max_consecutive:
            should_cache = False
            alpha = 0.0
            reason = f"max_consecutive({state.consecutive_cache})"
        else:
            reason = f"cache(a={alpha:.2f})" if should_cache else f"compute(d={delta:.4f}>=t={eff_thresh:.4f})"

        return (should_cache, delta, eff_thresh, alpha, reason)

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

    def update_after_compute(self, state, output_cpu, was_cached):
        if self.correction_strength > 0 and state.prev_output is not None:
            change = output_cpu - state.prev_output
            if state.output_trend is None:
                state.output_trend = change.clone()
            else:
                state.output_trend = 0.7 * state.output_trend + 0.3 * change
        # IEC
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
def make_v5_wrapper(total_steps, warmup_steps, safety_margin, correction_strength,
                    end_step, max_consecutive, base_threshold, adac_min_alpha,
                    iec_enabled, iec_gamma, chunk_size, verbose):
    state = TaijiArmorV5State()
    sched = TaijiArmorV5Scheduler(
        total_steps, warmup_steps, safety_margin, correction_strength,
        end_step, max_consecutive, base_threshold, adac_min_alpha,
        iec_enabled, iec_gamma, chunk_size, verbose
    )

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
                print(f"[V5] step {state.step_idx-1}/{total_steps} CACHE  d={delta:.5f} t={thresh:.5f} a={alpha:.2f} {reason}")
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
            print(f"[V5] step {state.step_idx-1}/{total_steps} COMPUTE {dt:.1f}s {reason}")
        return output

    return state, wrapper


# ============================================================
# ComfyUI 节点
# ============================================================
class MiniMaxH3TaijiArmorV5:
    """玄枢太极装甲 v5（末梦秋花v5）：极简截断漂移版。

    l1 delta + AdaCorrection alpha + EMA趋势 + FeatFix + IEC反馈 +
    phase_aware + max_consecutive截断 + end_step保护。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 6, "min": 1, "max": 200,
                    "tooltip": "总去噪步数"}),
                "warmup_steps": ("INT", {"default": 2, "min": 0, "max": 100,
                    "tooltip": "探针步数：前N步全算"}),
                "safety_margin": ("FLOAT", {"default": 1.6, "min": 0.5, "max": 4.0, "step": 0.1,
                    "tooltip": "安全边际：阈值×此值，越大越保守"}),
                "correction_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "EMA趋势补偿强度"}),
                "end_step": ("INT", {"default": -2, "min": -20, "max": 100,
                    "tooltip": "末尾保护步数，负数倒数"}),
                "max_consecutive": ("INT", {"default": 3, "min": 1, "max": 10,
                    "tooltip": "连续缓存上限：截断漂移累积"}),
                "base_threshold": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 0.5, "step": 0.01,
                    "tooltip": "缓存判定基准阈值"}),
                "iec_gamma": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "IEC误差反馈增益"}),
                "chunk_size": ("INT", {"default": 2, "min": 1, "max": 8,
                    "tooltip": "batch分块delta（保守取最大块）"}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "玄枢太极装甲 v5（末梦秋花v5）"

    def apply(self, model, total_steps, warmup_steps, safety_margin, correction_strength,
              end_step, max_consecutive, base_threshold, iec_gamma, chunk_size, verbose):
        state, wrapper = make_v5_wrapper(
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
            verbose=verbose,
        )
        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        patched.model_options["xuanshu_taiji_armor_v5"] = {"state": state}
        for k in ["xuanshu_teacache_guard", "xuanshu_taiji_armor",
                  "xuanshu_taiji_armor_v4", "xuanshu_taiji_armor_v4_1",
                  "xuanshu_taiji_armor_v4_final", "xuanshu_taiji_armor_v4s"]:
            patched.model_options.pop(k, None)
        return (patched,)


NODE_CLASS_MAPPINGS_V5 = {"MiniMaxH3TaijiArmorV5": MiniMaxH3TaijiArmorV5}
NODE_DISPLAY_NAME_MAPPINGS_V5 = {"MiniMaxH3TaijiArmorV5": "玄枢太极装甲 v5（末梦秋花v5）"}
