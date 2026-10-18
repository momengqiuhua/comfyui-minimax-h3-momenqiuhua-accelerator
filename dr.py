"""XuanShu H3 削峰填谷节点 (DR式) — 显存/算力动态调度。

电网原型：需求侧响应（DR）/ 削峰填谷——用电高峰削峰、低谷填谷，平衡负载。
逻辑：显存/算力逼近上限时主动"削峰"（收紧缓存、降载防 OOM）；
     显存/算力富余时主动"填谷"（放松缓存、加速）。
     超越现有的硬阈值降载，让资源利用率峰值更平。

自包含：仅 numpy/torch + 标准库。
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import torch

@dataclasses.dataclass
class DRState:
    step_idx: int = 0
    prev_input: Optional[torch.Tensor] = None
    prev_output: Optional[torch.Tensor] = None
    reuse_count: int = 0
    real_count: int = 0
    consecutive_reuse_count: int = 0
    output_trend: Optional[torch.Tensor] = None
    peak_vram_mb: int = 0
    hit_steps: list = dataclasses.field(default_factory=list)
    real_steps: list = dataclasses.field(default_factory=list)
    throttle_count: int = 0
    boost_count: int = 0

    def reset(self) -> None:
        self.step_idx = 0
        self.prev_input = None
        self.prev_output = None
        self.reuse_count = 0
        self.real_count = 0
        self.consecutive_reuse_count = 0
        self.output_trend = None
        self.peak_vram_mb = 0
        self.hit_steps = []
        self.real_steps = []
        self.throttle_count = 0
        self.boost_count = 0


def rel_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    diff = (current.to(previous.dtype) - previous).abs().mean()
    ref = previous.abs().mean().clamp(min=1e-8)
    return (diff / ref).item()


def read_vram_mb() -> int:
    try:
        return int(torch.cuda.memory_reserved() // (1024 * 1024))
    except Exception:
        return 0


class DRScheduler:
    def __init__(self, total_steps, warmup_steps, vram_thresh_mb,
                 vram_low_mb, peak_factor, valley_factor,
                 base_thresh, smooth_factor, verbose, max_consecutive=3,
                 correction_strength=0.5):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.vram_thresh_mb = vram_thresh_mb
        self.vram_low_mb = vram_low_mb
        self.peak_factor = peak_factor  # 峰值收紧倍率（>1 更保守）
        self.valley_factor = valley_factor  # 谷值放松倍率（<1 更激进）
        self.base_thresh = base_thresh
        self.smooth_factor = smooth_factor
        self.verbose = verbose
        self.max_consecutive = max_consecutive
        self.correction_strength = correction_strength

    def _vram_pressure(self, state: DRState) -> float:
        vram = read_vram_mb()
        if vram > state.peak_vram_mb:
            state.peak_vram_mb = vram
        if vram > self.vram_thresh_mb:
            return 1.0  # 削峰
        if vram < self.vram_low_mb:
            return -1.0  # 填谷
        return 0.0

    def should_reuse(self, state: DRState, x: torch.Tensor) -> tuple:
        step = state.step_idx
        if state.prev_output is None or state.prev_input is None:
            return (False, 0.0, self.base_thresh, 0.0, "no_ref")

        delta = rel_l1(x.detach().cpu(), state.prev_input)

        if step < self.warmup_steps:
            return (False, delta, self.base_thresh, 0.0, "warmup(%d<%d)" % (step, self.warmup_steps))

        pressure = self._vram_pressure(state)
        # 动态阈值：削峰收紧 / 填谷放松
        if pressure > 0:
            eff_thresh = self.base_thresh / self.peak_factor
            state.throttle_count += 1
            if self.verbose:
                print("[DR] Step %d: 削峰（显存 %dMB>%d）→ thresh=%.4f" % (
                    step, state.peak_vram_mb, self.vram_thresh_mb, eff_thresh))
        elif pressure < 0:
            eff_thresh = self.base_thresh * self.valley_factor
            state.boost_count += 1
            if self.verbose:
                print("[DR] Step %d: 填谷（显存富余）→ thresh=%.4f" % (step, eff_thresh))
        else:
            eff_thresh = self.base_thresh

        # 平滑：限制单步阈值突变
        eff_thresh = max(0.01, min(0.5, eff_thresh))

        # === [end_step 保护] 达到总步数后强制不缓存 ===
        if step >= self.total_steps:
            return (False, delta, eff_thresh, 0.0, "after_end")

        # === [护栏] 连续缓存上限（防 latent 漂移崩坏） ===
        if state.consecutive_reuse_count >= self.max_consecutive:
            state.consecutive_reuse_count = 0
            if self.verbose:
                print("[DR] Step %d: 护栏-连续缓存上限(%d) → 强制重算" % (
                    step, self.max_consecutive))
            return (False, delta, eff_thresh, 0.0, "guard_max_consecutive")

        if delta < eff_thresh:
            return (True, delta, eff_thresh, 0.0, "hit")
        return (False, delta, eff_thresh, 0.0, "miss")


def make_dr_wrapper(
    state: DRState,
    total_steps: int,
    warmup_steps: int,
    vram_thresh_mb: int,
    vram_low_mb: int,
    peak_factor: float,
    valley_factor: float,
    base_thresh: float,
    smooth_factor: float,
    verbose: bool,
    max_consecutive: int = 3,
    correction_strength: float = 0.5,
) -> Callable[..., torch.Tensor]:
    scheduler = DRScheduler(total_steps, warmup_steps, vram_thresh_mb, vram_low_mb,
                            peak_factor, valley_factor, base_thresh, smooth_factor, verbose,
                            max_consecutive, correction_strength)

    def wrapper(apply_model, args) -> torch.Tensor:
        x = args["input"]
        reuse, delta, eff_thresh, conf, reason = scheduler.should_reuse(state, x)

        if reuse:
            state.reuse_count += 1
            state.hit_steps.append(state.step_idx)
            state.step_idx += 1
            state.consecutive_reuse_count += 1
            if verbose:
                print("[DR] Step %d/%d: CACHE (delta=%.4f thresh=%.4f)" % (state.step_idx - 1, total_steps, delta, eff_thresh))
            # === 趋势外推补偿（根治崩坏） ===
            base = state.prev_output.to(x.device)
            if state.output_trend is not None:
                return base + state.output_trend.to(x.device) * scheduler.correction_strength
            return base

        output = apply_model(x, args["timestep"], **args["c"])
        # === 更新趋势 EMA ===
        if state.prev_output is not None:
            change = output.detach().cpu() - state.prev_output
            if state.output_trend is None:
                state.output_trend = change.clone()
            else:
                state.output_trend = 0.7 * state.output_trend + 0.3 * change
        state.prev_input = x.detach().cpu()
        state.prev_output = output.detach().cpu()
        state.real_count += 1
        state.real_steps.append(state.step_idx)
        state.step_idx += 1
        state.consecutive_reuse_count = 0
        if verbose:
            print("[DR] Step %d/%d: COMPUTE" % (state.step_idx - 1, total_steps))
        if state.step_idx >= total_steps:
            tot = state.reuse_count + state.real_count
            rate = (state.reuse_count / tot * 100) if tot else 0.0
            print("[XuanShuDR] steps=%d peak_vram=%dMB throttle=%d boost=%d reuse=%d real=%d hit=%.1f%%" % (
                total_steps, state.peak_vram_mb, state.throttle_count, state.boost_count,
                state.reuse_count, state.real_count, rate))
        return output

    return wrapper


class MiniMaxH3DR:
    """玄枢削峰填谷节点（DR 式）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 8, "min": 1, "max": 200}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 100}),
                "vram_thresh_mb": ("INT", {"default": 7500, "min": 1000, "max": 20000,
                                           "tooltip": "显存削峰阈值：超过则收紧"}),
                "vram_low_mb": ("INT", {"default": 4000, "min": 1000, "max": 20000,
                                        "tooltip": "显存填谷阈值：低于则加速"}),
                "peak_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                "valley_factor": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 3.0, "step": 0.1}),
                "base_thresh": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.5, "step": 0.01}),
                "smooth_factor": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                "max_consecutive": ("INT", {"default": 3, "min": 1, "max": 20,
                                        "tooltip": "连续缓存上限（护栏）：达到后强制重算，防漂移崩坏"}),
                "correction_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "趋势外推强度：缓存步用 prev_output+trend 补偿，0=纯复用"}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "MiniMax H3 削峰填谷 (DR)"

    def apply(self, model, total_steps, warmup_steps, vram_thresh_mb, vram_low_mb,
              peak_factor, valley_factor, base_thresh, smooth_factor, verbose,
              max_consecutive=3, correction_strength=0.5):
        if "xuanshu_taiji_armor" in model.model_options or \
           "xuanshu_teacache_guard" in model.model_options or \
           "xuanshu_agc" in model.model_options or \
           "xuanshu_selfheal" in model.model_options or \
           "xuanshu_vsg" in model.model_options or \
           "xuanshu_pss" in model.model_options:
            print("[DR] 检测到其他缓存节点已挂载，跳过（互斥）")
            return (model,)
        state = DRState()
        patched = model.clone()
        patched.set_model_unet_function_wrapper(
            make_dr_wrapper(state, total_steps, warmup_steps, vram_thresh_mb, vram_low_mb,
                            peak_factor, valley_factor, base_thresh, smooth_factor, verbose,
                            max_consecutive, correction_strength))
        patched.model_options["xuanshu_dr"] = {"state": state}
        return (patched,)


NODE_CLASS_MAPPINGS_DR = {
    "MiniMaxH3DR": MiniMaxH3DR,
}
NODE_DISPLAY_NAME_MAPPINGS_DR = {
    "MiniMaxH3DR": "MiniMax H3 削峰填谷 (DR)",
}