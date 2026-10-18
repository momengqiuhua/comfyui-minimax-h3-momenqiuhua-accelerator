"""XuanShu H3 振荡阻尼节点 (PSS式) — 抑制画质指标低频振荡。

电网原型：电力系统稳定器（PSS）——抑制系统低频振荡（功率来回摆动）。
逻辑：检测画质/缓存指标在某几步好某几步差、发散收敛反复的振荡趋势时，
     施加阻尼，抑制发散，让输出稳定收敛。

v2 重做（2026-08-21）：根治空转（修正 err 为输入vs输入语义）。
v3（2026-08-21）：加护栏（突变保护 + 连续缓存上限），I2V 下实测 PSS 崩坏
  （猫多重轮廓/色彩分离），根因是 I2V latent delta 突变时缓存失误；对齐 AGC
  的 surge_factor 突变保护 + max_consecutive 连续缓存护栏后根治。

自包含：仅 numpy/torch + 标准库。
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import torch


@dataclasses.dataclass
class PSSState:
    step_idx: int = 0
    prev_input: Optional[torch.Tensor] = None
    prev_output: Optional[torch.Tensor] = None
    consecutive_reuse_count: int = 0
    reuse_count: int = 0
    real_count: int = 0
    error_history: list = dataclasses.field(default_factory=list)
    damping_applied: int = 0
    oscillation_detected: bool = False
    hit_steps: list = dataclasses.field(default_factory=list)
    real_steps: list = dataclasses.field(default_factory=list)

    def reset(self) -> None:
        self.step_idx = 0
        self.prev_input = None
        self.prev_output = None
        self.consecutive_reuse_count = 0
        self.reuse_count = 0
        self.real_count = 0
        self.error_history = []
        self.damping_applied = 0
        self.oscillation_detected = False
        self.hit_steps = []
        self.real_steps = []


def rel_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    diff = (current.to(previous.dtype) - previous).abs().mean()
    ref = previous.abs().mean().clamp(min=1e-8)
    return (diff / ref).item()


def detect_oscillation(errors: list, window: int = 6) -> bool:
    """检测误差序列正负交替（低频振荡先兆）。"""
    if len(errors) < window:
        return False
    tail = errors[-window:]
    signs = []
    for i in range(1, len(tail)):
        d = tail[i] - tail[i - 1]
        signs.append(1 if d > 1e-6 else (-1 if d < -1e-6 else 0))
    sign_changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1] and signs[i] != 0)
    return sign_changes >= window - 2


class PSSScheduler:
    def __init__(self, total_steps, warmup_steps, osc_window,
                 damping_strength, thresh, max_consecutive, surge_factor, verbose):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.osc_window = osc_window
        self.damping_strength = damping_strength
        self.thresh = thresh
        self.max_consecutive = max_consecutive
        self.surge_factor = surge_factor
        self.verbose = verbose

    def should_reuse(self, state: PSSState, x: torch.Tensor) -> tuple:
        step = state.step_idx
        if state.prev_output is None or state.prev_input is None:
            return (False, 0.0, self.thresh, 0.0, "no_ref")

        # 误差信号：当前输入相对上一输入的变化
        err = rel_l1(x.detach().cpu(), state.prev_input)
        state.error_history.append(err)

        # warmup 全算
        if step < self.warmup_steps:
            return (False, err, self.thresh, 0.0, "warmup(%d<%d)" % (step, self.warmup_steps))

        # 突变保护：delta 远超阈值 → 强制重算（防缓存失误）
        if err > self.thresh * self.surge_factor:
            return (False, err, self.thresh, 0.0, "surge_guard")
        # 连续缓存护栏：达到上限强制重算（防 latent 漂移崩坏）
        if state.consecutive_reuse_count >= self.max_consecutive:
            return (False, err, self.thresh, 0.0, "guard_max_consecutive")

        # 振荡检测 + 阻尼
        if detect_oscillation(state.error_history, self.osc_window):
            state.oscillation_detected = True
            eff_thresh = self.thresh * (1.0 - self.damping_strength)
            if self.verbose and state.step_idx % self.osc_window == 0:
                print("[PSS] Step %d: 检测到振荡 → 施加阻尼 thresh=%.4f" % (step, eff_thresh))
        else:
            state.oscillation_detected = False
            eff_thresh = self.thresh

        if err < eff_thresh:
            return (True, err, eff_thresh, 0.0, "hit")
        return (False, err, eff_thresh, 0.0, "miss")


def make_pss_wrapper(
    state: PSSState,
    total_steps: int,
    warmup_steps: int,
    osc_window: int,
    damping_strength: float,
    thresh: float,
    max_consecutive: int,
    surge_factor: float,
    verbose: bool,
) -> Callable[..., torch.Tensor]:
    scheduler = PSSScheduler(total_steps, warmup_steps, osc_window,
                             damping_strength, thresh, max_consecutive,
                             surge_factor, verbose)

    def wrapper(apply_model, args) -> torch.Tensor:
        x = args["input"]
        reuse, err, eff_thresh, conf, reason = scheduler.should_reuse(state, x)

        if reuse:
            state.reuse_count += 1
            state.hit_steps.append(state.step_idx)
            state.step_idx += 1
            state.consecutive_reuse_count += 1
            if verbose:
                print("[PSS] Step %d/%d: CACHE (err=%.4f thresh=%.4f)" % (
                    state.step_idx - 1, total_steps, err, eff_thresh))
            return state.prev_output.to(x.device)

        output = apply_model(x, args["timestep"], **args["c"])
        state.prev_input = x.detach().cpu()
        state.prev_output = output.detach().cpu()
        state.consecutive_reuse_count = 0
        state.real_count += 1
        state.real_steps.append(state.step_idx)
        state.step_idx += 1
        if verbose:
            print("[PSS] Step %d/%d: COMPUTE (err=%.4f thresh=%.4f)" % (
                state.step_idx - 1, total_steps, err, eff_thresh))
        if state.step_idx >= total_steps:
            tot = state.reuse_count + state.real_count
            rate = (state.reuse_count / tot * 100) if tot else 0.0
            print("[XuanShuPSS] steps=%d warmup=%d osc=%s damp=%d reuse=%d real=%d hit=%.1f%%" % (
                total_steps, warmup_steps, state.oscillation_detected,
                state.damping_applied, state.reuse_count, state.real_count, rate))
        return output

    return wrapper


class MiniMaxH3PSS:
    """玄枢振荡阻尼节点（PSS 式）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 8, "min": 1, "max": 200}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 100}),
                "osc_window": ("INT", {"default": 6, "min": 3, "max": 20,
                                       "tooltip": "振荡检测窗口"}),
                "damping_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                               "tooltip": "阻尼强度：检测到振荡时阈值收紧比例"}),
                "thresh": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.5, "step": 0.01}),
                "max_consecutive": ("INT", {"default": 3, "min": 1, "max": 20,
                                        "tooltip": "连续缓存上限（护栏）：达到后强制重算，防漂移崩坏"}),
                "surge_factor": ("FLOAT", {"default": 4.0, "min": 2.0, "max": 10.0, "step": 0.5,
                                        "tooltip": "突变保护：delta 超当前阈值此倍数则强制重算"}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "MiniMax H3 振荡阻尼 (PSS)"

    def apply(self, model, total_steps, warmup_steps, osc_window,
              damping_strength, thresh, max_consecutive, surge_factor, verbose):
        if "xuanshu_taiji_armor" in model.model_options or \
           "xuanshu_teacache_guard" in model.model_options or \
           "xuanshu_agc" in model.model_options or \
           "xuanshu_selfheal" in model.model_options or \
           "xuanshu_vsg" in model.model_options:
            print("[PSS] 检测到其他缓存节点已挂载，跳过（互斥）")
            return (model,)
        state = PSSState()
        patched = model.clone()
        patched.set_model_unet_function_wrapper(
            make_pss_wrapper(state, total_steps, warmup_steps, osc_window,
                             damping_strength, thresh, max_consecutive,
                             surge_factor, verbose))
        patched.model_options["xuanshu_pss"] = {"state": state}
        return (patched,)


NODE_CLASS_MAPPINGS_PSS = {
    "MiniMaxH3PSS": MiniMaxH3PSS,
}
NODE_DISPLAY_NAME_MAPPINGS_PSS = {
    "MiniMaxH3PSS": "MiniMax H3 振荡阻尼 (PSS)",
}
