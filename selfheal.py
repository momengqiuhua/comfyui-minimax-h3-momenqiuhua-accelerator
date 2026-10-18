"""XuanShu H3 自愈重构节点 (自愈电网式) — 崩坏隔离 + 局部重采样恢复。

电网原型：自愈电网 + 网络重构。
逻辑：采样中检测到崩坏（异常 delta 突增/质量指标越限）时，
     不整条重跑，而是记录最近健康 checkpoint，隔离崩坏段，
     从最近健康 latent 用更保守参数重采样后半段。

自包含：仅 numpy/torch + 标准库。
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import torch

@dataclasses.dataclass
class SelfHealState:
    step_idx: int = 0
    prev_input: Optional[torch.Tensor] = None
    prev_output: Optional[torch.Tensor] = None
    reuse_count: int = 0
    real_count: int = 0
    consecutive_reuse_count: int = 0
    output_trend: Optional[torch.Tensor] = None
    # 自愈状态
    heal_triggered: bool = False
    heal_count: int = 0
    last_healthy_step: int = 0
    last_healthy_output: Optional[torch.Tensor] = None
    anomaly_count: int = 0
    # 诊断
    delta_history: list = dataclasses.field(default_factory=list)
    hit_steps: list = dataclasses.field(default_factory=list)
    real_steps: list = dataclasses.field(default_factory=list)

    def reset(self) -> None:
        self.step_idx = 0
        self.prev_input = None
        self.prev_output = None
        self.reuse_count = 0
        self.real_count = 0
        self.consecutive_reuse_count = 0
        self.output_trend = None
        self.heal_triggered = False
        self.heal_count = 0
        self.last_healthy_step = 0
        self.last_healthy_output = None
        self.anomaly_count = 0
        self.delta_history = []
        self.hit_steps = []
        self.real_steps = []


def rel_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    diff = (current.to(previous.dtype) - previous).abs().mean()
    ref = previous.abs().mean().clamp(min=1e-8)
    return (diff / ref).item()


class SelfHealScheduler:
    def __init__(self, total_steps, warmup_steps, heal_thresh,
                 anomaly_window, heal_step_size, conservative_steps,
                 max_heals, verbose, max_consecutive=3, correction_strength=0.5):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.heal_thresh = heal_thresh
        self.anomaly_window = anomaly_window
        self.heal_step_size = heal_step_size
        self.conservative_steps = conservative_steps
        self.max_heals = max_heals
        self.verbose = verbose
        self.max_consecutive = max_consecutive
        self.correction_strength = correction_strength

    def _is_anomaly(self, delta: float, history: list) -> bool:
        """判断当前 delta 是否为异常（远超历史均值+容差）。"""
        if len(history) < self.anomaly_window:
            return False
        window = history[-self.anomaly_window:]
        mean = sum(window) / len(window)
        # 异常判定：当前 delta 远超窗口均值 * heal_thresh 倍率
        return delta > mean * max(1.5, self.heal_thresh) + 0.02

    def should_reuse(self, state: SelfHealState, x: torch.Tensor) -> tuple:
        step = state.step_idx
        if state.prev_output is None or state.prev_input is None:
            return (False, 0.0, 0.0, "no_ref", False)

        delta = rel_l1(x.detach().cpu(), state.prev_input)
        state.delta_history.append(delta)

        # warmup 全算
        if step < self.warmup_steps:
            return (False, delta, 0.0, "warmup(%d<%d)" % (step, self.warmup_steps), False)

        # === 自愈触发检测 ===
        if not state.heal_triggered:
            if self._is_anomaly(delta, state.delta_history):
                state.anomaly_count += 1
                if state.anomaly_count >= 2:
                    # 触发自愈：隔离崩坏段
                    state.heal_triggered = True
                    state.heal_count += 1
                    if self.verbose:
                        print("[SelfHeal] Step %d: 检测到崩坏 (delta=%.4f) → 触发自愈，从 Step %d 重构" % (
                            step, delta, state.last_healthy_step))
                    return (True, delta, state.last_healthy_step, "heal_trigger", True)
            else:
                state.anomaly_count = max(0, state.anomaly_count - 1)
        else:
            # 自愈中：用保守参数（不缓存），直到 conservative_steps 恢复
            if step < state.last_healthy_step + state.conservative_steps:
                state.heal_triggered = False  # 保守期结束，恢复正常判定
                return (False, delta, 0.0, "heal_conservative", False)

        # 正常缓存判定
        if state.heal_count >= self.max_heals:
            state.heal_triggered = False
            return (False, delta, 0.0, "max_heals", False)

        # === [end_step 保护] 达到总步数后强制不缓存 ===
        if step >= self.total_steps:
            return (False, delta, 0.0, "after_end", False)

        # 正常缓存（用相对保守的 thresh）
        # === [护栏] 连续缓存上限（防 latent 漂移崩坏） ===
        if state.consecutive_reuse_count >= self.max_consecutive:
            state.consecutive_reuse_count = 0
            if self.verbose:
                print("[SelfHeal] Step %d: 护栏-连续缓存上限(%d) → 强制重算" % (
                    step, self.max_consecutive))
            return (False, delta, 0.15, "guard_max_consecutive", False)

        thresh = 0.15
        if delta < thresh:
            return (True, delta, thresh, "hit", False)
        return (False, delta, thresh, "miss", False)


def make_selfheal_wrapper(
    state: SelfHealState,
    total_steps: int,
    warmup_steps: int,
    heal_thresh: float,
    anomaly_window: int,
    heal_step_size: int,
    conservative_steps: int,
    max_heals: int,
    verbose: bool,
    max_consecutive: int = 3,
    correction_strength: float = 0.5,
) -> Callable[..., torch.Tensor]:
    scheduler = SelfHealScheduler(
        total_steps, warmup_steps, heal_thresh, anomaly_window,
        heal_step_size, conservative_steps, max_heals, verbose, max_consecutive,
        correction_strength)

    def wrapper(apply_model, args) -> torch.Tensor:
        x = args["input"]
        reuse, delta, thresh, reason, heal = scheduler.should_reuse(state, x)

        # 记录健康 checkpoint（计算步时更新）
        if not reuse:
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
            state.last_healthy_step = state.step_idx
            state.last_healthy_output = output.detach().cpu()
            state.real_count += 1
            state.real_steps.append(state.step_idx)
            state.step_idx += 1
            state.consecutive_reuse_count = 0
            if verbose:
                print("[SelfHeal] Step %d: COMPUTE (delta=%.4f %s)" % (state.step_idx - 1, delta, reason))
            if state.step_idx >= total_steps:
                _report(state, total_steps, warmup_steps)
            return output
        else:
            state.reuse_count += 1
            state.hit_steps.append(state.step_idx)
            state.step_idx += 1
            state.consecutive_reuse_count += 1
            if heal and state.last_healthy_output is not None:
                # 自愈重构：回到最近健康输出（加趋势外推防停滞）
                if verbose:
                    print("[SelfHeal] 重构恢复：回到 Step %d 的健康输出" % state.last_healthy_step)
                base = state.last_healthy_output.to(x.device)
                if state.output_trend is not None:
                    return base + state.output_trend.to(x.device) * scheduler.correction_strength
                return base
            if verbose:
                print("[SelfHeal] Step %d: CACHE (delta=%.4f %s)" % (state.step_idx - 1, delta, reason))
            base = state.prev_output.to(x.device)
            if state.output_trend is not None:
                return base + state.output_trend.to(x.device) * scheduler.correction_strength
            return base

    return wrapper


def _report(state, total_steps, warmup_steps):
    tot = state.reuse_count + state.real_count
    rate = (state.reuse_count / tot * 100) if tot else 0.0
    print("[XuanShuSelfHeal] steps=%d warmup=%d heal_triggered=%d/%d reuse=%d real=%d hit=%.1f%%" % (
        total_steps, warmup_steps, state.heal_count, state.max_heals if hasattr(state, 'max_heals') else 0,
        state.reuse_count, state.real_count, rate))


class MiniMaxH3SelfHeal:
    """玄枢自愈重构节点（自愈电网式）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 8, "min": 1, "max": 200}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 100}),
                "heal_thresh": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1,
                                          "tooltip": "delta 异常判定倍率（相对窗口均值）"}),
                "anomaly_window": ("INT", {"default": 3, "min": 2, "max": 10}),
                "heal_step_size": ("INT", {"default": 2, "min": 1, "max": 10}),
                "conservative_steps": ("INT", {"default": 3, "min": 1, "max": 10,
                                               "tooltip": "自愈后保守重采样步数"}),
                "max_heals": ("INT", {"default": 2, "min": 0, "max": 10}),
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
    TITLE = "MiniMax H3 自愈重构"

    def apply(self, model, total_steps, warmup_steps, heal_thresh,
              anomaly_window, heal_step_size, conservative_steps, max_heals, verbose,
              max_consecutive=3, correction_strength=0.5):
        if "xuanshu_taiji_armor" in model.model_options or \
           "xuanshu_teacache_guard" in model.model_options or \
           "xuanshu_taiji_armor_v4" in model.model_options or \
           "xuanshu_agc" in model.model_options:
            print("[SelfHeal] 检测到其他缓存节点已挂载，跳过（互斥）")
            return (model,)
        state = SelfHealState()
        state.max_heals = max_heals
        patched = model.clone()
        patched.set_model_unet_function_wrapper(
            make_selfheal_wrapper(state, total_steps, warmup_steps, heal_thresh,
                                  anomaly_window, heal_step_size, conservative_steps,
                                  max_heals, verbose, max_consecutive, correction_strength))
        patched.model_options["xuanshu_selfheal"] = {"state": state}
        return (patched,)


NODE_CLASS_MAPPINGS_SELFHEAL = {
    "MiniMaxH3SelfHeal": MiniMaxH3SelfHeal,
}
NODE_DISPLAY_NAME_MAPPINGS_SELFHEAL = {
    "MiniMaxH3SelfHeal": "MiniMax H3 自愈重构",
}