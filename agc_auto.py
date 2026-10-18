"""XuanShu H3 闭环自整定节点 (AGC 式) — 实时闭环调缓存阈值。

电网原型：自动发电控制 AGC（自动发电控制/二次调频）。
逻辑：采样中实时测量"频率偏差"（帧间一致性质量指标），
     通过闭环控制器（增量式 PID）自动微调缓存阈值，而非开环预设。
     解决太极装甲 v4 不稳的核心——从"猜阈值"变"测出来调"。

自包含：仅 numpy/torch + 标准库。
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import torch

@dataclasses.dataclass
class AGCState:
    step_idx: int = 0
    prev_input: Optional[torch.Tensor] = None
    prev_output: Optional[torch.Tensor] = None
    reuse_count: int = 0
    real_count: int = 0
    consecutive_reuse_count: int = 0
    output_trend: Optional[torch.Tensor] = None
    # PID 闭环状态
    integral: float = 0.0
    prev_error: float = 0.0
    thresh: float = 0.15
    # 质量观测
    quality_history: list = dataclasses.field(default_factory=list)
    hit_steps: list = dataclasses.field(default_factory=list)
    real_steps: list = dataclasses.field(default_factory=list)
    # 振荡检测
    oscillation_count: int = 0

    def reset(self) -> None:
        self.step_idx = 0
        self.prev_input = None
        self.prev_output = None
        self.reuse_count = 0
        self.real_count = 0
        self.consecutive_reuse_count = 0
        self.output_trend = None
        self.integral = 0.0
        self.prev_error = 0.0
        self.thresh = 0.15
        self.quality_history = []
        self.hit_steps = []
        self.real_steps = []
        self.oscillation_count = 0


def rel_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    diff = (current.to(previous.dtype) - previous).abs().mean()
    ref = previous.abs().mean().clamp(min=1e-8)
    return (diff / ref).item()


def detect_oscillation(errors: list, window: int = 5) -> bool:
    """检测误差序列是否出现正负交替（低频振荡先兆）。"""
    if len(errors) < window:
        return False
    tail = errors[-window:]
    signs = [1 if e > 0 else (-1 if e < 0 else 0) for e in tail]
    sign_changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1] and signs[i] != 0)
    return sign_changes >= window - 2


class AGCScheduler:
    def __init__(self, total_steps, warmup_steps, base_thresh,
                 setpoint, kp, ki, kd, clamp_min, clamp_max,
                 meas_window, adaptive, verbose, max_consecutive=3,
                 correction_strength=0.5, surge_factor=4.0):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_thresh = base_thresh
        self.setpoint = setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.meas_window = meas_window
        self.adaptive = adaptive
        self.verbose = verbose
        self.max_consecutive = max_consecutive
        self.correction_strength = correction_strength
        self.surge_factor = surge_factor

    def _quality_error(self, state: AGCState, delta: float) -> float:
        """质量误差：缓存造成的偏差 delta 相对设定点的偏差。
        delta 过大→缓存太激进（画面变差）→需要收紧阈值；
        delta 过小→缓存太保守（浪费算力）→需要放松阈值。
        """
        return delta - self.setpoint

    def _pid_update(self, state: AGCState, error: float) -> float:
        """增量式 PID：error>0(缓存太激进/delta过大)→收紧阈值(降)；error<0(保守)→放松(升)。
        修复：原代码 thresh+adjustment 方向反了，导致激进时反而升阈值→更激进→崩坏。"""
        d_err = error - state.prev_error
        adjustment = self.kp * d_err + self.ki * error
        state.integral += error
        # 抗积分饱和
        state.integral = max(-1.0, min(1.0, state.integral))
        state.prev_error = error
        # 反号：error 大(激进)→降阈值收紧；error 小(保守)→升阈值放松
        new_thresh = state.thresh - adjustment
        # 限幅
        new_thresh = max(self.clamp_min, min(self.clamp_max, new_thresh))
        return new_thresh

    def _measure_quality(self, state: AGCState, x: torch.Tensor) -> float:
        """测量当前帧质量：与前一输入差异的平滑版，作为"频率偏差"。"""
        if state.prev_input is None:
            return self.setpoint
        return rel_l1(x.detach().cpu(), state.prev_input)

    def should_reuse(self, state: AGCState, x: torch.Tensor) -> tuple:
        step = state.step_idx
        if state.prev_output is None or state.prev_input is None:
            return (False, 0.0, state.thresh, 0.0, "no_ref")

        delta = self._measure_quality(state, x)
        state.quality_history.append(delta)

        # warmup 阶段全算，不闭环调节
        if step < self.warmup_steps:
            state.real_steps.append(step)
            return (False, delta, state.thresh, 0.0, "warmup(%d<%d)" % (step, self.warmup_steps))

        effective_end = self.total_steps  # AGC 全程可缓存（由 warmup 保护起算）
        if step >= effective_end:
            return (False, delta, state.thresh, 0.0, "after_end")

        # === 闭环控制：每 meas_window 步更新一次阈值 ===
        if self.adaptive and (step % max(1, self.meas_window) == 0):
            # 用最近窗口的平均 delta 作为误差源
            window_avg = sum(state.quality_history[-self.meas_window:]) / max(1, len(state.quality_history[-self.meas_window:]))
            error = self._quality_error(state, window_avg)
            state.thresh = self._pid_update(state, error)
            if self.verbose:
                print("[AGC] Step %d: 闭环整定 err=%.4f → thresh=%.4f" % (step, error, state.thresh))

        # 振荡抑制：检测到正负交替时收紧
        if detect_oscillation([self._quality_error(state, d) for d in state.quality_history[-6:]]):
            state.thresh = max(self.clamp_min, state.thresh * 0.7)
            state.oscillation_count += 1
            if self.verbose:
                print("[AGC] Step %d: 检测到振荡趋势 → 收紧 thresh=%.4f" % (step, state.thresh))

        # 缓存决策
        # === [突变保护] delta 远超当前阈值 → 强制重算（防 PID 滞后） ===
        if delta > state.thresh * self.surge_factor:
            state.real_count += 1
            state.real_steps.append(step)
            state.consecutive_reuse_count = 0
            if self.verbose:
                print("[AGC] Step %d: 突变保护 delta=%.4f>thresh*%.1f=%.4f → 强制重算" % (
                    step, delta, self.surge_factor, state.thresh * self.surge_factor))
            return (False, delta, state.thresh, 0.0, "surge_guard")

        # === [护栏] 连续缓存上限（防 latent 漂移崩坏） ===
        if state.consecutive_reuse_count >= self.max_consecutive:
            state.real_count += 1
            state.real_steps.append(step)
            state.consecutive_reuse_count = 0
            if self.verbose:
                print("[AGC] Step %d/%d: 护栏-连续缓存上限(%d) → 强制重算" % (
                    step, self.total_steps, self.max_consecutive))
            return (False, delta, state.thresh, 0.0, "guard_max_consecutive")

        if delta < state.thresh:
            state.reuse_count += 1
            state.hit_steps.append(step)
            if self.verbose:
                print("[AGC] Step %d/%d: CACHE (delta=%.4f thresh=%.4f)" % (
                    step, self.total_steps, delta, state.thresh))
            return (True, delta, state.thresh, 1.0, "hit")

        state.real_count += 1
        state.real_steps.append(step)
        if self.verbose:
            print("[AGC] Step %d/%d: COMPUTE (delta=%.4f thresh=%.4f)" % (
                step, self.total_steps, delta, state.thresh))
        return (False, delta, state.thresh, 0.0, "miss")


def make_agc_wrapper(
    state: AGCState,
    total_steps: int,
    warmup_steps: int,
    base_thresh: float,
    setpoint: float,
    kp: float,
    ki: float,
    kd: float,
    clamp_min: float,
    clamp_max: float,
    meas_window: int,
    adaptive: bool,
    verbose: bool,
    max_consecutive: int = 3,
    correction_strength: float = 0.5,
    surge_factor: float = 4.0,
) -> Callable[..., torch.Tensor]:
    scheduler = AGCScheduler(
        total_steps, warmup_steps, base_thresh, setpoint, kp, ki, kd,
        clamp_min, clamp_max, meas_window, adaptive, verbose, max_consecutive,
        correction_strength, surge_factor)

    def wrapper(apply_model, args) -> torch.Tensor:
        x = args["input"]
        reuse, delta, thresh, confidence, reason = scheduler.should_reuse(state, x)

        if reuse:
            state.consecutive_reuse_count += 1
            state.step_idx += 1
            # === 趋势外推补偿（根治崩坏：纯复用导致 latent 停滞糊化） ===
            base = state.prev_output.to(x.device)
            if state.output_trend is not None:
                out = base + state.output_trend.to(x.device) * scheduler.correction_strength
            else:
                out = base
            return out

        output = apply_model(x, args["timestep"], **args["c"])
        # === 更新趋势 EMA（记录输出演化方向，供缓存步外推） ===
        if state.prev_output is not None:
            change = output.detach().cpu() - state.prev_output
            if state.output_trend is None:
                state.output_trend = change.clone()
            else:
                state.output_trend = 0.7 * state.output_trend + 0.3 * change
        state.prev_input = x.detach().cpu()
        state.prev_output = output.detach().cpu()
        state.step_idx += 1
        state.consecutive_reuse_count = 0
        if state.step_idx >= total_steps:
            tot = state.reuse_count + state.real_count
            rate = (state.reuse_count / tot * 100) if tot else 0.0
            print("[XuanShuAGC] steps=%d warmup=%d base_thresh=%.3f setpoint=%.4f "
                  "thresh_final=%.3f reuse=%d real=%d hit=%.1f%% osc=%d" % (
                      total_steps, warmup_steps, base_thresh, setpoint, state.thresh,
                      state.reuse_count, state.real_count, rate, state.oscillation_count))
        return output

    return wrapper


class MiniMaxH3AGCAuto:
    """玄枢闭环自整定节点（AGC 式）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 8, "min": 1, "max": 200}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 100,
                                         "tooltip": "前 N 步全算，建立稳定基准后再闭环"}),
                "base_thresh": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.5, "step": 0.01}),
                "setpoint": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.3, "step": 0.01,
                                       "tooltip": "期望的缓存偏差设定点（越小越保守）"}),
                "kp": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
                "ki": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "kd": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "clamp_min": ("FLOAT", {"default": 0.03, "min": 0.01, "max": 0.2, "step": 0.01}),
                "clamp_max": ("FLOAT", {"default": 0.35, "min": 0.1, "max": 0.5, "step": 0.01}),
                "meas_window": ("INT", {"default": 2, "min": 1, "max": 10}),
                "max_consecutive": ("INT", {"default": 3, "min": 1, "max": 20,
                                        "tooltip": "连续缓存上限（护栏）：达到后强制重算，防漂移崩坏"}),
                "correction_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "趋势外推强度：缓存步用 prev_output+trend 补偿，0=纯复用"}),
                "surge_factor": ("FLOAT", {"default": 4.0, "min": 2.0, "max": 10.0, "step": 0.5,
                                        "tooltip": "突变保护：delta 超当前阈值此倍数则强制重算，防 PID 滞后误缓存"}),
                "adaptive": ("BOOLEAN", {"default": True, "tooltip": "启用闭环 PID 自整定"}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "MiniMax H3 闭环自整定 (AGC)"

    def apply(self, model, total_steps, warmup_steps, base_thresh, setpoint,
              kp, ki, kd, clamp_min, clamp_max, meas_window, adaptive, verbose,
              max_consecutive=3, correction_strength=0.5, surge_factor=4.0):
        # 与装甲/Guard 互斥：检测到已挂载则返回原模型（避免双重包裹冲突）
        if "xuanshu_taiji_armor" in model.model_options or \
           "xuanshu_teacache_guard" in model.model_options or \
           "xuanshu_taiji_armor_v4" in model.model_options or \
           "xuanshu_agc" in model.model_options:
            print("[AGC] 检测到其他缓存节点已挂载，跳过（互斥）")
            return (model,)
        state = AGCState()
        state.thresh = base_thresh
        patched = model.clone()
        patched.set_model_unet_function_wrapper(
            make_agc_wrapper(state, total_steps, warmup_steps, base_thresh,
                             setpoint, kp, ki, kd, clamp_min, clamp_max,
                             meas_window, adaptive, verbose, max_consecutive,
                             correction_strength, surge_factor))
        patched.model_options["xuanshu_agc"] = {"state": state}
        return (patched,)


NODE_CLASS_MAPPINGS_AGC = {
    "MiniMaxH3AGCAuto": MiniMaxH3AGCAuto,
}
NODE_DISPLAY_NAME_MAPPINGS_AGC = {
    "MiniMaxH3AGCAuto": "MiniMax H3 闭环自整定 (AGC)",
}