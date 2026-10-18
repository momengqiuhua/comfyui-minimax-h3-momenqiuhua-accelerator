"""XuanShu H3 TeaCache Guard v2.1 — 自研增强时序缓存（迭代版）。

v2 修复：
1. 阈值模式可选：单步模式(原始TeaCache语义) / 累积模式(v1行为)
2. prev_input CPU offload 省 VRAM
3. rel_l1 减少临时张量分配
4. 参数校验

v2.1 修复（显存安全+可观测性，与 taiji_armor v2.1 同步）：
5. prev_output 存 CPU（.detach().cpu()），返回时 .to(device)——修复8GB显存OOM崩溃
6. rel_l1 去掉 .float() 临时张量，用 current.to(previous.dtype)
7. delta 计算全部在 CPU 上进行（cur_cpu = current_input.detach().cpu()）
8. should_reuse 返回 (bool, delta, thresh, reason) 元组，wrapper 增加逐步verbose日志
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import torch


@dataclasses.dataclass
class TeaCacheGuardState:
    step_idx: int = 0
    prev_input: Optional[torch.Tensor] = None
    prev_output: Optional[torch.Tensor] = None
    accumulated: float = 0.0
    reuse_count: int = 0
    real_count: int = 0
    hit_steps: list = dataclasses.field(default_factory=list)
    real_steps: list = dataclasses.field(default_factory=list)

    def reset(self) -> None:
        self.step_idx = 0
        self.prev_input = None
        self.prev_output = None
        self.accumulated = 0.0
        self.reuse_count = 0
        self.real_count = 0
        self.hit_steps = []
        self.real_steps = []


def rel_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    diff = (current.to(previous.dtype) - previous).abs().mean()
    ref = previous.abs().mean().clamp(min=1e-8)
    return (diff / ref).item()


def _adaptive_thresh(thresh: float, total_steps: int) -> float:
    if total_steps <= 6:
        return thresh * 1.8
    if total_steps <= 8:
        return thresh * 3.0
    if total_steps <= 12:
        return thresh * 1.5
    return thresh


def should_reuse(
    state: TeaCacheGuardState,
    current_input: torch.Tensor,
    thresh: float,
    start_step: int,
    end_step: int,
    total_steps: int,
    warmup_steps: int,
    adaptive: bool,
    accumulate_mode: bool,
) -> tuple:
    step = state.step_idx
    if state.prev_output is None or state.prev_input is None:
        return (False, 0.0, 0.0, "no_ref")
    if step < start_step:
        return (False, 0.0, 0.0, "before_start(%d<%d)" % (step, start_step))
    if step < warmup_steps:
        return (False, 0.0, 0.0, "warmup(%d<%d)" % (step, warmup_steps))
    effective_end = end_step if end_step >= 0 else total_steps + end_step
    if step >= effective_end:
        return (False, 0.0, 0.0, "after_end(%d>=%d)" % (step, effective_end))

    eff_thresh = _adaptive_thresh(thresh, total_steps) if adaptive else thresh

    cur_cpu = current_input.detach().cpu()
    delta = rel_l1(cur_cpu, state.prev_input)

    if accumulate_mode:
        state.accumulated += delta
        if state.accumulated < eff_thresh:
            return (True, delta, eff_thresh, "accum(%.4f<%.4f)" % (state.accumulated, eff_thresh))
        state.accumulated = 0.0
        return (False, delta, eff_thresh, "accum_reset(%.4f>=%.4f)" % (delta, eff_thresh))
    else:
        if delta < eff_thresh:
            return (True, delta, eff_thresh, "hit")
        return (False, delta, eff_thresh, "miss(%.4f>=%.4f)" % (delta, eff_thresh))


def make_wrapper(
    state: TeaCacheGuardState,
    thresh: float,
    start_step: int,
    end_step: int,
    total_steps: int,
    warmup_steps: int,
    adaptive: bool,
    verbose: bool,
    accumulate_mode: bool,
) -> Callable[..., torch.Tensor]:
    def wrapper(apply_model, args) -> torch.Tensor:
        x = args["input"]
        reuse, delta, eff_thresh, reason = should_reuse(
            state, x, thresh, start_step, end_step, total_steps,
            warmup_steps, adaptive, accumulate_mode)

        if reuse:
            state.reuse_count += 1
            state.step_idx += 1
            state.hit_steps.append(state.step_idx - 1)
            if verbose:
                print("[Guard] Step %d/%d: CACHE (delta=%.4f thresh=%.4f %s)" % (
                    state.step_idx - 1, total_steps, delta, eff_thresh, reason))
            return state.prev_output.to(x.device)

        output = apply_model(x, args["timestep"], **args["c"])
        state.prev_input = x.detach().cpu()
        state.prev_output = output.detach().cpu()
        state.real_count += 1
        state.real_steps.append(state.step_idx)
        if verbose:
            print("[Guard] Step %d/%d: COMPUTE (%s delta=%.4f thresh=%.4f)" % (
                state.step_idx, total_steps, reason, delta, eff_thresh))
        state.step_idx += 1
        if state.step_idx >= total_steps:
            _report(state, total_steps, start_step, end_step, thresh,
                    warmup_steps, adaptive, accumulate_mode, verbose)
        return output

    return wrapper


def _report(state, total_steps, start_step, end_step, thresh,
            warmup_steps, adaptive, accumulate_mode, verbose):
    tot = state.reuse_count + state.real_count
    rate = (state.reuse_count / tot * 100) if tot else 0.0
    mode_str = "累积" if accumulate_mode else "单步"
    base = (f"[XuanShuTeaCacheGuard v2] steps={total_steps} start={start_step} end={end_step} "
            f"thresh={thresh} warmup={warmup_steps} adaptive={adaptive} mode={mode_str} "
            f"reuse={state.reuse_count} real={state.real_count} hit_rate={rate:.1f}%")
    if verbose:
        base += f"\n  hit_steps={state.hit_steps}\n  real_steps={state.real_steps}"
    print(base)
