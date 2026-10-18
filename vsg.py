"""XuanShu H3 时序惯量节点 (VSG式) — 给视频时序加惯量缓冲，防帧间突变。

电网原型：虚拟同步机（VSG）——给新能源系统提供虚拟惯量，防频率崩溃。
逻辑：视频生成的帧间时序突变（画面跳变/闪烁）如同电网频率崩溃，
     给 latent 时序加"惯性轮"，平滑帧间变化，让缓存跳步沿时间轴平滑。

v2 重做（2026-08-21）：根治空转（真正接入缓存判定 + 修复 step_idx 计数）。
v3（2026-08-21）：加护栏（突变保护 + 连续缓存上限），I2V 下实测 VSG 崩坏
  （猫重影/色彩错位），根因是 I2V latent delta 突变时缓存失误；对齐 AGC 的
  surge_factor 突变保护 + max_consecutive 连续缓存护栏后根治。

自包含：仅 numpy/torch + 标准库。
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import torch


@dataclasses.dataclass
class VSGState:
    step_idx: int = 0
    prev_input: Optional[torch.Tensor] = None
    prev_output: Optional[torch.Tensor] = None
    inertia_buffer: Optional[torch.Tensor] = None
    consecutive_reuse_count: int = 0
    reuse_count: int = 0
    real_count: int = 0
    hit_steps: list = dataclasses.field(default_factory=list)
    real_steps: list = dataclasses.field(default_factory=list)

    def reset(self) -> None:
        self.step_idx = 0
        self.prev_input = None
        self.prev_output = None
        self.inertia_buffer = None
        self.consecutive_reuse_count = 0
        self.reuse_count = 0
        self.real_count = 0
        self.hit_steps = []
        self.real_steps = []


def rel_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    diff = (current.to(previous.dtype) - previous).abs().mean()
    ref = previous.abs().mean().clamp(min=1e-8)
    return (diff / ref).item()


def _vsg_should_reuse(state: VSGState, x: torch.Tensor, thresh: float,
                      warmup_steps: int, max_consecutive: int,
                      surge_factor: float, verbose: bool) -> tuple:
    """缓存判定：当前输入相对上一输入变化 < 阈值则复用，带 warmup/突变保护/连续缓存护栏。"""
    step = state.step_idx
    if state.prev_output is None or state.prev_input is None:
        return (False, 0.0, thresh, "no_ref")
    delta = rel_l1(x.detach().cpu(), state.prev_input)
    if step < warmup_steps:
        return (False, delta, thresh, "warmup(%d<%d)" % (step, warmup_steps))
    # 突变保护：delta 远超阈值 → 强制重算（防缓存失误）
    if delta > thresh * surge_factor:
        return (False, delta, thresh, "surge_guard")
    # 连续缓存护栏：达到上限强制重算（防 latent 漂移崩坏）
    if state.consecutive_reuse_count >= max_consecutive:
        return (False, delta, thresh, "guard_max_consecutive")
    if delta < thresh:
        return (True, delta, thresh, "hit")
    return (False, delta, thresh, "miss")


def make_vsg_wrapper(
    state: VSGState,
    total_steps: int,
    warmup_steps: int,
    inertia: float,
    damping: float,
    thresh: float,
    max_consecutive: int,
    surge_factor: float,
    verbose: bool,
) -> Callable[..., torch.Tensor]:
    def wrapper(apply_model, args) -> torch.Tensor:
        x = args["input"]
        reuse, delta, thresh_eff, reason = _vsg_should_reuse(
            state, x, thresh, warmup_steps, max_consecutive, surge_factor, verbose)

        if reuse:
            state.reuse_count += 1
            state.hit_steps.append(state.step_idx)
            state.step_idx += 1  # 修复计数：缓存步也推进
            state.consecutive_reuse_count += 1
            if state.inertia_buffer is not None:
                smooth = state.prev_output.to(x.device) * (1 - damping) \
                         + state.inertia_buffer.to(x.device) * damping
                state.inertia_buffer = smooth.detach().cpu()
                if verbose:
                    print("[VSG] Step %d/%d: CACHE+惯量平滑 (delta=%.4f)" % (
                        state.step_idx - 1, total_steps, delta))
                return smooth
            if verbose:
                print("[VSG] Step %d/%d: CACHE (delta=%.4f)" % (
                    state.step_idx - 1, total_steps, delta))
            return state.prev_output.to(x.device)

        output = apply_model(x, args["timestep"], **args["c"])
        out_cpu = output.detach().cpu()
        if state.inertia_buffer is None:
            state.inertia_buffer = out_cpu
        else:
            state.inertia_buffer = inertia * state.inertia_buffer + (1 - inertia) * out_cpu
        state.prev_input = x.detach().cpu()
        state.prev_output = out_cpu
        state.consecutive_reuse_count = 0
        state.real_count += 1
        state.real_steps.append(state.step_idx)
        state.step_idx += 1
        if verbose:
            print("[VSG] Step %d/%d: COMPUTE (delta=%.4f)" % (
                state.step_idx - 1, total_steps, delta))
        if state.step_idx >= total_steps:
            tot = state.reuse_count + state.real_count
            rate = (state.reuse_count / tot * 100) if tot else 0.0
            print("[XuanShuVSG] steps=%d warmup=%d thresh=%.3f inertia=%.2f damping=%.2f "
                  "reuse=%d real=%d hit=%.1f%%" % (
                      total_steps, warmup_steps, thresh, inertia, damping,
                      state.reuse_count, state.real_count, rate))
        return output

    return wrapper


class MiniMaxH3VSG:
    """玄枢时序惯量节点（VSG 式）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 8, "min": 1, "max": 200}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 100,
                                        "tooltip": "前 N 步全算，建立稳定基准后再缓存"}),
                "inertia": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "惯量系数：越高越平滑但响应越慢"}),
                "damping": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "阻尼系数：抑制帧间振荡"}),
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
    TITLE = "MiniMax H3 时序惯量 (VSG)"

    def apply(self, model, total_steps, warmup_steps, inertia, damping, thresh,
              max_consecutive, surge_factor, verbose):
        if "xuanshu_taiji_armor" in model.model_options or \
           "xuanshu_teacache_guard" in model.model_options or \
           "xuanshu_agc" in model.model_options or \
           "xuanshu_selfheal" in model.model_options:
            print("[VSG] 检测到其他缓存节点已挂载，跳过（互斥）")
            return (model,)
        state = VSGState()
        patched = model.clone()
        patched.set_model_unet_function_wrapper(
            make_vsg_wrapper(state, total_steps, warmup_steps, inertia, damping, thresh,
                             max_consecutive, surge_factor, verbose))
        patched.model_options["xuanshu_vsg"] = {"state": state}
        return (patched,)


NODE_CLASS_MAPPINGS_VSG = {
    "MiniMaxH3VSG": MiniMaxH3VSG,
}
NODE_DISPLAY_NAME_MAPPINGS_VSG = {
    "MiniMaxH3VSG": "MiniMax H3 时序惯量 (VSG)",
}
