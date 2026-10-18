"""玄枢缓存策略选择器 — 一个节点内切换 AGC/VSG/PSS 三种缓存策略。

用途：替代三个独立节点（MiniMaxH3AGCAuto / MiniMaxH3VSG / MiniMaxH3PSS），
     工作流里只放一个节点，通过 strategy 下拉切换缓存策略即可对比。
所有策略共用同一套护栏（warmup + 突变保护 + 连续缓存上限）。
"""
from __future__ import annotations

import torch

from .agc_auto import AGCState, make_agc_wrapper
from .vsg import VSGState, make_vsg_wrapper
from .pss import PSSState, make_pss_wrapper

STRATEGIES = ["AGC", "VSG", "PSS"]


class XuanShuCacheSelector:
    """统一缓存策略选择器。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strategy": (STRATEGIES, {"default": "AGC", "tooltip": "选择缓存策略：AGC闭环PID / VSG惯量平滑 / PSS振荡阻尼"}),
                # 通用
                "total_steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "warmup_steps": ("INT", {"default": 4, "min": 0, "max": 100, "tooltip": "前 N 步全算，建立稳定基准"}),
                "thresh": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.5, "step": 0.01, "tooltip": "缓存判定阈值"}),
                "max_consecutive": ("INT", {"default": 3, "min": 1, "max": 20, "tooltip": "连续缓存上限（护栏）"}),
                "surge_factor": ("FLOAT", {"default": 4.0, "min": 2.0, "max": 10.0, "step": 0.5, "tooltip": "突变保护倍数"}),
                "verbose": ("BOOLEAN", {"default": True}),
                # AGC 特有
                "setpoint": ("FLOAT", {"default": 0.08, "min": 0.01, "max": 0.5, "step": 0.01, "tooltip": "AGC目标误差"}),
                "kp": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "PID比例增益"}),
                "ki": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "PID积分增益"}),
                "correction_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "趋势外推强度（0=纯复用最优）"}),
                "meas_window": ("INT", {"default": 4, "min": 1, "max": 20, "tooltip": "PID整定窗口"}),
                "adaptive": ("BOOLEAN", {"default": True}),
                # VSG 特有
                "inertia": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "惯量系数（越高越平滑）"}),
                "damping": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "阻尼系数"}),
                # PSS 特有
                "osc_window": ("INT", {"default": 6, "min": 3, "max": 20, "tooltip": "振荡检测窗口"}),
                "damping_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "阻尼强度"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "玄枢缓存策略选择器 (AGC/VSG/PSS)"

    def apply(self, model, strategy, total_steps, warmup_steps, thresh,
              max_consecutive, surge_factor, verbose,
              setpoint, kp, ki, correction_strength, meas_window, adaptive,
              inertia, damping, osc_window, damping_strength):
        if "xuanshu_taiji_armor" in model.model_options or \
           "xuanshu_teacache_guard" in model.model_options or \
           "xuanshu_agc" in model.model_options or \
           "xuanshu_selfheal" in model.model_options or \
           "xuanshu_vsg" in model.model_options or \
           "xuanshu_pss" in model.model_options:
            print("[CacheSelector] 检测到其他缓存节点已挂载，跳过（互斥）")
            return (model,)

        patched = model.clone()
        if strategy == "AGC":
            state = AGCState()
            patched.set_model_unet_function_wrapper(make_agc_wrapper(
                state, total_steps, warmup_steps, thresh, setpoint, kp, ki, 0.0,
                0.05, 0.4, meas_window, adaptive, verbose, max_consecutive,
                correction_strength, surge_factor))
            patched.model_options["xuanshu_agc"] = {"state": state}
            if verbose:
                print("[CacheSelector] 策略=AGC thresh=%.3f warmup=%d" % (thresh, warmup_steps))
        elif strategy == "VSG":
            state = VSGState()
            patched.set_model_unet_function_wrapper(make_vsg_wrapper(
                state, total_steps, warmup_steps, inertia, damping, thresh,
                max_consecutive, surge_factor, verbose))
            patched.model_options["xuanshu_vsg"] = {"state": state}
            if verbose:
                print("[CacheSelector] 策略=VSG thresh=%.3f inertia=%.2f damping=%.2f" % (thresh, inertia, damping))
        else:  # PSS
            state = PSSState()
            patched.set_model_unet_function_wrapper(make_pss_wrapper(
                state, total_steps, warmup_steps, osc_window, damping_strength,
                thresh, max_consecutive, surge_factor, verbose))
            patched.model_options["xuanshu_pss"] = {"state": state}
            if verbose:
                print("[CacheSelector] 策略=PSS thresh=%.3f osc_window=%d" % (thresh, osc_window))
        return (patched,)


NODE_CLASS_MAPPINGS_SELECTOR = {
    "XuanShuCacheSelector": XuanShuCacheSelector,
}
NODE_DISPLAY_NAME_MAPPINGS_SELECTOR = {
    "XuanShuCacheSelector": "玄枢缓存策略选择器",
}
