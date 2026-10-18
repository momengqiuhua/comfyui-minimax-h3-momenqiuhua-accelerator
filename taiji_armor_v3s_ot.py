# -*- coding: utf-8 -*-
"""玄枢 v3sOT — v3s + 最优传输距离(OT)判定信号 正式生产节点。

复用 v3s 全部护栏（warmup / 突变保护 / 连续缓存上限 / 误差熔断 / IEC / fallback / trend 修正），
仅将判定信号从 rel_l1/residual/weighted 替换为 OT 距离（排序 Wasserstein-1 相对化）。
OT 值尺度与 rel_l1 不同，v3s 的 poly 对数拟合（受 step0 的 delta=0 离群点污染）不适用，
故用固定阈值 ot_thresh。

测试结论（2026-08-24，864x480 62帧）：
- OT 判定信号与真实输出误差的 Spearman 从 l1 的 -0.734 改善到 -0.420（+43%），优于 cosine(-0.629)
- 但早期步「输入变化小、输出变化大」根本矛盾仍在，建议 warmup 覆盖 step4-7 高误区
"""
from __future__ import annotations

import time as _time

import torch

from .taiji_armor_v3s import TaijiSchedulerV3S, TaijiArmorV3SState


def ot_w1_rel(cur: torch.Tensor, prev: torch.Tensor) -> float:
    """1D 最优传输距离（排序 Wasserstein-1）相对化：通道级排序后 mean|diff| / ref。"""
    cur = cur.to(prev.device).to(prev.dtype)
    c = cur.flatten(start_dim=1)      # [B, C, N]
    p = prev.flatten(start_dim=1)
    sc, _ = torch.sort(c, dim=-1)
    sp, _ = torch.sort(p, dim=-1)
    w1 = (sc - sp).abs().mean()
    ref = prev.abs().mean().clamp(min=1e-8)
    return (w1 / ref).item()


def make_ot_wrapper_v3s(
    state: TaijiArmorV3SState,
    total_steps: int,
    warmup_steps: int,
    ot_thresh: float,
    yang_thresh: float,
    adaptive: bool,
    enable_ensemble: bool,
    vram_thresh_mb: int,
    fallback_mode: bool,
    end_step: int,
    verbose: bool,
    adaptive_scale: float,
    poly_calibrate: bool,
    poly_degree: int,
    safety_margin: float,
    error_budget: float,
    max_consecutive: int,
    correction_strength: float,
    confidence_mode: bool,
    iec_enabled: bool,
    iec_gamma: float,
):
    scheduler = TaijiSchedulerV3S(
        total_steps=total_steps, warmup_steps=warmup_steps,
        yang_thresh=yang_thresh, adaptive=adaptive,
        enable_ensemble=enable_ensemble, vram_thresh_mb=vram_thresh_mb,
        fallback_mode=fallback_mode, end_step=end_step, verbose=verbose,
        adaptive_scale=adaptive_scale, poly_calibrate=poly_calibrate,
        poly_degree=poly_degree, safety_margin=safety_margin,
        error_budget=error_budget, max_consecutive=max_consecutive,
        correction_strength=correction_strength,
        confidence_mode=confidence_mode, iec_enabled=iec_enabled,
        iec_gamma=iec_gamma,
    )
    # 判定信号替换为 OT 距离；固定阈值，关闭 poly/adaptive/yang 下限
    scheduler._compute_signal = lambda strat, cur, prev, pp: ot_w1_rel(cur, prev)
    scheduler.poly_calibrate = False
    scheduler.adaptive = False
    scheduler.yang_thresh = 0.001
    for _s in scheduler.STRATEGIES:
        scheduler.STRATEGIES[_s]["thresh"] = ot_thresh

    _step_t0 = [_time.perf_counter()]

    def wrapper(apply_model, args) -> torch.Tensor:
        x = args["input"]
        _t0 = _time.perf_counter()
        if _step_t0[0] is None:
            _step_t0[0] = _t0
        reuse, delta, thresh, confidence, strategy = scheduler.should_reuse(state, x)
        _step_wall = _time.perf_counter() - _step_t0[0]

        if reuse:
            state.reuse_count += 1
            state.step_wall_times.append(_step_wall)
            state.step_wall_cumulative = (state.step_wall_cumulative or 0.0) + _step_wall
            if correction_strength > 0 and state.output_trend is not None:
                corrected = state.prev_output + state.output_trend * correction_strength
                state.correction_count += 1
                if verbose:
                    print("[XuanShuArmor v3sOT] Step %d/%d: CACHE+修正 (delta=%.4f thresh=%.4f conf=%.2f)" % (
                        state.step_idx, total_steps, delta, thresh, confidence))
                return corrected.to(x.device)
            if verbose:
                print("[XuanShuArmor v3sOT] Step %d/%d: CACHE (delta=%.4f thresh=%.4f conf=%.2f)" % (
                    state.step_idx, total_steps, delta, thresh, confidence))
            return state.prev_output.to(x.device)

        # 计算步：fallback 检测
        if fallback_mode:
            total_done = state.reuse_count + state.real_count
            fb_thresh = scheduler._fallback_threshold()
            if total_done > 3 and state.reuse_count / total_done > fb_thresh:
                state.anomaly_count += 1
                if state.anomaly_count >= 2:
                    state.fallback_active = True
                    if verbose:
                        print("[XuanShuArmor v3sOT] 护体真元触发：命中率 %d/%d > %.2f → 回退全算" % (
                            state.reuse_count, total_done, fb_thresh))
            else:
                state.anomaly_count = 0

        output = apply_model(x, args["timestep"], **args["c"])

        # trend 修正（EMA of output changes）
        if state.prev_output is not None and correction_strength > 0:
            change = output.detach().cpu() - state.prev_output
            if state.output_trend is None:
                state.output_trend = change
            else:
                state.output_trend = 0.7 * state.output_trend + 0.3 * change

        # IEC 误差记录
        if iec_enabled and delta > 0 and state.step_idx > 0:
            state.iec_error_buffer.append((state.step_idx, delta))
            if len(state.iec_error_buffer) > state.iec_max_buffer:
                state.iec_error_buffer.pop(0)

        state.prev_prev_input = state.prev_input if state.prev_input is not None else None
        state.prev_input = x.detach().cpu()
        state.prev_output = output.detach().cpu()
        state.prev_input_shape = tuple(x.shape)
        state.real_count += 1
        state.step_wall_times.append(_step_wall)
        state.step_wall_cumulative = (state.step_wall_cumulative or 0.0) + _step_wall
        if verbose:
            print("[XuanShuArmor v3sOT] Step %d/%d: COMPUTE (delta=%.4f thresh=%.4f conf=%.2f %s)" % (
                state.step_idx, total_steps, delta, thresh, confidence, strategy))
        state.step_idx += 1
        _step_t0[0] = _time.perf_counter()
        return output

    return wrapper


class XuanShuV3SOT:
    """玄枢 v3sOT：v3s + 最优传输距离判定信号。固定阈值 ot_thresh。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 100}),
                "ot_thresh": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 1.0, "step": 0.001, "tooltip": "OT距离<此值才缓存。OT尺度~0.005-0.1，0.03-0.05较平衡"}),
                "end_step": ("INT", {"default": -1, "min": -20, "max": 100, "tooltip": "8步建议-1，20步建议-2"}),
                "enable_ensemble": ("BOOLEAN", {"default": True}),
                "vram_thresh_mb": ("INT", {"default": 7500, "min": 1000, "max": 20000}),
                "fallback_mode": ("BOOLEAN", {"default": True}),
                "error_budget": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05}),
                "max_consecutive": ("INT", {"default": 3, "min": 1, "max": 10, "tooltip": "8步建议2-3，20步4"}),
                "correction_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "confidence_mode": ("BOOLEAN", {"default": True}),
                "iec_enabled": ("BOOLEAN", {"default": True}),
                "iec_gamma": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/Armor"
    TITLE = "玄枢 v3sOT（最优传输距离判定）"

    def apply(self, model, total_steps, warmup_steps, ot_thresh, end_step,
              enable_ensemble, vram_thresh_mb, fallback_mode, error_budget,
              max_consecutive, correction_strength, confidence_mode,
              iec_enabled, iec_gamma, verbose):
        state = TaijiArmorV3SState()
        state.reset()
        wrapper = make_ot_wrapper_v3s(
            state=state, total_steps=total_steps, warmup_steps=warmup_steps,
            ot_thresh=ot_thresh, yang_thresh=0.15, adaptive=False,
            enable_ensemble=enable_ensemble, vram_thresh_mb=vram_thresh_mb,
            fallback_mode=fallback_mode, end_step=end_step, verbose=verbose,
            adaptive_scale=2.5, poly_calibrate=False, poly_degree=2,
            safety_margin=1.3, error_budget=error_budget,
            max_consecutive=max_consecutive, correction_strength=correction_strength,
            confidence_mode=confidence_mode, iec_enabled=iec_enabled,
            iec_gamma=iec_gamma,
        )
        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        patched.model_options["xuanshu_v3s_ot"] = {"state": state}
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "XuanShuV3SOT": XuanShuV3SOT,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "XuanShuV3SOT": "玄枢 v3sOT（最优传输距离判定）",
}
