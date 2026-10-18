"""XuanShu H3 Detail Guard — 中远距细节守护节点。

针对 MiniMax H3 在 RTX 5060 8GB 低分辨率(480x864/576x1024)下，
画面中远距离主体（人物/物体尺度小、细节像素占比低）时易崩坏
（五官糊化、纹理丢失、结构错乱）的专项守护。

核心机制（对应中远距崩坏治理方法）：
1. 主体尺度感知的保守调度：
   主体越远(scale 越小) → 缓存越谨慎(刚步越多) → 稀少细节不被复用糊化
2. 二次采样放大决策：
   当主体尺度低于阈值且分辨率不足时，输出"是否建议二次采样高清放大"
   (低 denoise 少量重绘保结构 + LatentUpscale 提分辨率) 的建议字符串，
   供上层工作流决定是否追加放大段。
3. 运动烈度保护：
   中远距 + 快速运动(残差大) → 强制刚步，避免低步数在远处主体的糊化。

不直接 patch 采样流程，而是以模型 wrapper 形式介入调度，
与 太极装甲(taiji) / TeaCacheGuard 定位错开：三者可串接。
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from .taiji_armor import TaijiArmorState, rel_l1, timestep_residual, attn_var_signal


# ─────────────────────────────────────────────────────────────────────
# 中远距调度器
# ─────────────────────────────────────────────────────────────────────
class DetailGuardScheduler:
    """主体尺度感知的保守调度器。

    中远距(主体 scale 小)时：
    - 缓存阈值被压缩(thresh *= scale 因子) → 更少复用 → 更多刚步守细节
    - 快速运动(残差大) → 强制刚步(与转身保守同源，但针对远处主体更敏感)
    """

    def __init__(
        self,
        total_steps: int,
        warmup_steps: int,
        subject_scale: float,      # 主体尺度 0.05~1.0，越小=越远
        detail_thresh: float,      # 基础缓存阈值(对应近景/画质优先)
        motion_thresh: float,      # 运动烈度阈值：残差超过则强制刚步(中远距敏感)
        adaptive: bool,
        verbose: bool,
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.subject_scale = max(0.05, min(1.0, subject_scale))
        self.detail_thresh = detail_thresh
        self.motion_thresh = motion_thresh
        self.adaptive = adaptive
        self.verbose = verbose

    def should_reuse(self, state: TaijiArmorState, current_input: torch.Tensor) -> bool:
        step = state.step_idx
        if state.prev_output is None or state.prev_input is None:
            state.yang_steps.append(step)
            return False

        # warmup 前 N 步全算（结构未稳）
        if step < self.warmup_steps:
            state.yang_steps.append(step)
            return False

        # 末段保护：最后 1 步全算（细节重要）
        if step >= self.total_steps - 1:
            state.yang_steps.append(step)
            return False

        # 运动烈度(残差)：中远距时 motion_thresh 更敏感 → 快速运动强制刚步
        delta = rel_l1(current_input, state.prev_input)
        res = timestep_residual(current_input, state.prev_input)
        state.accumulate_strategy("detail", delta)

        # 中远距调制：主体越远，缓存阈值越低(更保守)。scale=1(近景)不压缩。
        # scale=0.3 → 阈值×0.5；scale=0.1 → 阈值×0.2 → 几乎不缓存
        scale_factor = 0.15 + 0.85 * self.subject_scale
        eff_thresh = self.detail_thresh * scale_factor

        # 自适应收敛：后期更保守(降阈值守细节)
        if self.adaptive:
            progress = step / max(1, self.total_steps)
            if progress >= 0.7:
                eff_thresh *= 0.7   # 后期显著保守

        # 快速运动 → 刚步(中远距崩坏主因之一)
        if delta > self.motion_thresh:
            state.yang_steps.append(step)
            return False

        # 低变化 → 柔步缓存
        if delta < eff_thresh and res < eff_thresh:
            state.yin_steps.append(step)
            return True

        state.yang_steps.append(step)
        return False


def make_detail_wrapper(
    state: TaijiArmorState,
    total_steps: int,
    warmup_steps: int,
    subject_scale: float,
    detail_thresh: float,
    motion_thresh: float,
    adaptive: bool,
    verbose: bool,
) -> Callable[..., torch.Tensor]:
    scheduler = DetailGuardScheduler(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        subject_scale=subject_scale,
        detail_thresh=detail_thresh,
        motion_thresh=motion_thresh,
        adaptive=adaptive,
        verbose=verbose,
    )

    def wrapper(apply_model, args) -> torch.Tensor:
        # 迭代v3(修复state跨生成累计): ComfyUI缓存导致apply只执行一次，
        # 若上次采样已完成(step_idx>=total)则本次为新采样，自动重置state
        if state.step_idx >= total_steps:
            state.reset()
        x = args["input"]
        if scheduler.should_reuse(state, x):
            state.reuse_count += 1
            state.step_idx += 1
            # 迭代v2(一阶预测缓存, 文献TaylorSeers/AB-Cache): 前两步差分外推替代零阶复用
            if state.prev_prev_output is not None:
                return state.prev_output + (state.prev_output - state.prev_prev_output)
            return state.prev_output

        output = apply_model(x, args["timestep"], **args["c"])
        state.prev_input = x.detach()
        state.prev_prev_output = state.prev_output
        state.prev_output = output.detach()
        state.real_count += 1
        state.step_idx += 1
        if state.step_idx >= total_steps:
            _report(state, total_steps, warmup_steps, subject_scale,
                    detail_thresh, motion_thresh, adaptive, verbose)
        return output

    return wrapper


def _report(state, total_steps, warmup_steps, subject_scale, detail_thresh,
            motion_thresh, adaptive, verbose):
    tot = state.reuse_count + state.real_count
    rate = (state.reuse_count / tot * 100) if tot else 0.0
    print(f"[XuanShuDetailGuard] steps={total_steps} warmup={warmup_steps} "
          f"subject_scale={subject_scale} detail_thresh={detail_thresh} "
          f"motion_thresh={motion_thresh} adaptive={adaptive} "
          f"yang={len(state.yang_steps)} yin={len(state.yin_steps)} "
          f"reuse={state.reuse_count} real={state.real_count} hit_rate={rate:.1f}%")


# ─────────────────────────────────────────────────────────────────────
# 二次采样放大决策
# ─────────────────────────────────────────────────────────────────────
def upscale_advice(subject_scale: float, width: int, height: int) -> str:
    """中远距 + 低分辨率 → 建议二次采样放大(低denoise保结构)。"""
    short_edge = min(width, height)
    # 中远距：主体尺度越小越需要放大补细节
    if subject_scale >= 0.5:
        return ("[DetailGuard] 主体近景或中景，分辨率尚可，暂不需二次采样放大。"
                "若仍觉细节不足可手动追加 LatentUpscale 1.5x + denoise 0.3。")
    if short_edge < 576:
        up = 1.5 if subject_scale >= 0.25 else 2.0
        denoise = 0.3 if subject_scale >= 0.25 else 0.35
        return (f"[DetailGuard] 中远距+低分辨率(短边{short_edge})：建议二次采样高清放大 "
                f"LatentUpscale {up}x(目标≥{int(short_edge*up)}) + KSampler denoise {denoise} "
                f"少量重绘保结构；降低运动强度并加主体参照/face-robust权重更稳。")
    # 短边足够但主体远 → 主要靠保守调度，放大可选
    return ("[DetailGuard] 分辨率尚可(短边≥576)，但主体中远距：主要靠保守调度守细节，"
            "可选 LatentUpscale 1.5x + denoise 0.25 二次放大，配合降低运动强度更佳。")
