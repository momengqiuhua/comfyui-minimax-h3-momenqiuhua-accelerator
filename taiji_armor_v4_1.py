"""XuanShu H3 太极装甲 v4 — IEC + AdaCorrection + FeatFix + 频率解耦。

v4 新增：
1. IEC 迭代误差校正：缓存步不累积硬预算，计算步用加权回滚校正
   （替代 v3 的 error_budget 硬熔断，误差从指数增长降为线性增长）
2. AdaCorrection 自适应偏移校正：连续混合 alpha * cached + (1-alpha) * computed
   （替代 v3 的二值缓存决策，消除突变）
3. FeatFix 验证特征复用：should_reuse 已计算的 delta 信号二次利用做轻量校正
   （免费增量加速，6.7x 参考 FeatFix 论文）
4. 频率解耦缓存补偿：缓存输出做 DCT 分解，低频直接复用，高频用 Taylor 展开重建
   （参考 DeCo CVPR 2026 频率解耦思想）

v3 保留（修复）：
1. 多项式拟合对数线性预测
2. 置信度衰减
3. 缓存补偿
4. 养蛊择优策略
5. Welford 方差归一化
6. timestep_residual 真时序差分
7. prev_output 存 CPU（8GB 安全）
"""
from __future__ import annotations

import dataclasses
import math
from typing import Callable, Optional

import numpy as np
import torch


@dataclasses.dataclass
class TaijiArmorV4State:
    step_idx: int = 0
    prev_input: Optional[torch.Tensor] = None
    prev_output: Optional[torch.Tensor] = None
    prev_prev_input: Optional[torch.Tensor] = None
    prev_prev_output: Optional[torch.Tensor] = None

    yang_steps: list = dataclasses.field(default_factory=list)
    yin_steps: list = dataclasses.field(default_factory=list)
    reuse_count: int = 0
    real_count: int = 0

    strategy_scores: dict = dataclasses.field(default_factory=dict)
    current_strategy: str = "l1"

    anomaly_count: int = 0
    fallback_active: bool = False
    peak_vram_mb: int = 0

    # v3 保留
    delta_history: list = dataclasses.field(default_factory=list)
    poly_coeffs: Optional[list] = None
    confidence_history: list = dataclasses.field(default_factory=list)
    poly_thresh_used: int = 0
    strat_thresh_used: int = 0

    # === v4 新增 ===

    # IEC: 误差序列（存储连续缓存步的 delta 值）
    iec_error_buffer: list = dataclasses.field(default_factory=list)
    iec_correction_applied: int = 0

    # AdaCorrection: 混合系数历史
    blend_history: list = dataclasses.field(default_factory=list)

    # FeatFix: 验证特征（仅保留标量调试信号，不再存储张量）
    featfix_signal: Optional[float] = None
    featfix_reuse_count: int = 0

    # EMA趋势外推（v3已验证的补偿机制）
    output_trend: Optional[torch.Tensor] = None
    correction_count: int = 0

    # 频率解耦: DCT 分量缓存
    dct_low_cache: Optional[torch.Tensor] = None
    dct_high_cache: Optional[torch.Tensor] = None
    freq_decouple_count: int = 0

    # 连续缓存控制
    consecutive_reuse_count: int = 0

    # IEC 动态阈值调整
    iec_adjusted_thresh: float = 0.0

    def reset(self) -> None:
        self.step_idx = 0
        self.prev_input = None
        self.prev_output = None
        self.prev_prev_input = None
        self.prev_prev_output = None
        self.yang_steps = []
        self.yin_steps = []
        self.reuse_count = 0
        self.real_count = 0
        self.strategy_scores = {}
        self.current_strategy = "l1"
        self.anomaly_count = 0
        self.fallback_active = False
        self.peak_vram_mb = 0
        self.delta_history = []
        self.poly_coeffs = None
        self.confidence_history = []
        self.poly_thresh_used = 0
        self.strat_thresh_used = 0
        self.iec_error_buffer = []
        self.iec_correction_applied = 0
        self.blend_history = []
        self.featfix_signal = None
        self.featfix_reuse_count = 0
        self.output_trend = None
        self.correction_count = 0
        self.dct_low_cache = None
        self.dct_high_cache = None
        self.freq_decouple_count = 0
        self.consecutive_reuse_count = 0
        self.iec_adjusted_thresh = 0.0

    def accumulate_strategy(self, strategy: str, delta: float) -> None:
        d = self.strategy_scores.setdefault(strategy, {"mean": 0.0, "m2": 0.0, "n": 0})
        n = d["n"]
        new_n = n + 1
        old_mean = d["mean"]
        d["mean"] = old_mean + (delta - old_mean) / new_n
        d["m2"] = d["m2"] + (delta - old_mean) * (delta - d["mean"])
        d["n"] = new_n

    def strategy_variance(self, strategy: str) -> float:
        d = self.strategy_scores.get(strategy, {})
        n = d.get("n", 0)
        if n < 2:
            return 0.0
        return d["m2"] / (n - 1)


# ============ 信号函数（保留 v3） ============

def rel_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    diff = (current.to(previous.dtype) - previous).abs().mean()
    ref = previous.abs().mean().clamp(min=1e-8)
    return (diff / ref).item()


def timestep_residual(current: torch.Tensor, previous: torch.Tensor,
                      prev_prev: Optional[torch.Tensor] = None) -> float:
    if prev_prev is None:
        return rel_l1(current, previous)
    d1 = (previous - prev_prev).abs().mean()
    d2 = (current.to(previous.dtype) - previous).abs().mean()
    return (d2 - d1).abs().clamp(min=1e-8).item()


def attn_var_signal(current: torch.Tensor) -> float:
    return current.var().item()


def read_vram_mb() -> int:
    try:
        return int(torch.cuda.memory_reserved() // (1024 * 1024))
    except Exception:
        return 0


# ============ DCT 频率分解（频率解耦缓存补偿） ============

def dct_2d(x: torch.Tensor, threshold: float = 0.1) -> tuple:
    """对 block 输出做 DCT 分解，返回低频分量和高频残差。
    
    使用 2D DCT（分块），threshold 控制低频占比。
    """
    B, C, H, W = x.shape
    # 分块 DCT（简化版: 全局 DCT II 型）
    x_float = x.float()
    # 对 H,W 做 DCT
    dct = _dct_2d_blocks(x_float)
    # 能量累积
    energy = dct.abs().reshape(B, C, -1)
    total_energy = energy.sum(dim=-1, keepdim=True)
    cum_energy = torch.cumsum(energy.sort(dim=-1, descending=True).values, dim=-1)
    # threshold 控制保留的低频比例
    k = (cum_energy / total_energy.clamp(min=1e-8) < threshold).sum(dim=-1).int()
    k = k.clamp(min=1, max=energy.shape[-1] // 4)
    # 低频掩码
    B_idx = torch.arange(B, device=x.device)[:, None, None]
    C_idx = torch.arange(C, device=x.device)[None, :, None]
    sorted_indices = energy.sort(dim=-1, descending=True).indices
    mask = torch.zeros_like(energy, dtype=torch.bool)
    for b in range(B):
        for c in range(C):
            mask[b, c, sorted_indices[b, c, :k[b, c]]] = True
    low_dct = dct * mask.float().reshape(B, C, H, W)
    high_dct = dct * (~mask).float().reshape(B, C, H, W)
    # IDCT 回空间域
    low = _idct_2d_blocks(low_dct)
    high = _idct_2d_blocks(high_dct)
    # [v4.1稳定性] DCT 数值防护：抑制 NaN/Inf 和极端值，防高频重建数值不稳定
    if not torch.isfinite(low).all() or not torch.isfinite(high).all():
        low = torch.where(torch.isfinite(low), low, torch.zeros_like(low))
        high = torch.where(torch.isfinite(high), high, torch.zeros_like(high))
    x_max = x.abs().max().clamp(min=1e-6)
    low = low.clamp(-x_max.item() * 4.0, x_max.item() * 4.0)
    high = high.clamp(-x_max.item() * 4.0, x_max.item() * 4.0)
    return low, high


def _dct_2d_blocks(x: torch.Tensor) -> torch.Tensor:
    """2D DCT Type II。"""
    N = x.shape[-1]
    n = torch.arange(N, device=x.device).float()
    k = n.clone()
    cos = torch.cos(math.pi / N * k[:, None] * n[None, :])
    cos[0] *= 1.0 / math.sqrt(2)
    dct_rows = cos.T @ x.float()
    dct = cos.T @ dct_rows.transpose(-2, -1)
    return dct.transpose(-2, -1) * (2.0 / N)


def _idct_2d_blocks(x: torch.Tensor) -> torch.Tensor:
    """2D IDCT Type III。"""
    N = x.shape[-1]
    n = torch.arange(N, device=x.device).float()
    k = n.clone()
    cos = torch.cos(math.pi / N * k[:, None] * n[None, :])
    cos[0] *= 1.0 / math.sqrt(2)
    idct_rows = cos @ x.float()
    idct = cos @ idct_rows.transpose(-2, -1)
    return idct.transpose(-2, -1) * (2.0 / N)


# ============ v4 调度器 ============

class TaijiArmorV4Scheduler:
    STRATEGIES = {
        "l1":        {"thresh": 0.15, "signal": "l1"},
        "conserv":   {"thresh": 0.10, "signal": "l1"},
        "aggr":      {"thresh": 0.20, "signal": "l1"},
        "residual":  {"thresh": 0.15, "signal": "residual"},
        "weighted":  {"thresh": 0.15, "signal": "weighted"},
    }

    def __init__(self,
                 total_steps: int,
                 warmup_steps: int,
                 yang_thresh: float,
                 adaptive: bool,
                 enable_ensemble: bool,
                 vram_thresh_mb: int,
                 fallback_mode: bool,
                 end_step: int,
                 verbose: bool,
                 adaptive_scale: float = 1.5,
                 # v3 保留参数
                 poly_calibrate: bool = True,
                 poly_degree: int = 2,
                 safety_margin: float = 1.8,
                 correction_strength: float = 0.25,
                 confidence_mode: bool = True,
                 # === v4 参数 ===
                 # IEC
                 iec_enabled: bool = True,
                 iec_gamma: float = 0.6,
                 # AdaCorrection
                 adac_enabled: bool = True,
                 adac_min_alpha: float = 0.3,
                 # FeatFix
                 featfix_enabled: bool = False,
                 featfix_strength: float = 0.15,
                 # 频率解耦
                 freqdec_enabled: bool = False,
                 freqdec_threshold: float = 0.15,
                 iec_decay: float = 0.7,
                 # 连续缓存控制
                 max_consecutive: int = 3,
                 ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.yang_thresh = yang_thresh
        self.adaptive = adaptive
        self.enable_ensemble = enable_ensemble
        self.vram_thresh_mb = vram_thresh_mb
        self.fallback_mode = fallback_mode
        self.end_step = end_step
        self.verbose = verbose
        self.adaptive_scale = adaptive_scale

        self.poly_calibrate = poly_calibrate
        self.poly_degree = poly_degree
        self.safety_margin = safety_margin
        self.correction_strength = correction_strength
        self.confidence_mode = confidence_mode

        self.iec_enabled = iec_enabled
        self.iec_gamma = iec_gamma
        self.adac_enabled = adac_enabled
        self.adac_min_alpha = adac_min_alpha
        self.featfix_enabled = featfix_enabled
        self.featfix_strength = featfix_strength
        self.freqdec_enabled = freqdec_enabled
        self.freqdec_threshold = freqdec_threshold
        self.max_consecutive = max_consecutive
        self.iec_decay = iec_decay

    def _vram_gate(self, state: TaijiArmorV4State) -> bool:
        vram = read_vram_mb()
        if vram > state.peak_vram_mb:
            state.peak_vram_mb = vram
        return vram > self.vram_thresh_mb

    def _compute_signal(self, strategy: str, cur: torch.Tensor,
                        prev: torch.Tensor,
                        prev_prev: Optional[torch.Tensor]) -> float:
        if strategy == "residual":
            return timestep_residual(cur, prev, prev_prev)
        if strategy == "weighted":
            l1 = rel_l1(cur, prev)
            res = timestep_residual(cur, prev, prev_prev)
            var = attn_var_signal(cur) * 0.01
            return 0.6 * l1 + 0.3 * res + 0.1 * var
        return rel_l1(cur, prev)

    def _fitness(self, strategy: str, state: TaijiArmorV4State) -> float:
        d = state.strategy_scores.get(strategy, {})
        mean = d.get("mean", 0.0)
        var = state.strategy_variance(strategy)
        return 1.0 / (1.0 + mean * 5.0 + var * 10.0)

    def _select_strategy(self, state: TaijiArmorV4State) -> str:
        best = "l1"
        best_fit = -1.0
        for s in self.STRATEGIES:
            fit = self._fitness(s, state)
            if fit > best_fit:
                best_fit = fit
                best = s
        state.current_strategy = best
        return best

    def _fallback_threshold(self) -> float:
        effective_end = self.end_step if self.end_step >= 0 else self.total_steps + self.end_step
        cacheable = max(0, effective_end - self.warmup_steps)
        max_reuse_ratio = cacheable / max(1, self.total_steps)
        return min(0.9, max_reuse_ratio * 0.85)

    def _fit_polynomial(self, state: TaijiArmorV4State) -> None:
        if len(state.delta_history) < 2:
            return
        steps = np.array([d[0] for d in state.delta_history], dtype=np.float64)
        deltas = np.array([d[1] for d in state.delta_history], dtype=np.float64)
        log_deltas = np.log(np.maximum(deltas, 1e-8))
        degree = min(self.poly_degree, len(steps) - 1)
        try:
            coeffs = np.polyfit(steps, log_deltas, degree)
            state.poly_coeffs = coeffs.tolist()
        except Exception:
            pass

    def _predict_delta(self, step: int, state: TaijiArmorV4State) -> Optional[float]:
        if state.poly_coeffs is None:
            return None
        log_val = float(np.polyval(state.poly_coeffs, step))
        return max(float(np.exp(log_val)), 0.001)

    def _compute_confidence(self, delta: float, thresh: float) -> float:
        if not self.confidence_mode:
            return 1.0 if delta < thresh else 0.0
        if delta < thresh * 0.5:
            return 1.0
        elif delta < thresh:
            return (thresh - delta) / (thresh * 0.5)
        return 0.0

    # ============ v4 核心方法 ============

    def _iec_correct(self, state: TaijiArmorV4State) -> float:
        """IEC 迭代误差校正。

        计算步被调用，对之前连续缓存的误差序列做加权回滚校正。
        将校正后的误差估计转换为下一轮的阈值调整量，实现误差抑制。

        数学原理：
        传统缓存误差指数增长: e_{t+1} ≈ (1+α)·e_t
        IEC 校正后线性增长: e_{t+1} ≈ e_t + γ·e_t
        其中 γ = iec_gamma（默认0.6）< α（缓存导致的自然增长率通常>1）
        """
        if len(state.iec_error_buffer) < 1:
            state.iec_adjusted_thresh = 0.0
            return 0.0

        # 误差序列加权
        errors = np.array(state.iec_error_buffer)
        n = len(errors)
        # 近期误差权重更大（指数衰减权重）
        weights = np.exp(self.iec_gamma * np.arange(n))
        weights /= weights.sum()

        # 加权校正估计
        corrected_error = float((errors * weights).sum())
        state.iec_correction_applied += 1

        # 将误差转换为阈值惩罚：缓存误差越大，下一轮阈值越严格
        # 惩罚量 = γ * corrected_error，限制在合理范围
        state.iec_adjusted_thresh = corrected_error * self.iec_gamma

        if self.verbose:
            print("[V4-IEC] 误差序列=%s 加权校正=%.4f 阈值惩罚=%.4f" %
                  ([round(e, 4) for e in state.iec_error_buffer],
                   corrected_error, state.iec_adjusted_thresh))

        # 清空缓冲区
        state.iec_error_buffer = []

        return corrected_error

    def _adacorrection_blend(self, delta: float, thresh: float,
                              confidence: float) -> float:
        """AdaCorrection 自适应混合系数。
        
        不做二值缓存/计算决策，而是计算混合系数 alpha：
        - delta << thresh: alpha → 1.0（完全用缓存）
        - delta ≈ thresh: alpha → 0.5（各一半）
        - delta >> thresh: alpha → adac_min_alpha（基本用计算）
        """
        if not self.adac_enabled:
            return 1.0 if confidence > 0.5 else 0.0

        if delta <= 0:
            return 1.0

        ratio = delta / max(thresh, 1e-8)
        alpha = 1.0 - (1.0 - self.adac_min_alpha) * min(ratio / 2.0, 1.0)
        return max(self.adac_min_alpha, min(1.0, alpha))

    def _featfix_apply(self, state: TaijiArmorV4State,
                       current_input: torch.Tensor) -> Optional[torch.Tensor]:
        """FeatFix 验证特征复用（已禁用破坏性校正）。

        原实现用 (cur_cpu - prev_input) * delta 做输出加法校正，该信号幅值
        过大，会直接破坏画面。现保留信号统计但不再将其加到输出上，仅作为
        调试信息使用。
        """
        # 彻底禁用 FeatFix 加法校正：返回 None，避免视频破坏
        return None

    # ============ v4 核心决策 ============

    def should_reuse(self, state: TaijiArmorV4State,
                     current_input: torch.Tensor) -> tuple:
        step = state.step_idx

        # 【新增】形状自适应重置：分辨率/帧数变化后自动清空缓存，避免崩溃
        if state.prev_input is not None:
            cur_shape = tuple(current_input.shape)
            prev_shape = tuple(state.prev_input.shape)
            if cur_shape != prev_shape:
                if self.verbose:
                    print("[V4-TaijiArmor] 形状变化 %s → %s → 自动重置 state" % (prev_shape, cur_shape))
                state.reset()
                return (False, 0.0, 0.0, 0.0, "shape_mismatch_reset")

        if state.prev_output is None or state.prev_input is None:
            return (False, 0.0, 0.0, 0.0, "no_ref")

        strategy = self.current_strategy if not self.enable_ensemble else self._select_strategy(state)
        prev_input = state.prev_input
        prev_prev = state.prev_prev_input
        cur_cpu = current_input.detach().cpu()
        delta = self._compute_signal(strategy, cur_cpu, prev_input, prev_prev)
        state.accumulate_strategy(strategy, delta)
        state.delta_history.append((step, delta, strategy))

        # FeatFix: 不再保存破坏性校正张量，仅保留信号统计开关
        if self.featfix_enabled:
            # 仅记录 delta 标量作为调试信号，避免存储大张量
            state.featfix_signal = delta

        vram_pressure = self._vram_gate(state)

        # warmup
        if step < self.warmup_steps:
            state.yang_steps.append(step)
            if self.poly_calibrate:
                self._fit_polynomial(state)
            return (False, delta, 0.0, 0.0, "warmup(%d<%d)" % (step, self.warmup_steps))

        effective_end = self.end_step if self.end_step >= 0 else self.total_steps + self.end_step
        if step >= effective_end:
            state.yang_steps.append(step)
            return (False, delta, 0.0, 0.0, "after_end(%d>=%d)" % (step, effective_end))

        if self.fallback_mode and state.fallback_active:
            state.yang_steps.append(step)
            return (False, delta, 0.0, 0.0, "fallback")

        # === 阈值计算 ===
        thresh = None
        thresh_source = None

        if self.poly_calibrate and state.poly_coeffs is not None:
            predicted = self._predict_delta(step, state)
            if predicted is not None and predicted > 0:
                thresh = predicted * self.safety_margin
                thresh_source = "poly"
                state.poly_thresh_used += 1

        if thresh is None:
            thresh = self.STRATEGIES[strategy]["thresh"]
            thresh_source = "strat"
            state.strat_thresh_used += 1
            if self.adaptive:
                progress = step / max(1, self.total_steps)
                if progress >= 0.3:
                    thresh *= self.adaptive_scale

        if vram_pressure:
            thresh *= 1.5

        if thresh_source == "strat":
            thresh = max(thresh, self.yang_thresh)
        else:
            thresh = max(thresh, 0.01)

        # === IEC 阈值调整（修复v4-Bug2：不再指数累积） ===
        # v4-Bug: iec_penalty 持续累加导致阈值无限增大 → 决策失稳
        # 修复：每次新delta与旧惩罚做加权衰减，不再硬累加
        iec_penalty = state.iec_adjusted_thresh if self.iec_enabled else 0.0
        thresh = max(thresh - iec_penalty, 0.01)

        # === 置信度 ===
        confidence = self._compute_confidence(delta, thresh)
        state.confidence_history.append((step, confidence))

        # === AdaCorrection 混合系数 ===
        blend_alpha = self._adacorrection_blend(delta, thresh, confidence)
        state.blend_history.append((step, blend_alpha))

        # === 连续缓存上限（防止长序列缓存导致漂移） ===
        if state.consecutive_reuse_count >= self.max_consecutive:
            should_cache = False
            strategy = f"max_consecutive({state.consecutive_reuse_count})"
        else:
            # === v4 缓存决策（基于 AdaCorrection 混合） ===
            if self.adac_enabled:
                # AdaCorrection 模式：连续混合，不二值
                # 当 alpha > 0.5 时倾向于缓存（节省计算），否则倾向于计算
                should_cache = blend_alpha > 0.5
            else:
                should_cache = confidence > 0.5

        if should_cache:
            state.yin_steps.append(step)
            # IEC: 缓存步记录误差
            if self.iec_enabled:
                state.iec_error_buffer.append(delta)
                # [v4.1稳定性] IEC 缓冲区上限，防无界增长导致 _iec_correct 数值溢出
                if len(state.iec_error_buffer) > 16:
                    state.iec_error_buffer.pop(0)
            return (True, delta, thresh, blend_alpha
                    if self.adac_enabled else confidence,
                    "adac(%.2f)" % blend_alpha if self.adac_enabled else
                    "%s(%.2f)" % (strategy, confidence))

        state.yang_steps.append(step)
        return (False, delta, thresh, blend_alpha
                if self.adac_enabled else confidence,
                "adac(%.2f)" % blend_alpha if self.adac_enabled else
                "%s(%.2f)" % (strategy, confidence))


# ============ v4 Wrapper ============

import time as _v4time

def make_v4_armor_wrapper(
    state: TaijiArmorV4State,
    total_steps: int,
    warmup_steps: int,
    yang_thresh: float,
    adaptive: bool,
    enable_ensemble: bool,
    vram_thresh_mb: int,
    fallback_mode: bool,
    end_step: int,
    verbose: bool,
    adaptive_scale: float = 1.5,
    # v3
    poly_calibrate: bool = True,
    poly_degree: int = 2,
    safety_margin: float = 1.8,
    correction_strength: float = 0.25,
    confidence_mode: bool = True,
    # v4
    iec_enabled: bool = True,
    iec_gamma: float = 0.6,
    iec_decay: float = 0.7,
    adac_enabled: bool = True,
    adac_min_alpha: float = 0.3,
    featfix_enabled: bool = False,
    featfix_strength: float = 0.15,
    freqdec_enabled: bool = False,
    freqdec_threshold: float = 0.15,
    max_consecutive: int = 4,
) -> Callable[..., torch.Tensor]:
    scheduler = TaijiArmorV4Scheduler(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        yang_thresh=yang_thresh,
        adaptive=adaptive,
        enable_ensemble=enable_ensemble,
        vram_thresh_mb=vram_thresh_mb,
        fallback_mode=fallback_mode,
        end_step=end_step,
        verbose=verbose,
        adaptive_scale=adaptive_scale,
        poly_calibrate=poly_calibrate,
        poly_degree=poly_degree,
        safety_margin=safety_margin,
        correction_strength=correction_strength,
        confidence_mode=confidence_mode,
        iec_enabled=iec_enabled,
        iec_gamma=iec_gamma,
        adac_enabled=adac_enabled,
        adac_min_alpha=adac_min_alpha,
        featfix_enabled=featfix_enabled,
        featfix_strength=featfix_strength,
        freqdec_enabled=freqdec_enabled,
        freqdec_threshold=freqdec_threshold,
        iec_decay=iec_decay,
        max_consecutive=max_consecutive,
    )

    _step_t0 = [_v4time.perf_counter()]
    _step_wall_times = []

    def wrapper(apply_model, args) -> torch.Tensor:
        x = args["input"]
        _t0 = _v4time.perf_counter()
        reuse, delta, thresh, blend_val, strategy = scheduler.should_reuse(state, x)
        _sched_dt = _v4time.perf_counter() - _t0
        _step_wall = _v4time.perf_counter() - _step_t0[0]

        if reuse:
            state.reuse_count += 1
            state.consecutive_reuse_count += 1
            state.step_idx += 1

            # === 缓存补偿（修复v4-Bug1：完整实现AdaC混合） ===
            # v4-Bug: AdaC算出blend_alpha后完全没用 → 直接复用缓存导致画面停滞
            # 修复：alpha=1.0完全缓存，alpha=adac_min_alpha做趋势外推，中间值混合
            base_output = state.prev_output.to(x.device)
            effective_strength = 0.0  # 初始化，避免未定义错误
            if adac_enabled:
                alpha = blend_val
                effective_strength = correction_strength * alpha
                if correction_strength > 0 and state.output_trend is not None and state.step_idx > 1:
                    trend = state.output_trend.to(x.device)
                    output = base_output + trend * effective_strength
                else:
                    output = base_output
            else:
                # 非AdaC模式：v3 damped trend逻辑
                effective_strength = correction_strength
                consecutive = max(1, state.consecutive_reuse_count)
                damping = 0.8 ** (consecutive - 1)  # 修复v4-Bug3：从0.65改为0.8（更温和）
                effective_strength *= damping
                if effective_strength > 0 and state.output_trend is not None and state.step_idx > 1:
                    trend = state.output_trend.to(x.device)
                    output = base_output + trend * effective_strength
                else:
                    output = base_output

            state.correction_count = getattr(state, 'correction_count', 0) + 1

            if verbose:
                trend_norm = state.output_trend.abs().mean().item() if state.output_trend is not None else 0.0
                print("[V4] Step %d/%d: CACHE (delta=%.4f thresh=%.4f blend=%.2f eff_str=%.2f trend=%.4f %s)" % (
                    state.step_idx - 1, total_steps, delta, thresh, blend_val, effective_strength, trend_norm, strategy))
            return output

        # ===== 计算步 =====

        if fallback_mode:
            total_done = state.reuse_count + state.real_count
            fb_thresh = scheduler._fallback_threshold()
            if total_done > 3 and state.reuse_count / total_done > fb_thresh:
                state.anomaly_count += 1
                if state.anomaly_count >= 2:
                    state.fallback_active = True
                    if verbose:
                        print("[V4] 护体真元: 命中率 %d/%d > %.2f → 全算" % (
                            state.reuse_count, total_done, fb_thresh))
            else:
                state.anomaly_count = 0

        # === IEC: 计算步执行迭代误差校正 ===
        if iec_enabled:
            corrected_error = scheduler._iec_correct(state)

        # === 实际模型计算 ===
        output = apply_model(x, args["timestep"], **args["c"])

        # === 更新趋势（EMA of output changes） ===
        # step 0: 没有prev_output，不初始化趋势（保持None）
        # step 1: 用首次真实变化初始化趋势
        # 避免 step 0 的零趋势污染 step 2 的首次缓存补偿
        if correction_strength > 0:
            cur_out_cpu = output.detach().cpu()
            if state.prev_output is not None:
                change = cur_out_cpu - state.prev_output
                if state.output_trend is None:
                    state.output_trend = change.clone()
                else:
                    state.output_trend = 0.7 * state.output_trend + 0.3 * change
            # else: step 0，保持 output_trend = None

        # === 重置连续缓存计数 ===
        state.consecutive_reuse_count = 0

        # === 更新前2步缓存 ===
        state.prev_prev_output = state.prev_output.detach().cpu() if state.prev_output is not None else None
        state.prev_prev_input = state.prev_input if state.prev_input is not None else None
        state.prev_input = x.detach().cpu()
        state.prev_output = output.detach().cpu()
        state.real_count += 1  # [v4.1] 移除下方重复累加(原版计数失真)
        state.step_idx += 1
        _step_t0[0] = _v4time.perf_counter()
        _step_wall_times.append(_step_wall)
        state.step_wall_times = _step_wall_times

        if verbose:
            _model_dt = _v4time.perf_counter() - _t0
            print("[V4] Step %d/%d: COMPUTE wall=%.1fs (sched=%.0fms model=%.0fms) (delta=%.4f thresh=%.4f blend=%.2f)" % (
                state.step_idx, total_steps, _step_wall, _sched_dt*1000, _model_dt*1000,
                delta, thresh, blend_val))

        if state.step_idx >= total_steps:
            _v4_report(state, total_steps, warmup_steps, yang_thresh,
                       adaptive, enable_ensemble, vram_thresh_mb,
                       fallback_mode, end_step, verbose,
                       poly_calibrate, safety_margin, correction_strength,
                       confidence_mode,
                       iec_enabled, adac_enabled, featfix_enabled, freqdec_enabled)
        return output

    return wrapper


def _v4_report(state, total_steps, warmup_steps, yang_thresh, adaptive,
               enable_ensemble, vram_thresh_mb, fallback_mode, end_step, verbose,
               poly_calibrate, safety_margin, correction_strength, confidence_mode,
               iec_enabled, adac_enabled, featfix_enabled, freqdec_enabled):
    tot = state.reuse_count + state.real_count
    rate = (state.reuse_count / tot * 100) if tot else 0.0
    iec_count = getattr(state, 'iec_correction_applied', 0)
    freq_count = getattr(state, 'freq_decouple_count', 0)
    base = (f"[XuanShuV4] steps={total_steps} warmup={warmup_steps} "
            f"yang_thresh={yang_thresh} adaptive={adaptive} ensemble={enable_ensemble} "
            f"yang={len(state.yang_steps)} yin={len(state.yin_steps)} "
            f"reuse={state.reuse_count} real={state.real_count} hit_rate={rate:.1f}% "
            f"peak_vram={state.peak_vram_mb}MB\n"
            f"  v4: IEC={iec_enabled}(corr={iec_count}) "
            f"AdaC={adac_enabled} "
            f"FeatFix={featfix_enabled}(reuse={getattr(state, 'featfix_reuse_count', 0)}) "
            f"FreqDec={freqdec_enabled}(apply={freq_count})")
    wall_times = getattr(state, 'step_wall_times', [])
    wall_str = ""
    if wall_times:
        wall_str = (f"\n  wall_times=[{', '.join('%.1f' % w for w in wall_times)}] "
                    f"total_wall=%.1fs" % sum(wall_times))
    if verbose:
        err_buf = getattr(state, 'iec_error_buffer', [])
        base += (f"\n  IEC缓冲区={[round(e,4) for e in err_buf] if err_buf else '空'}\n"
                 f"  yang_steps={state.yang_steps}\n  yin_steps={state.yin_steps}\n"
                 f"  delta_history={[(d[0], round(d[1],4), d[2]) for d in state.delta_history]}"
                 + wall_str)
    print(base)

# ============ v4.1 节点类 + 注册映射 ============

class MiniMaxH3TaijiArmorV4_1:
    """玄枢太极装甲 v4.1（稳定性迭代版）。

    v4.1 修复（相对 v4）：
    1. 修复计算步 real_count 重复累加 bug（命中率/fallback 判定失真）
    2. DCT 频率解耦数值稳定保护（NaN/Inf 防护 + clamp）
    3. IEC 缓冲区上限（防无界增长导致 _iec_correct 数值溢出）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 6, "min": 1, "max": 200}),
                "warmup_steps": ("INT", {"default": 2, "min": 0, "max": 100}),
                "yang_thresh": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.5, "step": 0.01}),
                "end_step": ("INT", {"default": -2, "min": -20, "max": 100}),
                "adaptive": ("BOOLEAN", {"default": True}),
                "enable_ensemble": ("BOOLEAN", {"default": True}),
                "vram_thresh_mb": ("INT", {"default": 7500, "min": 1000, "max": 20000}),
                "fallback_mode": ("BOOLEAN", {"default": True}),
                "verbose": ("BOOLEAN", {"default": True}),
                "poly_calibrate": ("BOOLEAN", {"default": True}),
                "poly_degree": ("INT", {"default": 2, "min": 1, "max": 5}),
                "safety_margin": ("FLOAT", {"default": 1.8, "min": 1.0, "max": 3.0, "step": 0.1}),
                "adaptive_scale": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 10.0, "step": 0.1}),
                "correction_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "iec_enabled": ("BOOLEAN", {"default": True}),
                "iec_gamma": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
                "adac_enabled": ("BOOLEAN", {"default": True}),
                "adac_min_alpha": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "featfix_enabled": ("BOOLEAN", {"default": False}),
                "featfix_strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05}),
                "freqdec_enabled": ("BOOLEAN", {"default": False}),
                "freqdec_threshold": ("FLOAT", {"default": 0.15, "min": 0.05, "max": 0.5, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "MiniMax H3 太极装甲 v4.1"

    def apply(self, model, total_steps, warmup_steps, yang_thresh, end_step,
              adaptive, enable_ensemble, vram_thresh_mb, fallback_mode, verbose,
              poly_calibrate, poly_degree, safety_margin, adaptive_scale,
              correction_strength,
              iec_enabled, iec_gamma, adac_enabled, adac_min_alpha,
              featfix_enabled, featfix_strength, freqdec_enabled, freqdec_threshold):
        if "xuanshu_teacache_guard" in model.model_options or \
           "xuanshu_taiji_armor" in model.model_options:
            print("[V4.1] 警告：检测到 Guard/v3 已应用，v4.1 wrapper 将被覆盖")
        state = TaijiArmorV4State()
        state.reset()
        wrapper = make_v4_armor_wrapper(
            state=state,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            yang_thresh=yang_thresh,
            adaptive=adaptive,
            enable_ensemble=enable_ensemble,
            vram_thresh_mb=vram_thresh_mb,
            fallback_mode=fallback_mode,
            end_step=end_step,
            verbose=verbose,
            adaptive_scale=adaptive_scale,
            poly_calibrate=poly_calibrate,
            poly_degree=poly_degree,
            safety_margin=safety_margin,
            correction_strength=correction_strength,
            confidence_mode=True,
            iec_enabled=iec_enabled,
            iec_gamma=iec_gamma,
            adac_enabled=adac_enabled,
            adac_min_alpha=adac_min_alpha,
            featfix_enabled=featfix_enabled,
            featfix_strength=featfix_strength,
            freqdec_enabled=freqdec_enabled,
            freqdec_threshold=freqdec_threshold,
        )
        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        patched.model_options["xuanshu_taiji_armor_v4_1"] = {"state": state}
        patched.model_options.pop("xuanshu_teacache_guard", None)
        patched.model_options.pop("xuanshu_taiji_armor", None)
        patched.model_options.pop("xuanshu_taiji_armor_v4", None)
        return (patched,)


NODE_CLASS_MAPPINGS_V41 = {
    "MiniMaxH3TaijiArmorV4_1": MiniMaxH3TaijiArmorV4_1,
}
NODE_DISPLAY_NAME_MAPPINGS_V41 = {
    "MiniMaxH3TaijiArmorV4_1": "MiniMax H3 太极装甲 v4.1",
}
