"""XuanShu H3 太极装甲 v3 — 多项式校准 + 置信度衰减 + 误差熔断 + 缓存补偿。

v2 修复（保留）：
1. nvidia-smi → torch.cuda.memory_reserved()
2. Welford 方差归一化修正
3. yang_thresh 接入判定
4. timestep_residual 真时序差分
5. fallback 阈值动态调整
6. end_step 末段保护
7. weighted 策略权重修正

v2.1 修复（保留）：
8. prev_output 存 CPU
9. 去掉 .float() 临时张量
10. delta 计算在 CPU
11. should_reuse 返回元组 + verbose
12. prev_prev_input 引用已CPU化的 prev_input

v3 新增：
13. 多项式校准：warmup期delta拟合多项式，逐步预测阈值（替代统一倍率）
14. 置信度衰减：连续缓存概率递减，替代二值决策
15. 误差累积熔断：连续缓存误差超限时强制计算
16. 缓存补偿：趋势外推修正（轻量替代LoRA训练）
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import numpy as np
import torch


@dataclasses.dataclass
class TaijiArmorState:
    step_idx: int = 0
    prev_input: Optional[torch.Tensor] = None
    prev_output: Optional[torch.Tensor] = None
    prev_prev_input: Optional[torch.Tensor] = None

    yang_steps: list = dataclasses.field(default_factory=list)
    yin_steps: list = dataclasses.field(default_factory=list)
    reuse_count: int = 0
    real_count: int = 0

    strategy_scores: dict = dataclasses.field(default_factory=dict)
    current_strategy: str = "l1"

    anomaly_count: int = 0
    fallback_active: bool = False

    peak_vram_mb: int = 0

    # v3 新增
    delta_history: list = dataclasses.field(default_factory=list)
    poly_coeffs: Optional[list] = None
    accumulated_error: float = 0.0
    consecutive_cache: int = 0
    max_consecutive_hit: int = 0
    output_trend: Optional[torch.Tensor] = None
    correction_count: int = 0
    confidence_history: list = dataclasses.field(default_factory=list)
    poly_thresh_used: int = 0
    strat_thresh_used: int = 0

    # wall-clock timing
    step_wall_cumulative: float = 0.0
    step_wall_times: list = dataclasses.field(default_factory=list)

    def reset(self) -> None:
        self.step_idx = 0
        self.prev_input = None
        self.prev_output = None
        self.prev_prev_input = None
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
        self.accumulated_error = 0.0
        self.consecutive_cache = 0
        self.max_consecutive_hit = 0
        self.output_trend = None
        self.correction_count = 0
        self.confidence_history = []
        self.poly_thresh_used = 0
        self.strat_thresh_used = 0
        self.step_wall_cumulative = 0.0
        self.step_wall_times = []

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


class TaijiScheduler:
    STRATEGIES = {
        "l1":        {"thresh": 0.15, "signal": "l1"},
        "conserv":   {"thresh": 0.10, "signal": "l1"},
        "aggr":      {"thresh": 0.20, "signal": "l1"},
        "residual":  {"thresh": 0.15, "signal": "residual"},
        "weighted":  {"thresh": 0.15, "signal": "weighted"},
    }

    def __init__(self, total_steps: int, warmup_steps: int, yang_thresh: float,
                 adaptive: bool, enable_ensemble: bool, vram_thresh_mb: int,
                 fallback_mode: bool, end_step: int, verbose: bool,
                 adaptive_scale: float = 2.5,
                 # v3 参数
                 poly_calibrate: bool = True,
                 poly_degree: int = 2,
                 safety_margin: float = 1.3,
                 error_budget: float = 0.5,
                 max_consecutive: int = 4,
                 correction_strength: float = 0.3,
                 confidence_mode: bool = True):
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
        self.error_budget = error_budget
        self.max_consecutive = max_consecutive
        self.correction_strength = correction_strength
        self.confidence_mode = confidence_mode

    def _vram_gate(self, state: TaijiArmorState) -> bool:
        vram = read_vram_mb()
        if vram > state.peak_vram_mb:
            state.peak_vram_mb = vram
        pressure = vram > self.vram_thresh_mb
        if self.verbose and pressure:
            print("[TaijiArmor] VRAM水位: %dMB > %dMB → 阈值×1.5" % (vram, self.vram_thresh_mb))
        return pressure

    def _compute_signal(self, strategy: str, cur: torch.Tensor,
                        prev: torch.Tensor, prev_prev: Optional[torch.Tensor]) -> float:
        if strategy == "residual":
            return timestep_residual(cur, prev, prev_prev)
        if strategy == "weighted":
            l1 = rel_l1(cur, prev)
            res = timestep_residual(cur, prev, prev_prev)
            var = attn_var_signal(cur) * 0.01
            return 0.6 * l1 + 0.3 * res + 0.1 * var
        return rel_l1(cur, prev)

    def _fitness(self, strategy: str, state: TaijiArmorState) -> float:
        d = state.strategy_scores.get(strategy, {})
        mean = d.get("mean", 0.0)
        var = state.strategy_variance(strategy)
        return 1.0 / (1.0 + mean * 5.0 + var * 10.0)

    def _select_strategy(self, state: TaijiArmorState) -> str:
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

    def _fit_polynomial(self, state: TaijiArmorState) -> None:
        """用 delta_history 拟合对数线性模型（指数增长预测）。

        delta 随去噪步数呈指数增长，因此拟合 log(delta) = a*step + b，
        预测时 delta = exp(a*step + b)。
        """
        if len(state.delta_history) < 2:
            return
        steps = np.array([d[0] for d in state.delta_history], dtype=np.float64)
        deltas = np.array([d[1] for d in state.delta_history], dtype=np.float64)
        log_deltas = np.log(np.maximum(deltas, 1e-8))
        degree = min(self.poly_degree, len(steps) - 1)
        try:
            coeffs = np.polyfit(steps, log_deltas, degree)
            state.poly_coeffs = coeffs.tolist()
            if self.verbose:
                predicted_strs = []
                for s in range(int(steps.min()), int(steps.max()) + 4):
                    pred = float(np.exp(np.polyval(coeffs, s)))
                    predicted_strs.append("s%d→%.4f" % (s, pred))
                print("[TaijiArmor] 对数线性拟合(deg=%d, n=%d): %s" % (
                    degree, len(steps), " ".join(predicted_strs)))
        except Exception as e:
            if self.verbose:
                print("[TaijiArmor] 对数线性拟合失败: %s" % e)

    def _predict_delta(self, step: int, state: TaijiArmorState) -> Optional[float]:
        """用对数线性模型预测指定步的delta（指数增长）。"""
        if state.poly_coeffs is None:
            return None
        log_val = float(np.polyval(state.poly_coeffs, step))
        return max(float(np.exp(log_val)), 0.001)

    def _compute_confidence(self, delta: float, thresh: float) -> float:
        """计算缓存置信度 [0, 1]。"""
        if not self.confidence_mode:
            return 1.0 if delta < thresh else 0.0
        if delta < thresh * 0.5:
            return 1.0
        elif delta < thresh:
            return (thresh - delta) / (thresh * 0.5)
        return 0.0

    def should_reuse(self, state: TaijiArmorState, current_input: torch.Tensor) -> tuple:
        step = state.step_idx

        if state.prev_output is None or state.prev_input is None:
            return (False, 0.0, 0.0, 0.0, "no_ref")

        # === 形状自适应重置：分辨率/帧数变化后自动清空缓存 ===
        cur_shape = tuple(current_input.shape)
        prev_shape = tuple(state.prev_input.shape)
        if cur_shape != prev_shape:
            if self.verbose:
                print("[TaijiArmor] 形状变化 %s → %s → 自动重置 state" % (prev_shape, cur_shape))
            state.reset()
            return (False, 0.0, 0.0, 0.0, "shape_mismatch_reset")

        # 始终计算 delta（用于数据收集）
        strategy = self.current_strategy if not self.enable_ensemble else self._select_strategy(state)
        prev_input = state.prev_input
        prev_prev = state.prev_prev_input
        cur_cpu = current_input.detach().cpu()
        delta = self._compute_signal(strategy, cur_cpu, prev_input, prev_prev)
        state.accumulate_strategy(strategy, delta)
        state.delta_history.append((step, delta, strategy))

        vram_pressure = self._vram_gate(state)

        # warmup 检查（但仍记录 delta 用于多项式拟合）
        if step < self.warmup_steps:
            state.yang_steps.append(step)
            # warmup 期间逐步拟合多项式
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

        # === 阈值计算：多项式优先，回退策略 ===
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

        # 多项式阈值不受 yang_thresh 地板限制（让早步用紧阈值防过缓存噪声）
        # 策略回退阈值仍受 yang_thresh 保护
        if thresh_source == "strat":
            thresh = max(thresh, self.yang_thresh)
        else:
            thresh = max(thresh, 0.01)

        # === 误差累积熔断 ===
        if state.accumulated_error + delta > self.error_budget and state.consecutive_cache > 0:
            if self.verbose:
                print("[TaijiArmor] 误差熔断: 累积=%.3f + 当前=%.3f > 预算=%.3f → 强制计算" % (
                    state.accumulated_error, delta, self.error_budget))
            state.accumulated_error = 0.0
            state.consecutive_cache = 0
            state.yang_steps.append(step)
            return (False, delta, thresh, 0.0, "error_break(%.2f)" % (state.accumulated_error + delta))

        # === 连续缓存上限 ===
        if state.consecutive_cache >= self.max_consecutive:
            if self.verbose:
                print("[TaijiArmor] 连续缓存上限: %d >= %d → 强制计算" % (
                    state.consecutive_cache, self.max_consecutive))
            state.consecutive_cache = 0
            state.accumulated_error = 0.0
            state.yang_steps.append(step)
            return (False, delta, thresh, 0.0, "max_consec(%d)" % self.max_consecutive)

        # === 置信度计算 ===
        confidence = self._compute_confidence(delta, thresh)
        state.confidence_history.append((step, confidence))

        # === 缓存决策 ===
        if confidence > 0.5:
            state.yin_steps.append(step)
            state.accumulated_error += delta
            state.consecutive_cache += 1
            if state.consecutive_cache > state.max_consecutive_hit:
                state.max_consecutive_hit = state.consecutive_cache
            return (True, delta, thresh, confidence, "%s(%.2f)" % (strategy, confidence))

        state.yang_steps.append(step)
        state.consecutive_cache = 0
        return (False, delta, thresh, confidence, "%s(%.2f)" % (strategy, confidence))


def make_armor_wrapper(
    state: TaijiArmorState,
    total_steps: int,
    warmup_steps: int,
    yang_thresh: float,
    adaptive: bool,
    enable_ensemble: bool,
    vram_thresh_mb: int,
    fallback_mode: bool,
    end_step: int,
    verbose: bool,
    adaptive_scale: float = 2.5,
    # v3 参数
    poly_calibrate: bool = True,
    poly_degree: int = 2,
    safety_margin: float = 1.3,
    error_budget: float = 0.5,
    max_consecutive: int = 4,
    correction_strength: float = 0.3,
    confidence_mode: bool = True,
) -> Callable[..., torch.Tensor]:
    scheduler = TaijiScheduler(
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
        error_budget=error_budget,
        max_consecutive=max_consecutive,
        correction_strength=correction_strength,
        confidence_mode=confidence_mode,
    )

    import time as _time
    _step_t0 = [_time.perf_counter()]  # 初始化为wrapper创建时刻，step1测量wrapper_call→step1_done的时间

    def wrapper(apply_model, args) -> torch.Tensor:
        x = args["input"]
        _t0 = _time.perf_counter()
        # 首次调用：测量本 step 从开始到结束的总耗时
        if _step_t0[0] is None:
            _step_t0[0] = _t0
        reuse, delta, thresh, confidence, strategy = scheduler.should_reuse(state, x)
        _sched_dt = _time.perf_counter() - _t0
        _step_wall = _time.perf_counter() - _step_t0[0]  # 本 step 总耗时（不含本 step 之后的 VAEDecode 等）

        if reuse:
            state.reuse_count += 1
            state.step_wall_times.append(_step_wall)
            state.step_wall_cumulative = (state.step_wall_cumulative or 0.0) + _step_wall

            # === 缓存补偿：趋势外推 ===
            if correction_strength > 0 and state.output_trend is not None:
                trend = state.output_trend
                corrected = state.prev_output + trend * correction_strength
                state.correction_count += 1
                if verbose:
                    print("[TaijiArmor] Step %d/%d: CACHE+修正 wall=%.1fs (delta=%.4f thresh=%.4f conf=%.2f strat=%s trend_norm=%.4f)" % (
                        state.step_idx - 1, total_steps, _step_wall, delta, thresh, confidence, strategy,
                        trend.abs().mean().item()))
                return corrected.to(x.device)
            else:
                if verbose:
                    print("[TaijiArmor] Step %d/%d: CACHE wall=%.1fs (delta=%.4f thresh=%.4f conf=%.2f strat=%s)" % (
                        state.step_idx - 1, total_steps, _step_wall, delta, thresh, confidence, strategy))
                return state.prev_output.to(x.device)

        # 计算步
        if fallback_mode:
            total_done = state.reuse_count + state.real_count
            fb_thresh = scheduler._fallback_threshold()
            if total_done > 3 and state.reuse_count / total_done > fb_thresh:
                state.anomaly_count += 1
                if state.anomaly_count >= 2:
                    state.fallback_active = True
                    if verbose:
                        print("[TaijiArmor] 护体真元触发：命中率 "
                              "%d/%d > %.2f → 回退全算" % (
                              state.reuse_count, total_done, fb_thresh))
            else:
                state.anomaly_count = 0

        output = apply_model(x, args["timestep"], **args["c"])

        # === 更新趋势（EMA of output changes） ===
        if state.prev_output is not None and correction_strength > 0:
            prev_out = state.prev_output
            cur_out_cpu = output.detach().cpu()
            change = cur_out_cpu - prev_out
            if state.output_trend is None:
                state.output_trend = change
            else:
                state.output_trend = 0.7 * state.output_trend + 0.3 * change

        state.prev_prev_input = state.prev_input if state.prev_input is not None else None
        state.prev_input = x.detach().cpu()
        state.prev_output = output.detach().cpu()
        state.real_count += 1
        state.step_wall_times.append(_step_wall)
        state.step_wall_cumulative = (state.step_wall_cumulative or 0.0) + _step_wall
        _model_dt = _time.perf_counter() - _t0
        if verbose:
            print("[TaijiArmor] Step %d/%d: COMPUTE wall=%.1fs (sched=%.0fms model=%.0fms) (%s delta=%.4f thresh=%.4f conf=%.2f)" % (
                state.step_idx, total_steps, _step_wall, _sched_dt*1000, _model_dt*1000,
                strategy, delta, thresh, confidence))
        state.step_idx += 1
        _step_t0[0] = _time.perf_counter()  # 重置为 step 结束时，下次测量从此刻开始
        if state.step_idx >= total_steps:
            _report(state, total_steps, warmup_steps, yang_thresh,
                    adaptive, enable_ensemble, vram_thresh_mb,
                    fallback_mode, end_step, verbose,
                    poly_calibrate, safety_margin, error_budget,
                    max_consecutive, correction_strength, confidence_mode)
        return output

    return wrapper


def _report(state, total_steps, warmup_steps, yang_thresh, adaptive,
            enable_ensemble, vram_thresh_mb, fallback_mode, end_step, verbose,
            poly_calibrate, safety_margin, error_budget,
            max_consecutive, correction_strength, confidence_mode):
    tot = state.reuse_count + state.real_count
    rate = (state.reuse_count / tot * 100) if tot else 0.0
    wall_total = state.step_wall_cumulative
    wall_times_str = ""
    if state.step_wall_times:
        wall_times_str = " | wall_times=[" + ", ".join("%.1f" % t for t in state.step_wall_times) + "]"
    base = (f"[XuanShuTaijiArmor v3] steps={total_steps} warmup={warmup_steps} "
            f"yang_thresh={yang_thresh} adaptive={adaptive} ensemble={enable_ensemble} "
            f"vram_th={vram_thresh_mb}MB fallback={fallback_mode} end_step={end_step} "
            f"yang={len(state.yang_steps)} yin={len(state.yin_steps)} "
            f"reuse={state.reuse_count} real={state.real_count} hit_rate={rate:.1f}% "
            f"peak_vram={state.peak_vram_mb}MB | total_wall={wall_total:.1f}s\n"
            f"  v3: poly={poly_calibrate} safety={safety_margin} err_budget={error_budget} "
            f"max_consec={max_consecutive} correction={correction_strength} conf_mode={confidence_mode}\n"
            f"  v3: poly_thresh_used={state.poly_thresh_used} strat_thresh_used={state.strat_thresh_used} "
            f"max_consec_hit={state.max_consecutive_hit} corrections={state.correction_count} "
            f"final_accumulated_err={state.accumulated_error:.4f}{wall_times_str}")
    if verbose:
        base += (f"\n  yang_steps={state.yang_steps}\n  yin_steps={state.yin_steps}\n"
                 f"  strategy={state.current_strategy}\n"
                 f"  delta_history={[(d[0], round(d[1],4), d[2]) for d in state.delta_history]}\n"
                 f"  confidence={[(c[0], round(c[1],2)) for c in state.confidence_history]}")
    print(base)
