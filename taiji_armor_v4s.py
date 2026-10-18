"""XuanShu H3 太极装甲 v4s — 末梦秋花v4(顺风飘)：轨迹确定性版。

v4s = v3s1 正确性四件套（timestep 真实步对齐 / prev_output_shape 形状保护 /
      整批分块保守 delta / 自定义采样器旁路）全保留
    + 轨迹确定性重构（目标：干掉护体真元导致的人物不一致）：

1. 静态计划表（Static Plan）：warmup 探针阶段全算收集 delta，
   结束后一次性锁定 plan_thresh = max(yang_thresh, median(delta)×safety_margin)，
   执行阶段阈值恒定，同一步永远得到同一决策 —— 消灭状态累积导致的
   中途行为漂移（护体回退 / 逐步调阈 / IEC 惩罚的分叉根源）。
2. 护体真元降级为纯监控：fallback_mode 保留参数，命中异常只打日志，
   永不改变执行路径。
3. 养蛊择优锁定一次：策略在探针结束后固定，执行阶段不再切换。
4. 移除全部状态驱动机制：IEC / 多项式重拟合 / 置信度衰减 / 误差熔断。
5. 身份一致性锁：mode=r2v（参考视频 / 图生视频续写）时直接禁用缓存全算，
   参考对齐不被跳帧破坏；mode=t2v（纯文生视频）才启用确定性缓存。
6. 连续缓存上限收紧：max_consecutive 默认 2，任何时刻最多连跳 2 步。
7. 缓存输出趋势插值：prev_output + trend×correction_strength，
   保持 latent 轨迹连续，消除跳帧跳变。

用法建议：
- 纯文生视频（t2v）：mode=t2v，可正常获得确定性缓存加速。
- 图生视频 / 参考续写（r2v）：mode=r2v，全程全算，人物一致性优先。
- 混用参考锚定的工作流若仍要加速：把 max_consecutive 保持 2 且
  correction_strength 建议 0.2~0.3，避免长段轨迹漂移。
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import numpy as np
import torch


@dataclasses.dataclass
class TaijiArmorV4SState:
    step_idx: int = 0
    last_ts: Optional[float] = None
    reported: bool = False
    prev_input: Optional[torch.Tensor] = None
    prev_output: Optional[torch.Tensor] = None
    prev_output_shape: Optional[tuple] = None
    prev_prev_input: Optional[torch.Tensor] = None
    prev_input_shape: Optional[tuple] = None

    yang_steps: list = dataclasses.field(default_factory=list)
    yin_steps: list = dataclasses.field(default_factory=list)
    reuse_count: int = 0
    real_count: int = 0

    strategy_scores: dict = dataclasses.field(default_factory=dict)
    current_strategy: str = "l1"

    anomaly_count: int = 0
    fallback_active: bool = False

    peak_vram_mb: int = 0
    identity_locked_steps: int = 0
    resolved_mode: Optional[str] = None

    # v4s 静态计划
    plan_ready: bool = False
    plan_thresh: float = 0.0
    plan_strategy: str = "l1"
    probe_deltas: list = dataclasses.field(default_factory=list)

    # 缓存纪律
    consecutive_cache: int = 0
    max_consecutive_hit: int = 0
    output_trend: Optional[torch.Tensor] = None
    correction_count: int = 0

    # wall-clock timing
    step_wall_cumulative: float = 0.0
    step_wall_times: list = dataclasses.field(default_factory=list)

    def reset(self, keep_timeline: bool = False) -> None:
        step_idx_keep = self.step_idx
        last_ts_keep = self.last_ts
        reported_keep = self.reported
        self.step_idx = 0
        self.last_ts = None
        self.reported = False
        self.prev_input = None
        self.prev_output = None
        self.prev_output_shape = None
        self.prev_prev_input = None
        self.prev_input_shape = None
        self.yang_steps = []
        self.yin_steps = []
        self.reuse_count = 0
        self.real_count = 0
        self.strategy_scores = {}
        self.current_strategy = "l1"
        self.anomaly_count = 0
        self.fallback_active = False
        self.peak_vram_mb = 0
        self.identity_locked_steps = 0
        self.resolved_mode = None
        self.plan_ready = False
        self.plan_thresh = 0.0
        self.plan_strategy = "l1"
        self.probe_deltas = []
        self.consecutive_cache = 0
        self.max_consecutive_hit = 0
        self.output_trend = None
        self.correction_count = 0
        self.step_wall_cumulative = 0.0
        self.step_wall_times = []
        if keep_timeline:
            self.step_idx = step_idx_keep
            self.last_ts = last_ts_keep
            self.reported = reported_keep

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


class TaijiSchedulerV4S:
    STRATEGIES = {
        "l1":        {"thresh": 0.15, "signal": "l1"},
        "conserv":   {"thresh": 0.10, "signal": "l1"},
        "aggr":      {"thresh": 0.20, "signal": "l1"},
        "residual":  {"thresh": 0.15, "signal": "residual"},
        "weighted":  {"thresh": 0.15, "signal": "weighted"},
    }

    def __init__(self, total_steps: int, warmup_steps: int, yang_thresh: float,
                 enable_ensemble: bool, vram_thresh_mb: int, fallback_mode: bool,
                 end_step: int, verbose: bool, safety_margin: float,
                 correction_strength: float, max_consecutive: int,
                 chunk_size: int, mode: str):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.yang_thresh = yang_thresh
        self.enable_ensemble = enable_ensemble
        self.vram_thresh_mb = vram_thresh_mb
        self.fallback_mode = fallback_mode
        self.end_step = end_step
        self.verbose = verbose
        self.safety_margin = safety_margin
        self.correction_strength = correction_strength
        self.max_consecutive = max_consecutive
        self.chunk_size = max(1, int(chunk_size))
        self.mode = mode

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

    def _compute_signal_chunked(self, strategy: str, cur: torch.Tensor,
                                prev: torch.Tensor,
                                prev_prev: Optional[torch.Tensor]) -> float:
        b = cur.shape[0]
        if b <= 1 or b <= self.chunk_size:
            return self._compute_signal(strategy, cur, prev, prev_prev)
        deltas = []
        for i in range(0, b, self.chunk_size):
            c = cur[i:i + self.chunk_size]
            p = prev[i:i + self.chunk_size]
            pp = prev_prev[i:i + self.chunk_size] if prev_prev is not None else None
            deltas.append(self._compute_signal(strategy, c, p, pp))
        return max(deltas)

    def _fitness(self, strategy: str, state: TaijiArmorV4SState) -> float:
        d = state.strategy_scores.get(strategy, {})
        mean = d.get("mean", 0.0)
        var = state.strategy_variance(strategy)
        return 1.0 / (1.0 + mean * 5.0 + var * 10.0)

    def _select_strategy(self, state: TaijiArmorV4SState) -> str:
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

    def _finalize_plan(self, state: TaijiArmorV4SState) -> None:
        """探针结束：一次性锁定 plan_thresh 与 plan_strategy（执行阶段恒定）。"""
        if state.plan_ready:
            return
        if len(state.probe_deltas) >= 1:
            vals = np.array([d[1] for d in state.probe_deltas], dtype=np.float64)
            median = float(np.median(vals))
            state.plan_thresh = max(self.yang_thresh, median * self.safety_margin)
        else:
            state.plan_thresh = self.yang_thresh
        state.plan_strategy = self._select_strategy(state) if self.enable_ensemble else "l1"
        state.plan_ready = True
        if self.verbose:
            med = float(np.median([d[1] for d in state.probe_deltas])) if state.probe_deltas else 0.0
            print("[TaijiArmor v4s] 静态计划锁定: plan_thresh=%.4f (median=%.4f × safety=%.2f, "
                  "yang_thresh=%.2f) strategy=%s —— 执行阶段阈值恒定" % (
                      state.plan_thresh, med, self.safety_margin, self.yang_thresh,
                      state.plan_strategy))

    def should_reuse(self, state: TaijiArmorV4SState, current_input: torch.Tensor) -> tuple:
        step = state.step_idx

        # 形状变化保护（保留 v3s1 正确性修复）
        cur_shape = tuple(current_input.shape)
        if state.prev_input_shape is not None and cur_shape != state.prev_input_shape:
            if self.verbose:
                print("[TaijiArmor v4s] 形状变化 %s → %s → 自动重置 state（计划重建）" % (
                    state.prev_input_shape, cur_shape))
            state.reset()
            state.prev_input_shape = cur_shape
            return (False, 0.0, 0.0, 0.0, "shape_reset")

        # 身份一致性锁：r2v 续写/图生视频直接全算
        if state.resolved_mode == "r2v":
            state.yang_steps.append(step)
            state.identity_locked_steps += 1
            return (False, 0.0, 0.0, 0.0, "r2v_identity_lock")

        if state.prev_output is None or state.prev_input is None:
            state.prev_input_shape = cur_shape
            return (False, 0.0, 0.0, 0.0, "no_ref")

        strategy = state.plan_strategy if state.plan_ready else "l1"
        prev_input = state.prev_input
        prev_prev = state.prev_prev_input
        cur_cpu = current_input.detach().cpu()
        delta = self._compute_signal_chunked(strategy, cur_cpu, prev_input, prev_prev)
        state.accumulate_strategy(strategy, delta)

        # 周流显存：仅监控，不改路径
        vram = read_vram_mb()
        if vram > state.peak_vram_mb:
            state.peak_vram_mb = vram
        if self.verbose and vram > self.vram_thresh_mb:
            print("[TaijiArmor v4s] VRAM 水位 %dMB 超阈值 %dMB —— 仅记录，不改变执行路径" % (
                vram, self.vram_thresh_mb))

        # warmup 探针：全算 + 收集 delta
        if step < self.warmup_steps:
            state.yang_steps.append(step)
            state.probe_deltas.append((step, delta))
            if self.verbose:
                print("[TaijiArmor v4s] PROBE Step %d/%d: delta=%.4f (strategy=%s)" % (
                    step, self.total_steps, delta, strategy))
            if step == self.warmup_steps - 1:
                self._finalize_plan(state)
            return (False, delta, 0.0, 0.0, "probe(%d<%d)" % (step, self.warmup_steps))

        # 末段保护
        effective_end = self.end_step if self.end_step >= 0 else self.total_steps + self.end_step
        if step >= effective_end:
            state.yang_steps.append(step)
            return (False, delta, 0.0, 0.0, "after_end(%d>=%d)" % (step, effective_end))

        # 护体真元：降级为纯监控（永不改路径）
        if self.fallback_mode:
            total_done = state.reuse_count + state.real_count
            fb_thresh = self._fallback_threshold()
            if total_done > 3 and state.reuse_count / total_done > fb_thresh:
                state.anomaly_count += 1
                if state.anomaly_count >= 2 and self.verbose:
                    print("[TaijiArmor v4s] 护体真元监控: 命中率 %d/%d > %.2f —— 仅记录，"
                          "不改变执行路径（轨迹确定性）" % (state.reuse_count, total_done, fb_thresh))
            else:
                state.anomaly_count = 0

        if not state.plan_ready:
            self._finalize_plan(state)
        thresh = state.plan_thresh

        # 连续缓存上限（确定性纪律）
        if state.consecutive_cache >= self.max_consecutive:
            state.consecutive_cache = 0
            state.yang_steps.append(step)
            return (False, delta, thresh, 0.0, "max_consec(%d)" % self.max_consecutive)

        # 静态计划决策
        if delta < thresh:
            state.yin_steps.append(step)
            state.consecutive_cache += 1
            if state.consecutive_cache > state.max_consecutive_hit:
                state.max_consecutive_hit = state.consecutive_cache
            return (True, delta, thresh, 1.0, "plan(%.4f<%.4f)" % (delta, thresh))

        state.yang_steps.append(step)
        state.consecutive_cache = 0
        return (False, delta, thresh, 0.0, "plan(%.4f>=%.4f)" % (delta, thresh))


def _is_standard_sampling_path(args) -> bool:
    return isinstance(args, dict) and "input" in args and "timestep" in args and "c" in args


def _resolve_mode(args) -> str:
    """auto 模式自动判定：检测跨注意力上下文里是否有参考/图像注入线索。

    线索规则（启发式，不依赖私有键名）：
    - c 中任意 3D 张量 [b, tokens, dim] 且 tokens >= 1024：token 规模远超文本，
      视为参考视频/图像特征注入 → r2v
    - c 中任意 4D 张量 [b, c, h, w] 且 h*w >= 1024：空间特征注入 → r2v
    - 否则视为纯文生视频 → t2v
    """
    c = args.get("c", {})
    if not isinstance(c, dict):
        return "t2v"
    for v in c.values():
        if isinstance(v, torch.Tensor) and v.dim() >= 3:
            if v.dim() == 3 and v.shape[1] >= 1024:
                return "r2v"
            if v.dim() == 4 and v.shape[-2] * v.shape[-1] >= 1024:
                return "r2v"
    return "t2v"


def make_armor_wrapper_v4s(
    state: TaijiArmorV4SState,
    total_steps: int,
    warmup_steps: int,
    yang_thresh: float,
    enable_ensemble: bool,
    vram_thresh_mb: int,
    fallback_mode: bool,
    end_step: int,
    verbose: bool,
    safety_margin: float,
    correction_strength: float,
    max_consecutive: int,
    chunk_size: int,
    mode: str,
) -> Callable[..., torch.Tensor]:
    scheduler = TaijiSchedulerV4S(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        yang_thresh=yang_thresh,
        enable_ensemble=enable_ensemble,
        vram_thresh_mb=vram_thresh_mb,
        fallback_mode=fallback_mode,
        end_step=end_step,
        verbose=verbose,
        safety_margin=safety_margin,
        correction_strength=correction_strength,
        max_consecutive=max_consecutive,
        chunk_size=chunk_size,
        mode=mode,
    )

    import time as _time
    _step_t0 = [_time.perf_counter()]

    def _parse_timestep(ts_raw):
        if ts_raw is None:
            return None
        try:
            if hasattr(ts_raw, "detach"):
                return float(ts_raw.detach().cpu().float().reshape(-1).max())
            return float(ts_raw)
        except Exception:
            return None

    def _maybe_report(state_inner, ts_mode_inner):
        if state_inner.step_idx >= max(0, total_steps - 1) and not state_inner.reported:
            state_inner.reported = True
            _report_v4s(state_inner, total_steps, warmup_steps, yang_thresh,
                        enable_ensemble, vram_thresh_mb, fallback_mode, end_step,
                        verbose, safety_margin, correction_strength,
                        max_consecutive, chunk_size, state_inner.resolved_mode or mode)

    def wrapper(apply_model, args) -> torch.Tensor:
        # 非标准采样路径旁路（保留 v3s1 正确性修复）
        if not _is_standard_sampling_path(args):
            if verbose:
                print("[TaijiArmor v4s] 非标准采样路径，旁路透传（不缓存）")
            x = args["input"]
            return apply_model(x, args.get("timestep"), **args.get("c", {}))

        x = args["input"]
        _t0 = _time.perf_counter()
        if _step_t0[0] is None:
            _step_t0[0] = _t0

        # auto 模式首步解析（一次性，后续复用 state.resolved_mode）
        if state.resolved_mode is None:
            if mode == "auto":
                state.resolved_mode = _resolve_mode(args)
            else:
                state.resolved_mode = mode
            if verbose:
                print("[TaijiArmor v4s] mode resolved: %s (%s)" % (
                    state.resolved_mode, "auto-detected" if mode == "auto" else "manual"))

        # timestep 真实步对齐（保留 v3s1 正确性修复）
        ts_val = _parse_timestep(args.get("timestep"))
        ts_mode = ts_val is not None
        if ts_mode:
            if state.last_ts is not None and abs(ts_val - state.last_ts) > 1e-3:
                state.step_idx += 1
            state.last_ts = ts_val

        # prev_output_shape 输出形状保护（保留）
        if state.prev_output_shape is not None and state.prev_output_shape[0] != x.shape[0]:
            if verbose:
                print("[TaijiArmor v4s] 输出形状保护：缓存输出 batch=%d ≠ 当前输入 batch=%d → reset 并强制计算" % (
                    state.prev_output_shape[0], x.shape[0]))
            state.reset(keep_timeline=True)

        reuse, delta, thresh, confidence, strategy = scheduler.should_reuse(state, x)
        _sched_dt = _time.perf_counter() - _t0
        _step_wall = _time.perf_counter() - _step_t0[0]

        if reuse:
            state.reuse_count += 1
            state.step_wall_times.append(_step_wall)
            state.step_wall_cumulative = (state.step_wall_cumulative or 0.0) + _step_wall

            if correction_strength > 0 and state.output_trend is not None:
                trend = state.output_trend
                corrected = state.prev_output + trend * correction_strength
                state.correction_count += 1
                if verbose:
                    print("[TaijiArmor v4s] Step %d/%d: CACHE+插值 wall=%.1fs (delta=%.4f plan_thresh=%.4f trend_norm=%.4f %s)" % (
                        state.step_idx - 1, total_steps, _step_wall, delta, thresh,
                        trend.abs().mean().item(), strategy))
                _maybe_report(state, ts_mode)
                return corrected.to(x.device)
            else:
                if verbose:
                    print("[TaijiArmor v4s] Step %d/%d: CACHE wall=%.1fs (delta=%.4f plan_thresh=%.4f %s)" % (
                        state.step_idx - 1, total_steps, _step_wall, delta, thresh, strategy))
                _maybe_report(state, ts_mode)
                return state.prev_output.to(x.device)

        # 计算步
        output = apply_model(x, args["timestep"], **args["c"])

        # 趋势插值更新（EMA of output changes）
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
        state.prev_output_shape = tuple(output.shape)
        state.prev_input_shape = tuple(x.shape)
        state.real_count += 1
        state.step_wall_times.append(_step_wall)
        state.step_wall_cumulative = (state.step_wall_cumulative or 0.0) + _step_wall
        _model_dt = _time.perf_counter() - _t0
        if verbose:
            print("[TaijiArmor v4s] Step %d/%d: COMPUTE wall=%.1fs (sched=%.0fms model=%.0fms) (%s delta=%.4f plan_thresh=%.4f)" % (
                state.step_idx, total_steps, _step_wall, _sched_dt*1000, _model_dt*1000,
                strategy, delta, thresh))
        if not ts_mode:
            state.step_idx += 1
        _step_t0[0] = _time.perf_counter()
        _maybe_report(state, ts_mode)
        return output

    return wrapper


def _report_v4s(state, total_steps, warmup_steps, yang_thresh, enable_ensemble,
                vram_thresh_mb, fallback_mode, end_step, verbose,
                safety_margin, correction_strength, max_consecutive, chunk_size, mode):
    tot = state.reuse_count + state.real_count
    rate = (state.reuse_count / tot * 100) if tot else 0.0
    wall_total = state.step_wall_cumulative
    wall_times_str = ""
    if state.step_wall_times:
        wall_times_str = " | wall_times=[" + ", ".join("%.1f" % t for t in state.step_wall_times) + "]"
    base = (f"[XuanShuTaijiArmor v4s] steps={total_steps} warmup={warmup_steps} "
            f"yang_thresh={yang_thresh} ensemble={enable_ensemble} "
            f"vram_th={vram_thresh_mb}MB fallback={fallback_mode}(monitor) end_step={end_step} "
            f"mode={mode} chunk_size={chunk_size} "
            f"plan_thresh={state.plan_thresh:.4f} plan_strategy={state.plan_strategy} "
            f"probe={[(d[0], round(d[1],4)) for d in state.probe_deltas]} "
            f"yang={len(state.yang_steps)} yin={len(state.yin_steps)} "
            f"reuse={state.reuse_count} real={state.real_count} hit_rate={rate:.1f}% "
            f"identity_locked={state.identity_locked_steps} peak_vram={state.peak_vram_mb}MB "
            f"| total_wall={wall_total:.1f}s\n"
            f"  v4s: safety={safety_margin} correction={correction_strength} "
            f"max_consec={max_consecutive} max_consec_hit={state.max_consecutive_hit} "
            f"corrections={state.correction_count}{wall_times_str}")
    if verbose:
        base += (f"\n  yang_steps={state.yang_steps}\n  yin_steps={state.yin_steps}\n"
                 f"  strategy={state.current_strategy}")
    print(base)


class MiniMaxH3TaijiArmorV4S:
    """玄枢 v4s（末梦秋花v4 顺风飘）：轨迹确定性版太极装甲。

    静态计划表 + 护体真元降级监控 + 养蛊择优锁定一次 + 身份一致性锁(r2v)，
    目标：消灭护体回退/动态调阈导致的人物不一致。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["auto", "t2v", "r2v"], {"default": "auto",
                                           "tooltip": "auto=首步自动判定：检测参考注入线索(图生/参考续写→r2v全算，纯文生→t2v缓存)；t2v=强制纯文生；r2v=强制参考续写全算"}),
                "total_steps": ("INT", {"default": 6, "min": 1, "max": 200,
                                          "tooltip": "推理总步数"}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 100,
                                           "tooltip": "探针步数：前 N 步全算并收集 delta，结束后锁定静态阈值"}),
                "yang_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01,
                                             "tooltip": "静态阈值下限：plan_thresh 不低于此值"}),
                "end_step": ("INT", {"default": -2, "min": -20, "max": 100,
                                      "tooltip": "末段保护：最后 N 步全算"}),
                "enable_ensemble": ("BOOLEAN", {"default": True,
                                                    "tooltip": "养蛊择优：探针结束后锁定一次策略，执行阶段不切换"}),
                "vram_thresh_mb": ("INT", {"default": 7500, "min": 1000, "max": 20000,
                                              "tooltip": "显存水位阈值(MB)：超限仅记录日志，不改变执行路径"}),
                "fallback_mode": ("BOOLEAN", {"default": True,
                                                 "tooltip": "护体真元：保留参数但降级为纯监控，永不改变执行路径"}),
                "verbose": ("BOOLEAN", {"default": True}),
                "safety_margin": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.1,
                                              "tooltip": "静态阈值 = 探针中位 delta × 此值"}),
                "correction_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                                    "tooltip": "缓存趋势插值强度：prev_output + trend×此值，保持轨迹连续"}),
                "max_consecutive": ("INT", {"default": 2, "min": 1, "max": 10,
                                              "tooltip": "连续缓存上限：默认2，任何时刻最多连跳2步（防长段漂移）"}),
                "chunk_size": ("INT", {"default": 2, "min": 1, "max": 8,
                                         "tooltip": "整批分块：batch>1 时按此块大小分块计算 delta 取最大（保守）"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "MiniMax H3 太极装甲 v4s（玄枢 v4s）"

    def apply(self, model, mode, total_steps, warmup_steps, yang_thresh, end_step,
              enable_ensemble, vram_thresh_mb, fallback_mode, verbose,
              safety_margin, correction_strength, max_consecutive, chunk_size):
        if warmup_steps >= total_steps:
            print(f"[TaijiArmor v4s] 警告：warmup({warmup_steps}) >= total({total_steps})，缓存完全失效")

        if mode == "r2v" and verbose:
            print("[TaijiArmor v4s] r2v 身份一致性锁：参考续写/图生视频模式，全程全算，不缓存")

        if mode == "auto" and verbose:
            print("[TaijiArmor v4s] auto 模式：首步采样时自动判定 t2v/r2v（检测参考注入线索）")

        if "xuanshu_teacache_guard" in model.model_options:
            print("[TaijiArmor v4s] 警告：检测到 Guard 已应用，v4s wrapper 将被覆盖")

        state = TaijiArmorV4SState()
        state.reset()
        wrapper = make_armor_wrapper_v4s(
            state=state,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            yang_thresh=yang_thresh,
            enable_ensemble=enable_ensemble,
            vram_thresh_mb=vram_thresh_mb,
            fallback_mode=fallback_mode,
            end_step=end_step,
            verbose=verbose,
            safety_margin=safety_margin,
            correction_strength=correction_strength,
            max_consecutive=max_consecutive,
            chunk_size=chunk_size,
            mode=mode,
        )
        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        patched.model_options["xuanshu_taiji_armor_v4s"] = {"state": state}
        patched.model_options.pop("xuanshu_teacache_guard", None)
        return (patched,)


NODE_CLASS_MAPPINGS_V4S = {"MiniMaxH3TaijiArmorV4S": MiniMaxH3TaijiArmorV4S}
NODE_DISPLAY_NAME_MAPPINGS_V4S = {"MiniMaxH3TaijiArmorV4S": "末梦秋花v4(顺风飘)"}
