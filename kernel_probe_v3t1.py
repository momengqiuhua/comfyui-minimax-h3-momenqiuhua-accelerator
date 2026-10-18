# -*- coding: utf-8 -*-
"""玄枢 v3t1 路线A：注意力 kernel 自动寻优节点 v3t1（无损加速 + 模型指纹隔离）。

v3t1 = v3t + 决策缓存加模型身份指纹：
- 全局决策缓存 _DECISIONS 的 key 由 (seq, heads, dtype) 扩展为
  (seq, heads, dtype, model_fingerprint)，不同模型架构/配置的决策互不串用。
- 模型指纹从 model.model_type 与 model.model_config（类名 / model_type /
  architecture 属性）提取；全部取不到时回退 v3t 原 key（不含指纹）。
- 其余逻辑（缓存 miss 自动重测、失败排除、累计运行统计）与 v3t 一致。
- quality_mode：speed_priority（默认，全候选同权比耗时）/
  lossless_priority（pytorch 无损优先，仅异常才用有损后端），
  不改变缓存 key 结构，(shape, heads, dtype, fingerprint) 可跨任务复用。
"""
import time
import os
import torch
from comfy.ldm.modules.attention import get_attention_function

# Marvis 实测计时日志：探测阶段耗时明细落盘（不参与逻辑）
_PROBE_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v3t1_probe_timing.log")

def _log_probe_line(line):
    try:
        with open(_PROBE_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {line}\n")
    except Exception:
        pass

# (后端名, 是否数学无损)。pytorch 是唯一无损候选；sage 全系走 QK int8 量化，
# comfy_kitchen_int8 是 int8 注意力。本机 8GB/sm_120 实测：两个量化后端只快
# 5~6%（85.0s → 80s），人脸余弦相似度却从 1.0 掉到 0.968(sage)/0.955(int8)。
# 取舍交给 quality_mode：speed_priority 全候选同权纯比耗时（恢复 v2 行为）；
# lossless_priority 才按无损优先选（pytorch 优先，仅异常才用有损后端）。
CANDIDATES = [("pytorch", True), ("sage", False), ("comfy_kitchen_int8", False)]

# 大 token 分流阈值：seq 超过该值时量化后端(sage/int8)在显存压力下退化。实测
# seq=38289 时 sage 崩到 27796ms、int8 25438ms（pytorch 仅 1032ms），seq=50568
# 更退化到 44~50s，完整探测白等 128~335s 仍只能选 pytorch；而 seq=24663 处量化
# 后端仍快 2.7 倍（sage 158ms vs pytorch 424ms）须保留探测。阈值 40000 挡不住
# 38289 的崩溃，下调到 32000：低于崩溃点留 6000 余量，高于收益点留 7000 余量。
# （原值 40000；回退即改回。）
_LARGE_TOKEN_ROUTE_THRESHOLD = 32000

# 全局决策缓存: key=(seq, heads, dtype[, fingerprint]) -> (backend_name, fn, ms)
_DECISIONS = {}
_STATS = {}  # backend -> [calls, total_ms] 运行统计


def _extract_model_fingerprint(m):
    """从 model / model_config 提取模型身份指纹；取不到返回 None（回退原 key）。"""
    parts = []
    mt = getattr(m, "model_type", None)
    if mt:
        parts.append(str(mt))
    cfg = getattr(m, "model_config", None)
    if cfg is not None:
        try:
            cls_name = type(cfg).__name__
            if cls_name:
                parts.append(cls_name)
            cfg_mt = getattr(cfg, "model_type", None)
            if cfg_mt:
                parts.append(str(cfg_mt))
            arch = getattr(cfg, "architecture", None)
            if arch:
                parts.append(str(arch))
        except Exception:
            pass
    if not parts:
        return None
    return ":".join(parts)


def _bench_backend(fn, q, k, v, heads, kwargs, runs=3):
    """warm 1 + 计时 runs 次，取中位数（消单次抖动）"""
    out = fn(q, k, v, heads, **kwargs)  # warm，顺便拿到返回值
    times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(q, k, v, heads, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2], out


def _make_probe_wrapper(enabled, fingerprint, quality_mode="speed_priority", report_stats="disable"):
    state = {"decided": False}
    counters = {"calls": 0}

    def _maybe_report_stats():
        if report_stats != "enable":
            return
        print("[v3t1 KernelProbe] 运行统计（累计 1000 次注意力调用）：")
        for name, (calls, total) in _STATS.items():
            print(f"  {name}: {calls} 次, 共 {total:.1f}s, 均 {total/max(calls, 1)*1000:.0f} ms")

    def wrapper(q, k, v, heads, mask=None, attn_precision=None,
                skip_reshape=False, skip_output_reshape=False, **kwargs):
        key_base = (tuple(q.shape), heads, str(q.dtype))
        key = (key_base + (fingerprint,)) if fingerprint else key_base
        call_kwargs = {"mask": mask, "attn_precision": attn_precision,
                       "skip_reshape": skip_reshape,
                       "skip_output_reshape": skip_output_reshape}

        # 命中缓存：直接分发并累计统计（Bug2：统计改为在 wrapper 内按调用次数触发）
        if key in _DECISIONS:
            name, fn, _ms = _DECISIONS[key]
            t0 = time.perf_counter()
            out = fn(q, k, v, heads, **call_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            st = _STATS.setdefault(name, [0, 0.0])
            st[0] += 1
            st[1] += time.perf_counter() - t0
            counters["calls"] += 1
            if counters["calls"] % 1000 == 0:
                _maybe_report_stats()
            return out

        # Bug1 修复：判据由 state["decided"] 改为 key not in _DECISIONS ——
        # 缓存 miss 一律重新实测寻优并写入缓存；decided 仅控制首次寻优的打印文案。
        if not state["decided"]:
            state["decided"] = True
            print(f"[v3t1 KernelProbe] 首次调用，实测寻优 shape={q.shape} heads={heads} dtype={q.dtype} "
                  f"fingerprint={fingerprint} quality_mode={quality_mode}")
        else:
            print(f"[v3t1 KernelProbe] 新 shape（缓存 miss），重新实测寻优 shape={q.shape} heads={heads} "
                  f"dtype={q.dtype} fingerprint={fingerprint} quality_mode={quality_mode}")
        # ---- 大 token 分流（Marvis 实测优化）----
        # seq 超阈值时跳过量化后端探测，直接锁定 pytorch；避免 sage/int8
        # 在大序列上 bench 退化（单次 44~50s）造成的 5 分钟+ 无效等待。
        seq_len = int(q.shape[-2])
        if seq_len > _LARGE_TOKEN_ROUTE_THRESHOLD:
            print(f"[v3t1 KernelProbe] 大 token 分流: seq={seq_len} > {_LARGE_TOKEN_ROUTE_THRESHOLD}, 跳过量化后端探测")
            _log_probe_line(f"ROUTE shape={tuple(q.shape)} heads={heads} dtype={q.dtype} seq={seq_len}>{_LARGE_TOKEN_ROUTE_THRESHOLD} 大token分流 跳过量化后端")
            _b0 = time.perf_counter()
            fn = None
            try:
                fn = get_attention_function("pytorch", None)
                if fn is None:
                    raise RuntimeError("pytorch 未注册")
                ms, out = _bench_backend(fn, q, k, v, heads, call_kwargs)
                _elapsed = time.perf_counter() - _b0
            except Exception as e:
                print(f"[v3t1 KernelProbe]   pytorch 不可用 ({str(e)[:60]})，回退正常探测")
            if fn is not None:
                _DECISIONS[key] = ("pytorch", fn, ms)
                _log_probe_line(f"ROUTE_OK shape={tuple(q.shape)} heads={heads} dtype={q.dtype} pick=pytorch bench中位={ms:.1f}ms 含warm总耗时={_elapsed*1000:.0f}ms")
                print(f"[v3t1 KernelProbe] 决策: pytorch ({ms:.1f} ms, 无损) — 已锁定，后续走缓存")
                return out
        _probe_t0 = time.perf_counter()
        best = None
        for name, lossless in CANDIDATES:
            try:
                fn = get_attention_function(name, None)
                if fn is None:
                    print(f"[v3t1 KernelProbe]   {name}: 未注册，跳过")
                    _log_probe_line(f"PROBE shape={tuple(q.shape)} heads={heads} {name}: 未注册跳过")
                    continue
                _b0 = time.perf_counter()
                ms, out = _bench_backend(fn, q, k, v, heads, call_kwargs)
                _elapsed = time.perf_counter() - _b0
                print(f"[v3t1 KernelProbe]   {name}: {ms:.1f} ms 可用 "
                      f"({'无损' if lossless else '有损量化'}) (3次中位)")
                _log_probe_line(f"PROBE shape={tuple(q.shape)} heads={heads} dtype={q.dtype} {name}: bench中位={ms:.1f}ms 含warm总耗时={_elapsed*1000:.0f}ms {'无损' if lossless else '有损量化'}")
                cand = (name, fn, ms, out, lossless)
                if quality_mode == "lossless_priority":
                    # 无损优先：pytorch 无损优先，仅当 pytorch 抛异常才用有损后端；同级比速度
                    if best is None:
                        best = cand
                    elif lossless and not best[4]:
                        best = cand
                    elif lossless == best[4] and ms < best[2]:
                        best = cand
                else:
                    # speed_priority：候选全部同权，纯比中位耗时（恢复 v2 行为）
                    if best is None or ms < best[2]:
                        best = cand
            except Exception as e:
                print(f"[v3t1 KernelProbe]   {name}: 不可用 ({str(e)[:60]})")
                _log_probe_line(f"PROBE shape={tuple(q.shape)} heads={heads} {name}: 不可用 {str(e)[:60]}")
        if best is None:
            print("[v3t1 KernelProbe] 全部候选失败，回退 pytorch（无损）")
            fn = get_attention_function("pytorch")
            best = ("pytorch", fn, float("nan"), fn(q, k, v, heads, **call_kwargs), True)
        _probe_total = time.perf_counter() - _probe_t0
        _DECISIONS[key] = (best[0], best[1], best[2])
        print(f"[v3t1 KernelProbe] 决策: {best[0]} ({best[2]:.1f} ms, "
              f"{'无损' if best[4] else '有损量化'}) — 已锁定，后续走缓存")
        _log_probe_line(f"DECIDE shape={tuple(q.shape)} heads={heads} dtype={q.dtype} pick={best[0]} 探测总耗时={_probe_total*1000:.0f}ms")
        return best[3]

    return wrapper


class XuanShuH3KernelProbe1:
    """v3t1 路线A：首步实测自动选择注意力后端 + 模型指纹决策缓存隔离。

    quality_mode（tooltip）：
    - speed_priority（默认）：候选全部同权，纯比中位耗时，选最快后端。
      sage / comfy_kitchen_int8 含 int8 量化（实测人脸余弦约 0.968），
      对漫剧有 R2V 参考锁一致性的场景不可感知；慢后端约 2.7 倍（960 档 264ms vs 99ms）。
    - lossless_priority：pytorch 无损优先，仅当 pytorch 抛异常才用有损后端；
      慢但数学无损，适合单人物特写等身份要求极高的场景。

    决策缓存 key=(shape, heads, dtype, fingerprint)，不随 quality_mode 变化，可跨任务复用。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["auto_probe", "passthrough"],),
                "report_stats": (["disable", "enable"],),
                "quality_mode": (["speed_priority", "lossless_priority"], {"default": "speed_priority"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "XuanShu/v3t"

    def patch(self, model, mode, report_stats, quality_mode):
        if mode == "passthrough":
            return (model,)
        fingerprint = _extract_model_fingerprint(model)
        if fingerprint is None:
            print("[v3t1 KernelProbe] 模型指纹提取失败，回退 v3t 原 key（按 shape+dtype 缓存）")
        m = model.clone()
        m.set_model_optimized_attention(
            _make_probe_wrapper(True, fingerprint, quality_mode, report_stats))
        return (m,)


NODE_CLASS_MAPPINGS_PROBE1 = {"XuanShuH3KernelProbe1": XuanShuH3KernelProbe1}
NODE_DISPLAY_NAME_MAPPINGS_PROBE1 = {"XuanShuH3KernelProbe1": "末梦秋花v3t1(等风来)"}
