# -*- coding: utf-8 -*-
"""玄枢 v3t 路线A：注意力 kernel 自动寻优节点（无损加速）。

原理：模型首个真实注意力调用到来时，用当时的真实 q/k/v 对各候选后端
实测计时（一次调用即百 ms~秒级，足够分辨），失败后端自动排除，
选定最快者并缓存决策（按 shape+dtype 为 key，shape 变化自动重测）。
纯调度层加速，不改数学、不损画质。
"""
import time
import torch
from comfy.ldm.modules.attention import get_attention_function

CANDIDATES = ["sage", "pytorch", "comfy_kitchen_int8"]

# 全局决策缓存: key=(seq, heads, dtype) -> (backend_name, fn, ms)
_DECISIONS = {}
_STATS = {}  # backend -> [calls, total_ms] 运行统计


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


def _make_probe_wrapper(enabled):
    state = {"decided": False}

    def wrapper(q, k, v, heads, mask=None, attn_precision=None,
                skip_reshape=False, skip_output_reshape=False, **kwargs):
        key = (tuple(q.shape), heads, str(q.dtype))
        kwargs.pop("mask", None) if mask is None else None
        call_kwargs = {"mask": mask, "attn_precision": attn_precision,
                       "skip_reshape": skip_reshape,
                       "skip_output_reshape": skip_output_reshape}

        if key in _DECISIONS:
            name, fn, _ms = _DECISIONS[key]
            t0 = time.perf_counter()
            out = fn(q, k, v, heads, **call_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            st = _STATS.setdefault(name, [0, 0.0])
            st[0] += 1
            st[1] += time.perf_counter() - t0
            return out

        if not state["decided"]:
            state["decided"] = True
            print(f"[v3t KernelProbe] 首次调用，实测寻优 shape={q.shape} heads={heads} dtype={q.dtype}")
            best = None
            for name in CANDIDATES:
                try:
                    fn = get_attention_function(name, None)
                    if fn is None:
                        print(f"[v3t KernelProbe]   {name}: 未注册，跳过")
                        continue
                    ms, out = _bench_backend(fn, q, k, v, heads, call_kwargs)
                    print(f"[v3t KernelProbe]   {name}: {ms:.1f} ms 可用 (3次中位)")
                    if best is None or ms < best[2]:
                        best = (name, fn, ms, out)
                except Exception as e:
                    print(f"[v3t KernelProbe]   {name}: 不可用 ({str(e)[:60]})")
            if best is None:
                print("[v3t KernelProbe] 全部候选失败，回退 sage")
                best = ("sage", get_attention_function("sage"),
                        float("nan"), wrapper(q, k, v, heads, **call_kwargs))
            _DECISIONS[key] = (best[0], best[1], best[2])
            print(f"[v3t KernelProbe] 决策: {best[0]} ({best[2]:.1f} ms) — 已锁定，后续走缓存")
            return best[3]

        # 未决策且无缓存（理论不可达）：回退 sage
        return get_attention_function("sage")(q, k, v, heads, **call_kwargs)

    return wrapper


class XuanShuH3KernelProbe:
    """v3t 路线A：首步实测自动选择注意力后端（无损）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["auto_probe", "passthrough"],),
                "report_stats": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "XuanShu/v3t"

    def patch(self, model, mode, report_stats):
        if mode == "passthrough":
            return (model,)
        m = model.clone()
        m.set_model_optimized_attention(_make_probe_wrapper(True))
        if report_stats == "enable" and _STATS:
            print("[v3t KernelProbe] 运行统计：")
            for name, (calls, total) in _STATS.items():
                print(f"  {name}: {calls} 次, 共 {total:.1f}s, 均 {total/max(calls,1)*1000:.0f} ms")
        return (m,)


NODE_CLASS_MAPPINGS_PROBE = {"XuanShuH3KernelProbe": XuanShuH3KernelProbe}
NODE_DISPLAY_NAME_MAPPINGS_PROBE = {"XuanShuH3KernelProbe": "玄枢 v3t Kernel 寻优器"}
