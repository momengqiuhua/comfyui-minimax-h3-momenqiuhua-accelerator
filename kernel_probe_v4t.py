# -*- coding: utf-8 -*-
"""末梦秋花 v4t（等风来）：注意力 kernel 寻优 + 磁盘缓存 + 显存感知路由。

v4t = v3t1 + 两项进化：
1. 磁盘缓存：(shape, heads, dtype, fingerprint) → backend_name 存 JSON，
   下次启动直接加载，跳过探针（省每次启动首次出图前的几秒等待）
2. 显存感知路由：探针时记录每个后端的显存峰值，运行时检查当前剩余显存，
   不够就降级选下一个（8G 卡刚需，防止长序列跑到一半 OOM）

其余逻辑与 v3t1 一致：ROUTE 大 token 分流、模型指纹隔离、quality_mode。
"""
import time
import os
import json
import torch
from comfy.ldm.modules.attention import get_attention_function

# ---- 磁盘缓存路径 ----
_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4t_kernel_cache.json")

# ---- 探针日志（跟 v3t1 兼容）----
_PROBE_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4t_probe_timing.log")


def _log_probe_line(line):
    try:
        with open(_PROBE_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {line}\n")
    except Exception:
        pass


# ---- 后端候选 ----
CANDIDATES = [("pytorch", True), ("sage", False), ("comfy_kitchen_int8", False)]

# 大 token 分流阈值（跟 v3t1 一致）
_LARGE_TOKEN_ROUTE_THRESHOLD = 32000


# ============================================================
# 磁盘缓存读写
# ============================================================
def _load_disk_cache():
    """从磁盘加载缓存，返回 dict。文件不存在或损坏返回空 dict。"""
    try:
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


def _save_disk_cache(cache):
    """写磁盘缓存，静默失败（不影响运行）。"""
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ============================================================
# 模型指纹
# ============================================================
def _extract_model_fingerprint(m):
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


# ============================================================
# 探针：测速 + 测显存
# ============================================================
def _bench_backend(fn, q, k, v, heads, kwargs, runs=3):
    """warm 1 + 计时 runs 次，返回 (中位ms, 显存峰值MB, out)。"""
    # 重置显存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    out = fn(q, k, v, heads, **kwargs)  # warm

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
    median_ms = times[len(times) // 2]

    # 显存峰值（MB）
    peak_mb = 0
    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_mb = peak_bytes / (1024 * 1024)

    return median_ms, peak_mb, out


# ============================================================
# 显存感知：当前剩余显存够不够跑某后端
# ============================================================
def _get_free_vram_mb():
    """返回当前空闲显存（MB），不可用返回 None。"""
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        return free_bytes / (1024 * 1024)
    except Exception:
        return None


# ============================================================
# 核心 wrapper
# ============================================================
def _make_probe_wrapper(enabled, fingerprint, quality_mode="speed_priority",
                        report_stats="disable", vram_aware=True, vram_margin_mb=200):
    # 内存缓存：key → (backend_name, fn, ms, peak_mb)
    _DECISIONS = {}
    _STATS = {}

    # 启动时加载磁盘缓存
    disk_cache = _load_disk_cache()
    _disk_hits = [0]  # 计数器

    counters = {"calls": 0}
    _downgraded = [False]  # 记录是否已降级过，避免重复打印

    def _maybe_report_stats():
        if report_stats != "enable":
            return
        print("[v4t KernelProbe] 运行统计（累计 1000 次注意力调用）：")
        for name, (calls, total) in _STATS.items():
            print(f"  {name}: {calls} 次, 共 {total:.1f}s, 均 {total/max(calls,1)*1000:.0f} ms")
        print(f"  磁盘缓存命中: {_disk_hits[0]} 次")

    def wrapper(q, k, v, heads, mask=None, attn_precision=None,
                skip_reshape=False, skip_output_reshape=False, **kwargs):
        key_base = (tuple(q.shape), heads, str(q.dtype))
        key = (key_base + (fingerprint,)) if fingerprint else key_base
        key_str = str(key)  # JSON 序列化用

        call_kwargs = {"mask": mask, "attn_precision": attn_precision,
                       "skip_reshape": skip_reshape,
                       "skip_output_reshape": skip_output_reshape}

        # ---- ① 内存缓存命中 ----
        if key in _DECISIONS:
            return _dispatch(key, q, k, v, heads, call_kwargs, vram_aware, vram_margin_mb)

        # ---- ② 磁盘缓存命中：直接获取 fn，跳过探针 ----
        if key_str in disk_cache:
            cached_name = disk_cache[key_str]
            try:
                fn = get_attention_function(cached_name, None)
                if fn is not None:
                    _DECISIONS[key] = (cached_name, fn, 0.0, 0.0)  # ms/peak 未知，不记录
                    _disk_hits[0] += 1
                    print(f"[v4t KernelProbe] 磁盘缓存命中 → {cached_name}（跳过探针）")
                    _log_probe_line(f"CACHE_HIT key={key_str} backend={cached_name}")
                    return _dispatch(key, q, k, v, heads, call_kwargs, vram_aware, vram_margin_mb)
            except Exception:
                pass  # 获取失败，走正常探针

        # ---- ③ 正常探针 ----
        return _probe_and_decide(key, key_str, q, k, v, heads, call_kwargs,
                                 fingerprint, quality_mode, vram_aware, vram_margin_mb,
                                 disk_cache)

    def _dispatch(key, q, k, v, heads, call_kwargs, vram_aware, vram_margin_mb):
        """根据 key 分发到最优后端，显存不够自动降级（只打印一次）。"""
        name, fn, ms, peak_mb = _DECISIONS[key]

        # 显存感知检查：当前剩余显存够不够？
        if vram_aware and peak_mb > 0 and not _downgraded[0]:
            free_mb = _get_free_vram_mb()
            if free_mb is not None and free_mb < (peak_mb + vram_margin_mb):
                # 当前后端显存不够，降级到 pytorch（最省显存）
                pytorch_fn = get_attention_function("pytorch", None)
                if pytorch_fn is not None and name != "pytorch":
                    print(f"[v4t KernelProbe] 显存不足(剩{free_mb:.0f}MB<需{peak_mb:.0f}MB)，"
                          f"{name} → 降级 pytorch（后续不再重复提示）")
                    name = "pytorch"
                    fn = pytorch_fn
                    _downgraded[0] = True  # 记住已降级，后续不重复打印

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

    def _probe_and_decide(key, key_str, q, k, v, heads, call_kwargs,
                          fingerprint, quality_mode, vram_aware, vram_margin_mb,
                          disk_cache):
        print(f"[v4t KernelProbe] 首次调用，实测寻优 shape={q.shape} heads={heads} "
              f"dtype={q.dtype} fingerprint={fingerprint} quality_mode={quality_mode}")

        # 大 token 分流
        seq_len = int(q.shape[-2])
        if seq_len > _LARGE_TOKEN_ROUTE_THRESHOLD:
            print(f"[v4t KernelProbe] 大 token 分流: seq={seq_len} > {_LARGE_TOKEN_ROUTE_THRESHOLD}, 跳过量化后端")
            fn = get_attention_function("pytorch", None)
            ms, peak_mb, out = _bench_backend(fn, q, k, v, heads, call_kwargs)
            _DECISIONS[key] = ("pytorch", fn, ms, peak_mb)
            disk_cache[key_str] = "pytorch"
            _save_disk_cache(disk_cache)
            print(f"[v4t KernelProbe] 决策: pytorch ({ms:.1f} ms, 显存{peak_mb:.0f}MB) — 已锁定")
            _log_probe_line(f"ROUTE key={key_str} pick=pytorch bench={ms:.1f}ms peak={peak_mb:.0f}MB")
            return out

        # 正常探针所有后端
        _probe_t0 = time.perf_counter()
        candidates = []  # (name, fn, ms, peak_mb, out, lossless)

        for name, lossless in CANDIDATES:
            try:
                fn = get_attention_function(name, None)
                if fn is None:
                    print(f"[v4t KernelProbe]   {name}: 未注册，跳过")
                    continue
                ms, peak_mb, out = _bench_backend(fn, q, k, v, heads, call_kwargs)
                print(f"[v4t KernelProbe]   {name}: {ms:.1f} ms, 显存峰值{peak_mb:.0f}MB "
                      f"({'无损' if lossless else '有损量化'})")
                _log_probe_line(f"PROBE key={key_str} {name}: bench={ms:.1f}ms peak={peak_mb:.0f}MB")
                candidates.append((name, fn, ms, peak_mb, out, lossless))
            except Exception as e:
                print(f"[v4t KernelProbe]   {name}: 不可用 ({str(e)[:60]})")
                _log_probe_line(f"PROBE key={key_str} {name}: 不可用 {str(e)[:60]}")

        if not candidates:
            print("[v4t KernelProbe] 全部候选失败，回退 pytorch")
            fn = get_attention_function("pytorch")
            out = fn(q, k, v, heads, **call_kwargs)
            _DECISIONS[key] = ("pytorch", fn, 0.0, 0.0)
            return out

        # 选最优：speed_priority 纯比速度，lossless_priority 无损优先
        if quality_mode == "lossless_priority":
            best = candidates[0]
            for c in candidates[1:]:
                if c[5] and not best[5]:
                    best = c
                elif c[5] == best[5] and c[2] < best[2]:
                    best = c
        else:
            best = min(candidates, key=lambda x: x[2])

        _probe_total = time.perf_counter() - _probe_t0
        _DECISIONS[key] = (best[0], best[1], best[2], best[3])

        # 写入磁盘缓存
        disk_cache[key_str] = best[0]
        _save_disk_cache(disk_cache)

        print(f"[v4t KernelProbe] 决策: {best[0]} ({best[2]:.1f} ms, "
              f"显存{best[3]:.0f}MB, {'无损' if best[5] else '有损量化'}) "
              f"— 已锁定，探测耗时 {_probe_total*1000:.0f}ms")
        _log_probe_line(f"DECIDE key={key_str} pick={best[0]} bench={best[2]:.1f}ms "
                        f"peak={best[3]:.0f}MB total={_probe_total*1000:.0f}ms")
        return best[4]

    return wrapper


# ============================================================
# ComfyUI 节点
# ============================================================
class XuanShuH3KernelProbeV4T:
    """末梦秋花 v4t（等风来）：kernel 寻优 + 磁盘缓存 + 显存感知路由。

    相比 v3t1 进化：
    - 磁盘缓存：跨会话记住最优后端，第二次跑跳过探针
    - 显存感知：长序列显存紧张时自动降级，防止 OOM

    quality_mode：
    - speed_priority（默认）：纯比速度，选最快后端
    - lossless_priority：pytorch 无损优先
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["auto_probe", "passthrough"],),
                "report_stats": (["disable", "enable"],),
                "quality_mode": (["speed_priority", "lossless_priority"], {"default": "speed_priority"}),
                "vram_aware": ("BOOLEAN", {"default": True,
                    "tooltip": "显存感知：剩余显存不够时自动降级到最省显存的后端"}),
                "vram_margin_mb": ("INT", {"default": 200, "min": 50, "max": 2000,
                    "tooltip": "显存安全余量(MB)：当前剩余显存 - 后端峰值显存 < 此值时降级"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "XuanShu/v3t"
    TITLE = "末梦秋花v4t(等风来)"

    def patch(self, model, mode, report_stats, quality_mode, vram_aware, vram_margin_mb):
        if mode == "passthrough":
            return (model,)
        fingerprint = _extract_model_fingerprint(model)
        if fingerprint is None:
            print("[v4t KernelProbe] 模型指纹提取失败，回退原 key")
        m = model.clone()
        m.set_model_optimized_attention(
            _make_probe_wrapper(True, fingerprint, quality_mode, report_stats,
                                vram_aware, vram_margin_mb))
        return (m,)


NODE_CLASS_MAPPINGS_V4T = {"XuanShuH3KernelProbeV4T": XuanShuH3KernelProbeV4T}
NODE_DISPLAY_NAME_MAPPINGS_V4T = {"XuanShuH3KernelProbeV4T": "末梦秋花v4t(等风来)"}
