"""末梦秋花 v5.2x（逆风飘·改进版）—— condition 层微抵抗参考图校正（改进实现）

与 v5.2flash 原版的差异仅三处，可用 mode 参数切换做 A/B：
  1. 度量：加权 + per-token 低分位，剔除恒定/零方差 tensor（修读数失真）
  2. 控制：比例-微分 + 软死区（去 bang-bang 抖动）
  3. 复位：reset_base 同时清零 step_idx（修双采/精修段第二段退化为 end_protect）

不改 latent、不跳步，红线 α<=max_alpha 保持不变。
mode="original" 时行为与原版逐位等价，便于同一节点内自证。
"""

import time
import torch
import torch.nn.functional as F
from typing import Optional, Any


# ============================================================
# 工具函数
# ============================================================
def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算两个 tensor 的余弦相似度（展平后，强制同设备）。"""
    a_f = a.detach().to("cpu", dtype=torch.float32).flatten()
    b_f = b.detach().to("cpu", dtype=torch.float32).flatten()
    if a_f.numel() < 2:
        return 1.0
    return torch.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def _deep_similarity(cur: Any, base: Any) -> float:
    """原版度量：递归计算嵌套结构的平均余弦相似度。"""
    sims = []

    def _walk(c, b):
        if isinstance(c, torch.Tensor) and isinstance(b, torch.Tensor):
            if c.shape == b.shape and c.numel() > 1:
                sims.append(_cosine_sim(c, b))
        elif isinstance(c, dict) and isinstance(b, dict):
            for k in c:
                if k in b:
                    _walk(c[k], b[k])
        elif isinstance(c, (list, tuple)) and isinstance(b, (list, tuple)):
            for i in range(min(len(c), len(b))):
                _walk(c[i], b[i])

    _walk(cur, base)
    if not sims:
        return 1.0
    return sum(sims) / len(sims)


# ---------- 改进度量 ----------
def _leaf_pairs(cur, base, out, path=()):
    if isinstance(cur, torch.Tensor) and isinstance(base, torch.Tensor):
        if cur.shape == base.shape and cur.numel() > 1:
            out.append((cur, base, path))
    elif isinstance(cur, dict) and isinstance(base, dict):
        for k in cur:
            if k in base:
                _leaf_pairs(cur[k], base[k], out, path + (str(k),))
    elif isinstance(cur, (list, tuple)) and isinstance(base, (list, tuple)):
        for i in range(min(len(cur), len(base))):
            _leaf_pairs(cur[i], base[i], out, path + (i,))
    return out


def _is_constant(t: torch.Tensor, eps: float = 1e-6) -> bool:
    """恒定 tensor 判别：fp32 下标准差趋 0（这类 tensor 的余弦恒为 1，必须剔除）。"""
    try:
        return float(t.detach().float().std()) < eps
    except Exception:
        return False


def _token_cos(cur: torch.Tensor, base: torch.Tensor, pct: float = 10.0):
    """per-token 余弦取低分位；低维 tensor 退化为整体余弦。返回 (sim, weight)"""
    a = cur.detach().to("cpu", dtype=torch.float32)
    b = base.detach().to("cpu", dtype=torch.float32)
    if a.ndim <= 1:
        v = a.flatten()
        w = b.flatten()
        if v.numel() < 2:
            return 1.0, 0
        return float(F.cosine_similarity(v.unsqueeze(0), w.unsqueeze(0)).item()), int(v.numel())
    dim = a.shape[-1]
    fa = a.reshape(-1, dim)
    fb = b.reshape(-1, dim)
    if fa.shape[0] < 10:
        c = F.cosine_similarity(fa, fb, dim=-1)
        return float(c.mean()), int(a.numel())
    c = F.cosine_similarity(fa, fb, dim=-1)
    return float(torch.quantile(c, pct / 100.0)), int(a.numel())


def similarity_weighted(cur: Any, base: Any, pct: float = 10.0) -> float:
    """改进度量：权重 = 元素数，恒定 tensor 剔除，per-token 低分位。"""
    leaves = _leaf_pairs(cur, base, [])
    num = 0.0
    den = 0.0
    for c, b, _ in leaves:
        if _is_constant(c) or _is_constant(b):
            continue
        s, w = _token_cos(c, b, pct)
        if w <= 0:
            continue
        num += s * w
        den += w
    if den <= 0:
        return 1.0
    return num / den


def _jitter(obj: Any, amp: float, gen: torch.Generator) -> Any:
    """调试用：对 condition 施加确定性随机漂移，模拟真实工况下的条件偏移。"""
    if torch.is_tensor(obj):
        t = obj.to(torch.float32)
        std = float(t.std()) if t.numel() > 1 else 0.0
        if std <= 0.0 or std != std:
            std = float(t.abs().mean()) or 1.0
        noise = torch.randn(t.shape, generator=gen, dtype=torch.float32) * (std * amp)
        return (t + noise.to(t.device)).to(obj.dtype)
    if isinstance(obj, dict):
        return type(obj)((k, _jitter(v, amp, gen)) for k, v in obj.items())
    if isinstance(obj, list):
        return [_jitter(v, amp, gen) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_jitter(v, amp, gen) for v in obj)
    return obj


def _mix_condition(cur: Any, base: Any, alpha: float) -> Any:
    """递归混合 condition：corrected = cur + alpha * (base - cur)。"""
    if abs(alpha) < 1e-6:
        return cur

    if isinstance(cur, torch.Tensor) and isinstance(base, torch.Tensor):
        if cur.shape == base.shape:
            return cur + alpha * (base.to(cur.device, cur.dtype) - cur)
        return cur
    elif isinstance(cur, dict) and isinstance(base, dict):
        result = {}
        for k in cur:
            if k in base:
                result[k] = _mix_condition(cur[k], base[k], alpha)
            else:
                result[k] = cur[k]
        return result
    elif isinstance(cur, (list, tuple)) and isinstance(base, (list, tuple)):
        result = []
        for i in range(len(cur)):
            if i < len(base):
                result.append(_mix_condition(cur[i], base[i], alpha))
            else:
                result.append(cur[i])
        return type(cur)(result)
    else:
        return cur


# ============================================================
# 状态
# ============================================================
class V52XState:
    def __init__(self):
        self.step_idx = 0
        self.base_condition: Optional[Any] = None
        self.base_input_shape = None
        self.correct_count = 0
        self.prev_e = None
        self.last_conc = None

    def reset_base(self, fix_step: bool = True):
        self.base_condition = None
        self.base_input_shape = None
        # 改进点 3：复位时步号与微分项一并清零，避免第二段直接落进 end_protect
        if fix_step:
            self.step_idx = 0
            self.prev_e = None
            self.last_conc = None


# ============================================================
# 调度器
# ============================================================
class V52XScheduler:
    def __init__(
        self,
        total_steps: int,
        warmup_steps: int = 3,
        end_protect_steps: int = 2,
        conc_high: float = 0.95,
        conc_low: float = 0.85,
        max_alpha: float = 0.3,
        ctrl_mode: str = "pd",
        kp: float = 3.0,
        kd: float = 0.8,
        deadband: float = 0.02,
    ):
        self.total_steps = int(total_steps)
        self.warmup_steps = max(0, int(warmup_steps))
        self.end_protect_steps = max(0, int(end_protect_steps))
        self.conc_high = float(conc_high)
        self.conc_low = float(conc_low)
        self.max_alpha = float(max_alpha)
        self.ctrl_mode = str(ctrl_mode)
        self.kp = float(kp)
        self.kd = float(kd)
        self.deadband = float(deadband)

    def _alpha_orig(self, concentration: float, step: int) -> tuple:
        if concentration >= self.conc_high:
            return (0.0, f"conc={concentration:.3f}>=high, no_correct")
        if concentration >= self.conc_low:
            raw_alpha = (self.conc_high - concentration) / (self.conc_high - self.conc_low) * self.max_alpha
        else:
            raw_alpha = self.max_alpha
        eff_end = self.total_steps - self.end_protect_steps
        if step < self.warmup_steps:
            raw_alpha *= 1.5
            phase = "warmup(x1.5)"
        elif step >= eff_end:
            raw_alpha *= 0.3
            phase = "end_protect(x0.3)"
        else:
            phase = "mid"
        raw_alpha = min(raw_alpha, 0.5)
        return (raw_alpha, f"conc={concentration:.3f} a={raw_alpha:.3f} {phase}")

    def _alpha_pd(self, concentration: float, step: int, state: "V52XState") -> tuple:
        # 改进点 2：比例-微分 + 软死区，连续输出替代硬阈值开关
        e = max(0.0, 1.0 - concentration - self.deadband)
        de = 0.0 if state.prev_e is None else (e - state.prev_e)
        state.prev_e = e
        raw_alpha = self.kp * e + self.kd * de

        eff_end = self.total_steps - self.end_protect_steps
        if step < self.warmup_steps:
            raw_alpha *= 1.2
            phase = "warmup(x1.2)"
        elif step >= eff_end:
            # 末尾：偏离仍在扩大才收力，否则照常保温
            raw_alpha *= 0.3 if de <= 0 else 0.8
            phase = "end_protect"
        else:
            phase = "mid"

        raw_alpha = max(0.0, min(raw_alpha, self.max_alpha))
        return (raw_alpha, f"conc={concentration:.3f} e={e:.3f} de={de:+.3f} a={raw_alpha:.3f} {phase}")

    def compute_alpha(self, concentration: float, step: int, state: "V52XState") -> tuple:
        if self.ctrl_mode == "pd":
            return self._alpha_pd(concentration, step, state)
        return self._alpha_orig(concentration, step)


# ============================================================
# Wrapper
# ============================================================
def make_v52x_wrapper(
    total_steps: int,
    warmup_steps: int,
    end_protect_steps: int,
    conc_high: float,
    conc_low: float,
    max_alpha: float,
    shape_protection: bool,
    report_stats: bool,
    metric_mode: str = "weighted",
    ctrl_mode: str = "pd",
    pct: float = 10.0,
    kp: float = 3.0,
    kd: float = 0.8,
    deadband: float = 0.02,
    fix_reset: bool = True,
    debug_conc_bias: float = 0.0,
    debug_drift: float = 0.0,
):
    state = V52XState()
    sched = V52XScheduler(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        end_protect_steps=end_protect_steps,
        conc_high=conc_high,
        conc_low=conc_low,
        max_alpha=max_alpha,
        ctrl_mode=ctrl_mode,
        kp=kp,
        kd=kd,
        deadband=deadband,
    )

    def wrapper(apply_model, args):
        x = args["input"]
        c = args.get("c", {})

        if shape_protection and state.base_input_shape is not None:
            if tuple(x.shape) != tuple(state.base_input_shape):
                if report_stats:
                    print(f"[V5.2x] 形状变化 → 重置基准")
                state.reset_base(fix_step=fix_reset)

        if state.base_condition is None:
            state.base_condition = _to_cpu_deep(c)
            state.base_input_shape = tuple(x.shape)
            if report_stats:
                print(f"[V5.2x] step 0: 基准 condition 已缓存, shape={x.shape}")

        # 调试注入：给当前 condition 施加每步不同的随机漂移，模拟真实条件偏移工况
        c_used = c
        if debug_drift > 0.0:
            _g = torch.Generator(device="cpu").manual_seed(20261017 + state.step_idx)
            c_used = _jitter(c, debug_drift, _g)
            if report_stats:
                print(f"[V5.2x] 漂移注入 amp={debug_drift:.2f} step={state.step_idx}")

        # 度量
        if metric_mode == "weighted":
            concentration = similarity_weighted(c_used, state.base_condition, pct)
        else:
            concentration = _deep_similarity(c_used, state.base_condition)

        conc_raw = concentration
        if debug_conc_bias > 0.0:
            # 调试注入：人为压低测量浓度，模拟条件漂移，用于强制激活校正链路
            concentration = max(0.0, concentration - debug_conc_bias)

        alpha, reason = sched.compute_alpha(concentration, state.step_idx, state)
        if debug_conc_bias > 0.0:
            reason = f"bias={debug_conc_bias:.2f} raw={conc_raw:.4f} " + reason

        if alpha > 1e-6:
            corrected_c = _mix_condition(c_used, state.base_condition, alpha)
            state.correct_count += 1
        else:
            corrected_c = c_used

        t0 = time.perf_counter()
        output = apply_model(x, args["timestep"], **corrected_c)

        if report_stats:
            dt = time.perf_counter() - t0
            print(f"[V5.2x] step {state.step_idx}/{total_steps} "
                  f"COMPUTE {dt:.1f}s {reason}")

        state.step_idx += 1
        return output

    return state, wrapper


def _to_cpu_deep(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", dtype=torch.float32)
    elif isinstance(obj, dict):
        return {k: _to_cpu_deep(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu_deep(v) for v in obj)
    else:
        return obj


# ============================================================
# ComfyUI 节点
# ============================================================
class MiniMaxH3TaijiArmorV52FlashX:
    """末梦秋花 v5.2x（逆风飘·改进）：度量失真修复 + PD 控制 + 复位修复。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 6, "min": 1, "max": 200}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 50}),
                "end_protect_steps": ("INT", {"default": 2, "min": 0, "max": 20}),
                "conc_high": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01}),
                "conc_low": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 0.99, "step": 0.01}),
                "max_alpha": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.5, "step": 0.05}),
                "shape_protection": ("BOOLEAN", {"default": True}),
                "report_stats": ("BOOLEAN", {"default": True}),
                "mode": (["improved", "original"], {"default": "improved",
                    "tooltip": "original=逐位复刻原版行为，用于 A/B 对照"}),
                "pct": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0,
                    "tooltip": "per-token 低分位百分位（越小越灵敏）"}),
                "kp": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "kd": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.1}),
                "deadband": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005}),
                "fix_reset": ("BOOLEAN", {"default": True,
                    "tooltip": "复位基准时同时清零步号"}),
                "debug_conc_bias": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "调试注入:人为压低测量浓度以强制激活校正,0=关闭"}),
                "debug_drift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "调试注入:每步给condition施加相对随机漂移,模拟真实条件偏移,0=关闭"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "末梦秋花v5.2x(逆风飘·改进)"

    def apply(self, model, total_steps, warmup_steps, end_protect_steps,
              conc_high, conc_low, max_alpha, shape_protection, report_stats,
              mode="improved", pct=10.0, kp=3.0, kd=0.8, deadband=0.02, fix_reset=True,
              debug_conc_bias=0.0, debug_drift=0.0):
        improved = (mode == "improved")
        state, wrapper = make_v52x_wrapper(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            end_protect_steps=end_protect_steps,
            conc_high=conc_high,
            conc_low=conc_low,
            max_alpha=max_alpha,
            shape_protection=shape_protection,
            report_stats=report_stats,
            metric_mode="weighted" if improved else "original",
            ctrl_mode="pd" if improved else "original",
            pct=pct,
            kp=kp,
            kd=kd,
            deadband=deadband,
            fix_reset=fix_reset if improved else False,
            debug_conc_bias=debug_conc_bias,
            debug_drift=debug_drift,
        )
        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        patched.model_options["xuanshu_v52x"] = {"state": state}
        for k in ["xuanshu_teacache_guard", "xuanshu_taiji_armor",
                  "xuanshu_taiji_armor_v4", "xuanshu_taiji_armor_v4_1",
                  "xuanshu_taiji_armor_v4_final", "xuanshu_taiji_armor_v4s",
                  "xuanshu_taiji_armor_v5", "xuanshu_taiji_armor_v5_1",
                  "xuanshu_v52flash"]:
            patched.model_options.pop(k, None)
        return (patched,)


NODE_CLASS_MAPPINGS_V52FLASHX = {"MiniMaxH3TaijiArmorV52FlashX": MiniMaxH3TaijiArmorV52FlashX}
NODE_DISPLAY_NAME_MAPPINGS_V52FLASHX = {"MiniMaxH3TaijiArmorV52FlashX": "末梦秋花v5.2x(逆风飘·改进)"}
