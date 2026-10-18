"""末梦秋花（逆风飘）v5x —— v5.2flash 纯修复版。

来源：taiji_armor_v5_2_flash.py（v5.2flash）逐行派生，仅修两处缺陷，
度量与调度逻辑与原版完全一致（不做 v5.2x 的加权度量 / PD 控制改造）。

修复项：
1. α 越限：原版末尾钳制写死 min(raw_alpha, 0.5)，warmup 段 ×1.5 后实际可达 0.45，
   越过用户设定的 max_alpha 红线（实测 0.309）。改为 min(raw_alpha, self.max_alpha)。
2. 分段步号越界：原版 reset_base() 不清 step_idx，双采第二段采样时步号从 6 续到 11，
   全部满足 step >= eff_end，整段被 ×0.3 收力（日志可见 step 6/6 → 11/6）。
   reset_base() 现在清零 step_idx；另在步号达到 total_steps 时自动分段复位。

参数与原版完全一致，可直接替换节点，无需改工作流连线。
"""

import time
import torch
from typing import Optional, Any


# ============================================================
# 工具函数（与原版一致）
# ============================================================
def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算两个 tensor 的余弦相似度（展平后，强制同设备）。"""
    a_f = a.detach().to("cpu", dtype=torch.float32).flatten()
    b_f = b.detach().to("cpu", dtype=torch.float32).flatten()
    if a_f.numel() < 2:
        return 1.0
    return torch.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def _deep_similarity(cur: Any, base: Any) -> float:
    """递归计算嵌套结构（dict/list/tuple/tensor）的平均余弦相似度。"""
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
# 状态（修复 2：reset_base 清零步号）
# ============================================================
class V52FlashYState:
    def __init__(self):
        self.step_idx = 0
        self.base_condition: Optional[Any] = None
        self.base_input_shape = None
        self.segment_idx = 0          # 已完成的采样段数，仅用于日志
        self.cache_count = 0
        self.correct_count = 0

    def reset_base(self):
        self.base_condition = None
        self.base_input_shape = None
        self.step_idx = 0             # ← 修复 2：分段复位必须归零
        self.segment_idx += 1


# ============================================================
# 调度器（修复 1：α 钳制到用户红线）
# ============================================================
class V52FlashYScheduler:
    def __init__(
        self,
        total_steps: int,
        warmup_steps: int = 3,
        end_protect_steps: int = 2,
        conc_high: float = 0.95,
        conc_low: float = 0.85,
        max_alpha: float = 0.3,
        shape_protection: bool = True,
        report_stats: bool = False,
    ):
        self.total_steps = int(total_steps)
        self.warmup_steps = max(0, int(warmup_steps))
        self.end_protect_steps = max(0, int(end_protect_steps))
        self.conc_high = float(conc_high)
        self.conc_low = float(conc_low)
        self.max_alpha = float(max_alpha)
        self.shape_protection = bool(shape_protection)
        self.report_stats = bool(report_stats)

    def compute_alpha(self, concentration: float, step: int) -> tuple:
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

        # ← 修复 1：红线改由 max_alpha 决定，不再写死 0.5
        raw_alpha = min(raw_alpha, self.max_alpha)

        return (raw_alpha, f"conc={concentration:.3f} a={raw_alpha:.3f} {phase}")


# ============================================================
# Wrapper
# ============================================================
def make_v5x_wrapper(
    total_steps: int,
    warmup_steps: int,
    end_protect_steps: int,
    conc_high: float,
    conc_low: float,
    max_alpha: float,
    shape_protection: bool,
    report_stats: bool,
):
    state = V52FlashYState()
    sched = V52FlashYScheduler(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        end_protect_steps=end_protect_steps,
        conc_high=conc_high,
        conc_low=conc_low,
        max_alpha=max_alpha,
        shape_protection=shape_protection,
        report_stats=report_stats,
    )

    def wrapper(apply_model, args):
        x = args["input"]
        c = args.get("c", {})

        # 修复 2 补充：步号跑满总步数 → 视为新采样段，自动复位
        if state.step_idx >= sched.total_steps:
            if report_stats:
                print(f"[V5x] step_idx={state.step_idx} >= total={total_steps} → 自动分段复位")
            state.reset_base()

        # 形状保护：input 形状变了，重置基准（reset_base 已含步号归零）
        if sched.shape_protection and state.base_input_shape is not None:
            if tuple(x.shape) != tuple(state.base_input_shape):
                if report_stats:
                    print("[V5x] 形状变化 → 重置基准")
                state.reset_base()

        if state.base_condition is None:
            state.base_condition = _to_cpu_deep(c)
            state.base_input_shape = tuple(x.shape)
            if report_stats:
                print(f"[V5x] seg {state.segment_idx} step 0: 基准 condition 已缓存, shape={x.shape}")

        concentration = _deep_similarity(c, state.base_condition)
        alpha, reason = sched.compute_alpha(concentration, state.step_idx)

        if alpha > 1e-6:
            corrected_c = _mix_condition(c, state.base_condition, alpha)
            state.correct_count += 1
        else:
            corrected_c = c

        t0 = time.perf_counter()
        output = apply_model(x, args["timestep"], **corrected_c)

        if report_stats:
            dt = time.perf_counter() - t0
            print(f"[V5x] seg {state.segment_idx} step {state.step_idx}/{total_steps} "
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
class MiniMaxH3TaijiArmorV52FlashY:
    """末梦秋花（逆风飘）v5x：v5.2flash 的纯修复版。

    度量（condition 余弦浓度）与调度（warmup / mid / end_protect）逐行沿用
    v5.2flash，只修 α 越限与分段步号越界两处缺陷。参数与原版一致。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 6, "min": 1, "max": 200,
                    "tooltip": "单段采样总步数（非整条链路总步数）"}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 50,
                    "tooltip": "前 N 步校正力度放大 x1.5，定型参考图"}),
                "end_protect_steps": ("INT", {"default": 2, "min": 0, "max": 20,
                    "tooltip": "最后 N 步校正力度缩小 x0.3，别动成片"}),
                "conc_high": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01,
                    "tooltip": "浓度高于此值不校正（没偏就不动）"}),
                "conc_low": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 0.99, "step": 0.01,
                    "tooltip": "浓度低于此值硬拉（明显偏了）"}),
                "max_alpha": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "校正力度硬上限，warmup 放大后也不得越过"}),
                "shape_protection": ("BOOLEAN", {"default": True,
                    "tooltip": "形状变化自动分段复位"}),
                "report_stats": ("BOOLEAN", {"default": False,
                    "tooltip": "打印每步浓度、校正力度与段号"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "末梦秋花（逆风飘）v5x"

    def apply(self, model, total_steps, warmup_steps, end_protect_steps,
              conc_high, conc_low, max_alpha, shape_protection, report_stats):
        state, wrapper = make_v5x_wrapper(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            end_protect_steps=end_protect_steps,
            conc_high=conc_high,
            conc_low=conc_low,
            max_alpha=max_alpha,
            shape_protection=shape_protection,
            report_stats=report_stats,
        )
        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        patched.model_options["xuanshu_v5x"] = {"state": state}
        # 清掉其他护甲标记，避免与旧节点混用
        for k in ["xuanshu_teacache_guard", "xuanshu_taiji_armor",
                  "xuanshu_taiji_armor_v4", "xuanshu_taiji_armor_v4_1",
                  "xuanshu_taiji_armor_v4_final", "xuanshu_taiji_armor_v4s",
                  "xuanshu_taiji_armor_v5", "xuanshu_taiji_armor_v5_1",
                  "xuanshu_v52flash", "xuanshu_v52x"]:
            patched.model_options.pop(k, None)
        return (patched,)


NODE_CLASS_MAPPINGS_V52FLASHY = {"MiniMaxH3TaijiArmorV52FlashY": MiniMaxH3TaijiArmorV52FlashY}
NODE_DISPLAY_NAME_MAPPINGS_V52FLASHY = {"MiniMaxH3TaijiArmorV52FlashY": "末梦秋花（逆风飘）v5x"}
