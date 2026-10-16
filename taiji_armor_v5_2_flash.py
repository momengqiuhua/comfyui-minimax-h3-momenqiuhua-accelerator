"""末梦秋花 v5.2flash（认人）—— condition 层微抵抗参考图校正。

核心思路：
- 不改 latent（v4 红线），只调 condition
- 不跳步，每次全算
- 反应式自适应：每步检测参考图特征浓度，偏了就拉，不偏不动
- 不记状态，每步重测，弹簧式微抵抗

从 v3s1 搬来：timestep 对齐、形状保护、整批分块思想
扔掉：缓存跳步、max_consecutive、IEC、五策略择优
"""

import time
import torch
from typing import Optional, Any


# ============================================================
# 工具函数
# ============================================================
def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算两个 tensor 的余弦相似度（展平后，强制同设备）。"""
    # 强制搬到 CPU，避免 GPU/CPU 设备不匹配
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
        return 1.0  # 无可比内容，视为完全一致
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
# 状态
# ============================================================
class V52FlashState:
    def __init__(self):
        self.step_idx = 0
        self.base_condition: Optional[Any] = None   # 基准 condition（冻结）
        self.base_input_shape = None                 # 形状保护
        self.cache_count = 0
        self.correct_count = 0

    def reset_base(self):
        self.base_condition = None
        self.base_input_shape = None


# ============================================================
# 调度器
# ============================================================
class V52FlashScheduler:
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
        """
        根据浓度和步号计算校正力度 α。
        返回 (alpha, reason)
        """
        # 浓度够高，不用校正
        if concentration >= self.conc_high:
            return (0.0, f"conc={concentration:.3f}>=high, no_correct")

        # 微抵抗区间：线性插值
        if concentration >= self.conc_low:
            raw_alpha = (self.conc_high - concentration) / (self.conc_high - self.conc_low) * self.max_alpha
        else:
            # 浓度很低，硬拉到 max_alpha
            raw_alpha = self.max_alpha

        # 阶段调整
        eff_end = self.total_steps - self.end_protect_steps

        if step < self.warmup_steps:
            # warmup：力度放大
            raw_alpha *= 1.5
            phase = f"warmup(x1.5)"
        elif step >= eff_end:
            # 末尾保护：力度缩小
            raw_alpha *= 0.3
            phase = f"end_protect(x0.3)"
        else:
            phase = "mid"

        # 红线：α 上限
        raw_alpha = min(raw_alpha, 0.5)

        return (raw_alpha, f"conc={concentration:.3f} a={raw_alpha:.3f} {phase}")


# ============================================================
# Wrapper
# ============================================================
def make_v52flash_wrapper(
    total_steps: int,
    warmup_steps: int,
    end_protect_steps: int,
    conc_high: float,
    conc_low: float,
    max_alpha: float,
    shape_protection: bool,
    report_stats: bool,
):
    state = V52FlashState()
    sched = V52FlashScheduler(
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

        # 形状保护：input 形状变了，重置基准
        if sched.shape_protection and state.base_input_shape is not None:
            if tuple(x.shape) != tuple(state.base_input_shape):
                if report_stats:
                    print(f"[V5.2flash] 形状变化 → 重置基准")
                state.reset_base()

        # 第一次：缓存基准 condition
        if state.base_condition is None:
            # 深拷贝 condition（CPU 上存，避免占显存）
            state.base_condition = _to_cpu_deep(c)
            state.base_input_shape = tuple(x.shape)
            if report_stats:
                print(f"[V5.2flash] step 0: 基准 condition 已缓存, shape={x.shape}")

        # 计算浓度（当前 condition vs 基准 condition）
        concentration = _deep_similarity(c, state.base_condition)

        # 计算校正力度
        alpha, reason = sched.compute_alpha(concentration, state.step_idx)

        # 混合 condition
        if alpha > 1e-6:
            corrected_c = _mix_condition(c, state.base_condition, alpha)
            state.correct_count += 1
        else:
            corrected_c = c

        # 全算（不跳步）
        t0 = time.perf_counter()
        output = apply_model(x, args["timestep"], **corrected_c)

        if report_stats:
            dt = time.perf_counter() - t0
            print(f"[V5.2flash] step {state.step_idx}/{total_steps} "
                  f"COMPUTE {dt:.1f}s {reason}")

        state.step_idx += 1
        return output

    return state, wrapper


def _to_cpu_deep(obj: Any) -> Any:
    """递归把嵌套结构中的 tensor 移到 CPU。"""
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
class MiniMaxH3TaijiArmorV52Flash:
    """末梦秋花 v5.2flash（认人）：condition 层微抵抗参考图校正。

    不改 latent，不跳步。每步检测参考图特征浓度，偏了就拉，不偏不动。
    弹簧式微抵抗：偏多了多拉，偏少少拉，不记状态每步重测。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 6, "min": 1, "max": 200,
                    "tooltip": "推理总步数"}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 50,
                    "tooltip": "前 N 步校正力度放大 x1.5，定型参考图"}),
                "end_protect_steps": ("INT", {"default": 2, "min": 0, "max": 20,
                    "tooltip": "最后 N 步校正力度缩小 x0.3，别动成片"}),
                "conc_high": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01,
                    "tooltip": "浓度高于此值不校正（没偏就不动）"}),
                "conc_low": ("FLOAT", {"default": 0.85, "min": 0.5, "max": 0.99, "step": 0.01,
                    "tooltip": "浓度低于此值硬拉（明显偏了）"}),
                "max_alpha": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "校正力度上限（红线，防止画面卡顿不动）"}),
                "shape_protection": ("BOOLEAN", {"default": True,
                    "tooltip": "形状变化自动重置基准"}),
                "report_stats": ("BOOLEAN", {"default": False,
                    "tooltip": "打印每步浓度和校正力度"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "末梦秋花v5.2s(逆风飘)"

    def apply(self, model, total_steps, warmup_steps, end_protect_steps,
              conc_high, conc_low, max_alpha, shape_protection, report_stats):
        state, wrapper = make_v52flash_wrapper(
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
        patched.model_options["xuanshu_v52flash"] = {"state": state}
        # 清理其他 wrapper，避免冲突
        for k in ["xuanshu_teacache_guard", "xuanshu_taiji_armor",
                  "xuanshu_taiji_armor_v4", "xuanshu_taiji_armor_v4_1",
                  "xuanshu_taiji_armor_v4_final", "xuanshu_taiji_armor_v4s",
                  "xuanshu_taiji_armor_v5", "xuanshu_taiji_armor_v5_1"]:
            patched.model_options.pop(k, None)
        return (patched,)


NODE_CLASS_MAPPINGS_V52FLASH = {"MiniMaxH3TaijiArmorV52Flash": MiniMaxH3TaijiArmorV52Flash}
NODE_DISPLAY_NAME_MAPPINGS_V52FLASH = {"MiniMaxH3TaijiArmorV52Flash": "末梦秋花v5.2s(逆风飘)"}
