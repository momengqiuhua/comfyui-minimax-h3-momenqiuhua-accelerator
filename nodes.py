"""XuanShu H3 自研优化节点集 v4（全方向迭代版）。

v4 新增：
- MiniMaxH3TaijiArmorV4: 太极装甲 v4（IEC + AdaCorrection + FeatFix + 频率解耦）
v3 保留：
- MiniMaxH3TaijiArmor: 太极装甲 v3（备用回退）
- MiniMaxH3TeaCacheGuard: Guard v2（备用回退）
"""
from __future__ import annotations

from .teacache_guard import TeaCacheGuardState, make_wrapper
from .taiji_armor import TaijiArmorState, make_armor_wrapper
from .taiji_armor_v4 import TaijiArmorV4State, make_v4_armor_wrapper
from .auto_optimizer import (
    MiniMaxH3AutoOptimizer,
    NODE_CLASS_MAPPINGS_AUTO,
    NODE_DISPLAY_NAME_MAPPINGS_AUTO,
)
# 2026-08-21 新节点导入（先进电网逻辑）
from .agc_auto import NODE_CLASS_MAPPINGS_AGC, NODE_DISPLAY_NAME_MAPPINGS_AGC
from .kernel_probe import NODE_CLASS_MAPPINGS_PROBE, NODE_DISPLAY_NAME_MAPPINGS_PROBE
from .selfheal import NODE_CLASS_MAPPINGS_SELFHEAL, NODE_DISPLAY_NAME_MAPPINGS_SELFHEAL
from .vsg import NODE_CLASS_MAPPINGS_VSG, NODE_DISPLAY_NAME_MAPPINGS_VSG
from .pss import NODE_CLASS_MAPPINGS_PSS, NODE_DISPLAY_NAME_MAPPINGS_PSS
from .dr import NODE_CLASS_MAPPINGS_DR, NODE_DISPLAY_NAME_MAPPINGS_DR
from .cache_selector import NODE_CLASS_MAPPINGS_SELECTOR, NODE_DISPLAY_NAME_MAPPINGS_SELECTOR
from .scada import NODE_CLASS_MAPPINGS_SCADA, NODE_DISPLAY_NAME_MAPPINGS_SCADA
from .taiji_armor_v4_1 import MiniMaxH3TaijiArmorV4_1
from .taiji_armor_v4_1 import NODE_CLASS_MAPPINGS_V41, NODE_DISPLAY_NAME_MAPPINGS_V41
from .taiji_armor_v4_final import NODE_CLASS_MAPPINGS_V4FINAL, NODE_DISPLAY_NAME_MAPPINGS_V4FINAL
from .taiji_armor_v3s import TaijiArmorV3SState, make_armor_wrapper_v3s
from .taiji_armor_v3s_ot import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_OT, NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_OT
# 2026-10-02 v3s1/v3t1 优化节点导入
from .taiji_armor_v3s1 import NODE_CLASS_MAPPINGS_V3S1, NODE_DISPLAY_NAME_MAPPINGS_V3S1
from .kernel_probe_v3t1 import NODE_CLASS_MAPPINGS_PROBE1, NODE_DISPLAY_NAME_MAPPINGS_PROBE1
# 2026-10-14 v4s 轨迹确定性节点导入
from .taiji_armor_v4s import NODE_CLASS_MAPPINGS_V4S, NODE_DISPLAY_NAME_MAPPINGS_V4S
# 2026-10-15 v5 极简截断漂移版导入
from .taiji_armor_v5 import NODE_CLASS_MAPPINGS_V5, NODE_DISPLAY_NAME_MAPPINGS_V5
from .taiji_armor_v5_1 import NODE_CLASS_MAPPINGS_V5_1, NODE_DISPLAY_NAME_MAPPINGS_V5_1
# 2026-10-17 v5.2flash 认人节点导入
from .taiji_armor_v5_2_flash import NODE_CLASS_MAPPINGS_V52FLASH, NODE_DISPLAY_NAME_MAPPINGS_V52FLASH
# 2026-10-17 v4t 等风来进化版（磁盘缓存+显存感知）
from .kernel_probe_v4t import NODE_CLASS_MAPPINGS_V4T, NODE_DISPLAY_NAME_MAPPINGS_V4T
# 2026-10-17 v5.2x 改进版（度量修复+PD控制+复位修复）
from .taiji_armor_v5_2_flash_x import NODE_CLASS_MAPPINGS_V52FLASHX, NODE_DISPLAY_NAME_MAPPINGS_V52FLASHX
# 2026-10-17 末梦秋花（逆风飘）v5x：v5.2flash 纯修复版（α 钳制 + 分段步号复位）
from .taiji_armor_v5_2_flash_y import NODE_CLASS_MAPPINGS_V52FLASHY, NODE_DISPLAY_NAME_MAPPINGS_V52FLASHY


class MiniMaxH3TeaCacheGuard:
    """自研增强 TeaCache v2：warmup 保护 + 动态阈值 + 命中监控 + 单步/累积模式。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01}),
                "start_step": ("INT", {"default": 2, "min": 0, "max": 100}),
                "end_step": ("INT", {"default": -2, "min": -20, "max": 100}),
                "total_steps": ("INT", {"default": 8, "min": 1, "max": 200}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 100,
                                        "tooltip": "前 N 步强制不缓存，让结构先稳定（治崩坏）"}),
                "adaptive_thresh": ("BOOLEAN", {"default": True,
                                                "tooltip": "低步数时自动更保守（动态放大阈值）"}),
                "accumulate_mode": ("BOOLEAN", {"default": False,
                                                "tooltip": "False=单步比较(推荐,命中率更高), True=累积比较(v1行为)"}),
                "verbose": ("BOOLEAN", {"default": True, "tooltip": "打印命中 step 位置用于监控"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "MiniMax H3 TeaCache Guard v2"

    def apply(self, model, rel_l1_thresh, start_step, end_step, total_steps,
              warmup_steps, adaptive_thresh, accumulate_mode, verbose):
        if warmup_steps >= total_steps:
            print(f"[XuanShuGuard v2] 警告：warmup({warmup_steps}) >= total({total_steps})，缓存完全失效")
        effective_end = end_step if end_step >= 0 else total_steps + end_step
        if start_step >= effective_end:
            print(f"[XuanShuGuard v2] 警告：start({start_step}) >= end({effective_end})，缓存窗口为空")

        if "xuanshu_taiji_armor" in model.model_options:
            print("[XuanShuGuard v2] 警告：检测到太极装甲已应用，Guard wrapper 将被覆盖")

        state = TeaCacheGuardState()
        state.reset()
        wrapper = make_wrapper(
            state=state,
            thresh=rel_l1_thresh,
            start_step=start_step,
            end_step=end_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            adaptive=adaptive_thresh,
            verbose=verbose,
            accumulate_mode=accumulate_mode,
        )
        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        patched.model_options["xuanshu_teacache_guard"] = {"state": state}
        patched.model_options.pop("xuanshu_taiji_armor", None)
        return (patched,)


class MiniMaxH3TaijiArmor:
    """玄枢太极装甲 v2：阴阳双轴 + 养蛊择优 + 周流显存 + 护体真元。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 6, "min": 1, "max": 200,
                                          "tooltip": "推理总步数"}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 100,
                                           "tooltip": "阳·刚步保护：前 N 步全算"}),
                "yang_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01,
                                             "tooltip": "最低阈值下限：实际阈值不低于此值"}),
                "end_step": ("INT", {"default": -2, "min": -20, "max": 100,
                                      "tooltip": "末段保护：最后 N 步全算（保护细节）"}),
                "adaptive": ("BOOLEAN", {"default": True,
                                            "tooltip": "自适应收敛：前期探索、后期保守"}),
                "enable_ensemble": ("BOOLEAN", {"default": True,
                                                    "tooltip": "养蛊择优：多策略选最稳"}),
                "vram_thresh_mb": ("INT", {"default": 7500, "min": 1000, "max": 20000,
                                              "tooltip": "显存水位阈值(MB)：超限自动更保守"}),
                "fallback_mode": ("BOOLEAN", {"default": True,
                                                 "tooltip": "护体真元：异常自动回退全算"}),
                "verbose": ("BOOLEAN", {"default": True}),
                "adaptive_scale": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 10.0, "step": 0.1,
                                               "tooltip": "v2自适应倍率：progress>=0.3时 thresh × 此值（poly_calibrate=True时被替代）"}),
                "poly_calibrate": ("BOOLEAN", {"default": True,
                                                "tooltip": "v3多项式校准：warmup期delta拟合多项式预测逐步阈值"}),
                "poly_degree": ("INT", {"default": 2, "min": 1, "max": 5,
                                          "tooltip": "多项式阶数：2=二次（推荐），1=线性"}),
                "safety_margin": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.1,
                                              "tooltip": "安全余量：thresh = 预测delta × 此值（2.0=50%命中率甜点）"}),
                "error_budget": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05,
                                             "tooltip": "误差预算：连续缓存累积误差超此值时强制计算"}),
                "max_consecutive": ("INT", {"default": 4, "min": 1, "max": 10,
                                              "tooltip": "连续缓存上限：连续命中超此值时强制计算"}),
                "correction_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                                    "tooltip": "缓存补偿强度：趋势外推修正因子（0=纯缓存，0.3=轻度修正）"}),
                "confidence_mode": ("BOOLEAN", {"default": True,
                                                 "tooltip": "v3置信度衰减：用连续概率替代二值缓存决策"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "MiniMax H3 太极装甲 v3"

    def apply(self, model, total_steps, warmup_steps, yang_thresh, end_step,
              adaptive, enable_ensemble, vram_thresh_mb, fallback_mode, verbose,
              adaptive_scale,
              poly_calibrate, poly_degree, safety_margin, error_budget,
              max_consecutive, correction_strength, confidence_mode):
        if warmup_steps >= total_steps:
            print(f"[XuanShuArmor v2] 警告：warmup({warmup_steps}) >= total({total_steps})，缓存完全失效")

        if "xuanshu_teacache_guard" in model.model_options:
            print("[XuanShuArmor v2] 警告：检测到 Guard 已应用，装甲 wrapper 将被覆盖")

        state = TaijiArmorState()
        state.reset()
        wrapper = make_armor_wrapper(
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
            error_budget=error_budget,
            max_consecutive=max_consecutive,
            correction_strength=correction_strength,
            confidence_mode=confidence_mode,
        )
        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        patched.model_options["xuanshu_taiji_armor"] = {"state": state}
        patched.model_options.pop("xuanshu_teacache_guard", None)
        return (patched,)


class XuanShuV3SDiag:
    """玄枢 v3s 误缓存诊断：Shadow Mode，缓存命中时仍计算真实输出对比。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 6, "min": 1, "max": 200,
                                          "tooltip": "推理总步数"}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 100,
                                           "tooltip": "前N步强制全算"}),
                "signal_mode": (["l1", "ot", "residual", "cosine", "weighted"],),
                "decision_thresh": ("FLOAT", {"default": 0.15, "min": 0.001, "max": 1.0, "step": 0.001,
                                                "tooltip": "缓存判定阈值：input delta < 此值才可能缓存"}),
                "end_step": ("INT", {"default": -2, "min": -20, "max": 100,
                                      "tooltip": "末段保护步数"}),
                "enable_ensemble": ("BOOLEAN", {"default": True}),
                "vram_thresh_mb": ("INT", {"default": 7500, "min": 1000, "max": 20000}),
                "fallback_mode": ("BOOLEAN", {"default": True}),
                "error_budget": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05}),
                "max_consecutive": ("INT", {"default": 4, "min": 1, "max": 10}),
                "correction_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "confidence_mode": ("BOOLEAN", {"default": True}),
                "iec_enabled": ("BOOLEAN", {"default": True}),
                "iec_gamma": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
                "false_hit_thresh": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 0.5, "step": 0.001,
                                                "tooltip": "output delta超过此值判定为误缓存"}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "玄枢 v3s 误缓存诊断"

    def apply(self, model, total_steps, warmup_steps, signal_mode, decision_thresh,
              end_step, enable_ensemble, vram_thresh_mb, fallback_mode, error_budget,
              max_consecutive, correction_strength, confidence_mode,
              iec_enabled, iec_gamma, false_hit_thresh, verbose):
        state = DiagState()
        state.reset()
        state.false_hit_thresh = false_hit_thresh
        wrapper = make_diag_wrapper(
            state=state,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            signal_mode=signal_mode,
            decision_thresh=decision_thresh,
            yang_thresh=0.001,
            end_step=end_step,
            enable_ensemble=enable_ensemble,
            vram_thresh_mb=vram_thresh_mb,
            fallback_mode=fallback_mode,
            error_budget=error_budget,
            max_consecutive=max_consecutive,
            correction_strength=correction_strength,
            confidence_mode=confidence_mode,
            iec_enabled=iec_enabled,
            iec_gamma=iec_gamma,
            false_hit_thresh=false_hit_thresh,
            verbose=verbose,
        )
        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        patched.model_options["xuanshu_v3s_diag"] = {"state": state}
        return (patched,)


class MiniMaxH3TaijiArmorV4:
    """玄枢太极装甲 v4：IEC + AdaCorrection + FeatFix + 频率解耦。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 6, "min": 1, "max": 200,
                                          "tooltip": "推理总步数"}),
                "warmup_steps": ("INT", {"default": 2, "min": 0, "max": 100,
                                           "tooltip": "v4默认2步warmup建立趋势后缓存"}),
                "yang_thresh": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.5, "step": 0.01,
                                             "tooltip": "最低阈值下限"}),
                "end_step": ("INT", {"default": -2, "min": -20, "max": 100,
                                      "tooltip": "末段保护：v4推荐-2（末2步全算，保证结尾收敛）"}),
                "adaptive": ("BOOLEAN", {"default": True}),
                "enable_ensemble": ("BOOLEAN", {"default": True}),
                "vram_thresh_mb": ("INT", {"default": 7500, "min": 1000, "max": 20000}),
                "fallback_mode": ("BOOLEAN", {"default": True}),
                "verbose": ("BOOLEAN", {"default": True}),
                # v3 保留
                "poly_calibrate": ("BOOLEAN", {"default": True}),
                "poly_degree": ("INT", {"default": 2, "min": 1, "max": 5}),
                "safety_margin": ("FLOAT", {"default": 1.8, "min": 1.0, "max": 3.0, "step": 0.1}),
                "adaptive_scale": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 10.0, "step": 0.1,
                                              "tooltip": "自适应阈值放大倍率：v4推荐1.5（过高会过度缓存导致画面过冲）"}),
                "correction_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                                                   "tooltip": "趋势外推补偿强度：v4推荐0.25（过高导致细节过冲/泛白）"}),
                # === v4 ===
                "iec_enabled": ("BOOLEAN", {"default": True,
                                             "tooltip": "IEC迭代误差校正：缓存步误差序列加权回滚，替代硬熔断"}),
                "iec_gamma": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1,
                                         "tooltip": "IEC衰减因子：越小误差增长越慢(0.6=线性增长)"}),
                "adac_enabled": ("BOOLEAN", {"default": True,
                                              "tooltip": "AdaCorrection自适应混合：alpha*cached+(1-alpha)*computed，替代二值"}),
                "adac_min_alpha": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                              "tooltip": "AdaCorrection最小混合系数：0.3=缓存步仍保留30%原始信号"}),
                "featfix_enabled": ("BOOLEAN", {"default": False,
                                                 "tooltip": "FeatFix验证特征复用：逻辑已禁用（会破坏画面），仅作开关记录"}),
                "featfix_strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05}),
                "freqdec_enabled": ("BOOLEAN", {"default": False,
                                                 "tooltip": "频率解耦补偿：DCT分解缓存输出，低频直接复用高频Taylor重建(实验性)"}),
                "freqdec_threshold": ("FLOAT", {"default": 0.15, "min": 0.05, "max": 0.5, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "MiniMax H3 太极装甲 v4"

    def apply(self, model, total_steps, warmup_steps, yang_thresh, end_step,
              adaptive, enable_ensemble, vram_thresh_mb, fallback_mode, verbose,
              poly_calibrate, poly_degree, safety_margin, adaptive_scale,
              correction_strength,
              iec_enabled, iec_gamma, adac_enabled, adac_min_alpha,
              featfix_enabled, featfix_strength, freqdec_enabled, freqdec_threshold):
        if "xuanshu_teacache_guard" in model.model_options:
            print("[XuanShuV4] 警告：检测到 Guard 已应用，v4 wrapper 将被覆盖")

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
        patched.model_options["xuanshu_taiji_armor_v4"] = {"state": state}
        patched.model_options.pop("xuanshu_teacache_guard", None)
        patched.model_options.pop("xuanshu_taiji_armor", None)
        return (patched,)


class MiniMaxH3TaijiArmorV3S:
    """玄枢太极装甲 v3s：v3修复版，保留全部v3参数+新增IEC+增强形状保护。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 6, "min": 1, "max": 200,
                                          "tooltip": "推理总步数"}),
                "warmup_steps": ("INT", {"default": 3, "min": 0, "max": 100,
                                           "tooltip": "阳·刚步保护：前 N 步全算"}),
                "yang_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01,
                                             "tooltip": "最低阈值下限：实际阈值不低于此值"}),
                "end_step": ("INT", {"default": -2, "min": -20, "max": 100,
                                      "tooltip": "末段保护：最后 N 步全算（保护细节）"}),
                "adaptive": ("BOOLEAN", {"default": True,
                                            "tooltip": "自适应收敛：前期探索、后期保守"}),
                "enable_ensemble": ("BOOLEAN", {"default": True,
                                                    "tooltip": "养蛊择优：多策略选最稳"}),
                "vram_thresh_mb": ("INT", {"default": 7500, "min": 1000, "max": 20000,
                                              "tooltip": "显存水位阈值(MB)：超限自动更保守"}),
                "fallback_mode": ("BOOLEAN", {"default": True,
                                                 "tooltip": "护体真元：异常自动回退全算"}),
                "verbose": ("BOOLEAN", {"default": True}),
                "adaptive_scale": ("FLOAT", {"default": 2.5, "min": 0.5, "max": 10.0, "step": 0.1,
                                               "tooltip": "v2自适应倍率：progress>=0.3时 thresh × 此值"}),
                "poly_calibrate": ("BOOLEAN", {"default": True,
                                                "tooltip": "v3多项式校准：warmup期delta拟合多项式预测逐步阈值"}),
                "poly_degree": ("INT", {"default": 2, "min": 1, "max": 5,
                                          "tooltip": "多项式阶数：2=二次（推荐），1=线性"}),
                "safety_margin": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 3.0, "step": 0.1,
                                              "tooltip": "安全余量：thresh = 预测delta × 此值"}),
                "error_budget": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05,
                                             "tooltip": "误差预算：连续缓存累积误差超此值时强制计算"}),
                "max_consecutive": ("INT", {"default": 4, "min": 1, "max": 10,
                                              "tooltip": "连续缓存上限：连续命中超此值时强制计算"}),
                "correction_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                                    "tooltip": "缓存补偿强度：趋势外推修正因子"}),
                "confidence_mode": ("BOOLEAN", {"default": True,
                                                 "tooltip": "v3置信度衰减：用连续概率替代二值缓存决策"}),
                # === v3s 新增 ===
                "iec_enabled": ("BOOLEAN", {"default": True,
                                             "tooltip": "IEC迭代误差校正：历史误差指数加权 → 阈值惩罚，动态收紧缓存"}),
                "iec_gamma": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1,
                                         "tooltip": "IEC衰减因子：越大惩罚越强(0.6=适中)"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "XuanShu/H3"
    TITLE = "MiniMax H3 太极装甲 v3s"

    def apply(self, model, total_steps, warmup_steps, yang_thresh, end_step,
              adaptive, enable_ensemble, vram_thresh_mb, fallback_mode, verbose,
              adaptive_scale,
              poly_calibrate, poly_degree, safety_margin, error_budget,
              max_consecutive, correction_strength, confidence_mode,
              iec_enabled, iec_gamma):
        if warmup_steps >= total_steps:
            print(f"[TaijiArmor v3s] 警告：warmup({warmup_steps}) >= total({total_steps})，缓存完全失效")

        if "xuanshu_teacache_guard" in model.model_options:
            print("[TaijiArmor v3s] 警告：检测到 Guard 已应用，装甲 wrapper 将被覆盖")

        state = TaijiArmorV3SState()
        state.reset()
        wrapper = make_armor_wrapper_v3s(
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
            error_budget=error_budget,
            max_consecutive=max_consecutive,
            correction_strength=correction_strength,
            confidence_mode=confidence_mode,
            iec_enabled=iec_enabled,
            iec_gamma=iec_gamma,
        )
        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        patched.model_options["xuanshu_taiji_armor_v3s"] = {"state": state}
        patched.model_options.pop("xuanshu_teacache_guard", None)
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TeaCacheGuard": MiniMaxH3TeaCacheGuard,
    "MiniMaxH3TaijiArmor": MiniMaxH3TaijiArmor,
    "MiniMaxH3TaijiArmorV4": MiniMaxH3TaijiArmorV4,
    "MiniMaxH3TaijiArmorV3S": MiniMaxH3TaijiArmorV3S,
    **NODE_CLASS_MAPPINGS_AUTO,
    **NODE_CLASS_MAPPINGS_AGC,
    **NODE_CLASS_MAPPINGS_SELFHEAL,
    **NODE_CLASS_MAPPINGS_VSG,
    **NODE_CLASS_MAPPINGS_PSS,
    **NODE_CLASS_MAPPINGS_DR,
    **NODE_CLASS_MAPPINGS_SELECTOR,
    **NODE_CLASS_MAPPINGS_SCADA,
    **NODE_CLASS_MAPPINGS_V41,
    **NODE_CLASS_MAPPINGS_V4FINAL,
    **NODE_CLASS_MAPPINGS_OT,
    **NODE_CLASS_MAPPINGS_PROBE,
    **NODE_CLASS_MAPPINGS_V3S1,
    **NODE_CLASS_MAPPINGS_PROBE1,
    **NODE_CLASS_MAPPINGS_V4S,
    **NODE_CLASS_MAPPINGS_V5,
    **NODE_CLASS_MAPPINGS_V5_1,
    **NODE_CLASS_MAPPINGS_V52FLASH,
    **NODE_CLASS_MAPPINGS_V52FLASHX,
    **NODE_CLASS_MAPPINGS_V52FLASHY,
    **NODE_CLASS_MAPPINGS_V4T,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TeaCacheGuard": "MiniMax H3 TeaCache Guard v2",
    "MiniMaxH3TaijiArmor": "MiniMax H3 太极装甲 v3",
    "MiniMaxH3TaijiArmorV4": "MiniMax H3 太极装甲 v4",
    "MiniMaxH3TaijiArmorV3S": "MiniMax H3 太极装甲 v3s",
    **NODE_DISPLAY_NAME_MAPPINGS_AUTO,
    **NODE_DISPLAY_NAME_MAPPINGS_AGC,
    **NODE_DISPLAY_NAME_MAPPINGS_SELFHEAL,
    **NODE_DISPLAY_NAME_MAPPINGS_VSG,
    **NODE_DISPLAY_NAME_MAPPINGS_PSS,
    **NODE_DISPLAY_NAME_MAPPINGS_DR,
    **NODE_DISPLAY_NAME_MAPPINGS_SELECTOR,
    **NODE_DISPLAY_NAME_MAPPINGS_SCADA,
    **NODE_DISPLAY_NAME_MAPPINGS_V41,
    **NODE_DISPLAY_NAME_MAPPINGS_V4FINAL,
    **NODE_DISPLAY_NAME_MAPPINGS_OT,
    **NODE_DISPLAY_NAME_MAPPINGS_PROBE,
    **NODE_DISPLAY_NAME_MAPPINGS_V3S1,
    **NODE_DISPLAY_NAME_MAPPINGS_PROBE1,
    **NODE_DISPLAY_NAME_MAPPINGS_V4S,
    **NODE_DISPLAY_NAME_MAPPINGS_V5,
    **NODE_DISPLAY_NAME_MAPPINGS_V5_1,
    **NODE_DISPLAY_NAME_MAPPINGS_V52FLASH,
    **NODE_DISPLAY_NAME_MAPPINGS_V52FLASHX,
    **NODE_DISPLAY_NAME_MAPPINGS_V52FLASHY,
    **NODE_DISPLAY_NAME_MAPPINGS_V4T,
}
