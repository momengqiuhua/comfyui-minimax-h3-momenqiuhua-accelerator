"""XuanShu H3 Auto Optimizer v2 — 一键最优配置决策（迭代版）。

v2 修复：
1. SolAttn 全部改为 False（与 abcde 测试结论一致）
2. 命中率基于实测数据修正
3. 增加竖屏 608×1088 预设
4. 预测耗时基于实测中位数
"""
from __future__ import annotations

OPTIMAL = {
    (480, 864): {
        "速度":  {"steps": 6, "warmup": 3, "lora": True, "sol": False, "thresh": 0.15,
                  "eta_s": 95, "hit": 5.0, "note": "Guard6步，最快生产"},
        "平衡":  {"steps": 8, "warmup": 3, "lora": True, "sol": False, "thresh": 0.15,
                  "eta_s": 131, "hit": 8.0, "note": "Guard8步，画质基准(D2方案)"},
        "画质":  {"steps": 12, "warmup": 3, "lora": True, "sol": False, "thresh": 0.12,
                  "eta_s": 180, "hit": 15.0, "note": "Guard12步，画质优先"},
    },
    (576, 1024): {
        "速度":  {"steps": 8, "warmup": 3, "lora": True, "sol": False, "thresh": 0.15,
                  "eta_s": 170, "hit": 8.0, "note": "高分辨率速度档"},
        "平衡":  {"steps": 10, "warmup": 4, "lora": True, "sol": False, "thresh": 0.12,
                  "eta_s": 200, "hit": 12.0, "note": "高分辨率平衡"},
        "画质":  {"steps": 20, "warmup": 4, "lora": False, "sol": False, "thresh": 0.10,
                  "eta_s": 390, "hit": 30.0, "note": "高分辨率画质优先，无LoRA"},
    },
    (608, 1088): {
        "速度":  {"steps": 6, "warmup": 3, "lora": True, "sol": False, "thresh": 0.15,
                  "eta_s": 105, "hit": 5.0, "note": "竖屏速度档"},
        "平衡":  {"steps": 8, "warmup": 3, "lora": True, "sol": False, "thresh": 0.15,
                  "eta_s": 145, "hit": 8.0, "note": "竖屏平衡档"},
        "画质":  {"steps": 12, "warmup": 3, "lora": True, "sol": False, "thresh": 0.12,
                  "eta_s": 200, "hit": 15.0, "note": "竖屏画质档"},
    },
}


class MiniMaxH3AutoOptimizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 480, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 864, "min": 256, "max": 2048, "step": 32}),
                "quality_mode": (["速度", "平衡", "画质"], {"default": "平衡"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "BOOLEAN", "BOOLEAN", "FLOAT")
    RETURN_NAMES = ("建议", "steps", "warmup", "use_lora", "use_sol", "tea_thresh")
    FUNCTION = "decide"
    CATEGORY = "XuanShu/H3"
    TITLE = "MiniMax H3 Auto Optimizer v2"

    def decide(self, width, height, quality_mode):
        key = self._nearest((width, height))
        cfg = OPTIMAL[key]
        c = cfg[quality_mode]
        w, h = key
        advice = (
            f"[XuanShu Auto v2] {w}x{h} / {quality_mode}模式："
            f"steps={c['steps']} warmup={c['warmup']} "
            f"LoRA={'是' if c['lora'] else '否'} Sol={'是' if c['sol'] else '否'} "
            f"thresh={c['thresh']} | 预计{c['eta_s']}s 命中{c['hit']}% | {c['note']}"
        )
        return (advice, c["steps"], c["warmup"], c["lora"], c["sol"], c["thresh"])

    @staticmethod
    def _nearest(size):
        best = None
        bd = 1e18
        target_ratio = size[0] / max(1, size[1])
        for k in OPTIMAL:
            k_ratio = k[0] / max(1, k[1])
            ratio_pen = abs(k_ratio - target_ratio) * 5000
            d = abs(k[0] - size[0]) + abs(k[1] - size[1]) + ratio_pen
            if d < bd:
                bd = d
                best = k
        return best


NODE_CLASS_MAPPINGS_AUTO = {
    "MiniMaxH3AutoOptimizer": MiniMaxH3AutoOptimizer,
}
NODE_DISPLAY_NAME_MAPPINGS_AUTO = {
    "MiniMaxH3AutoOptimizer": "MiniMax H3 Auto Optimizer v2",
}
