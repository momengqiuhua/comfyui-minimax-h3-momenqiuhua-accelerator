"""XuanShu H3 Optimizer v2 — 玄枢 H3 自研优化节点插件（迭代版）。

针对 RTX 5060 8GB 设备定制，基于 28 组消融实测 + abcde 五组验证 + v1 缺陷修复。
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
