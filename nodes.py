"""XuanShu H3 自研优化节点集 — 末梦秋花系列（旗舰三件套）。

- 末梦秋花v4t(等风来)：注意力 kernel 寻优 + 磁盘缓存 + 显存感知降级，无损加速
- 末梦秋花(逆风飘)v5x：Ref2VA 参考图人物一致性守护（condition 层微抵抗校正）
- 末梦秋花v3s1(顺风飘)：20 步标准链跳步缓存加速

历史迭代版本（v3/v3s/v4/v5 等）文件保留在仓库中备查，不再注册。
"""
from __future__ import annotations

from .kernel_probe_v4t import (
    NODE_CLASS_MAPPINGS_V4T,
    NODE_DISPLAY_NAME_MAPPINGS_V4T,
)
from .taiji_armor_v5_2_flash_y import (
    NODE_CLASS_MAPPINGS_V52FLASHY,
    NODE_DISPLAY_NAME_MAPPINGS_V52FLASHY,
)
from .taiji_armor_v3s1 import (
    NODE_CLASS_MAPPINGS_V3S1,
    NODE_DISPLAY_NAME_MAPPINGS_V3S1,
)

NODE_CLASS_MAPPINGS = {
    **NODE_CLASS_MAPPINGS_V4T,
    **NODE_CLASS_MAPPINGS_V52FLASHY,
    **NODE_CLASS_MAPPINGS_V3S1,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **NODE_DISPLAY_NAME_MAPPINGS_V4T,
    **NODE_DISPLAY_NAME_MAPPINGS_V52FLASHY,
    **NODE_DISPLAY_NAME_MAPPINGS_V3S1,
}
