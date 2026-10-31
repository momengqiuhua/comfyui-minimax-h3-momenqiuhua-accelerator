# XuanShu H3 Optimizer · 玄枢 H3 优化器

MiniMax H3 acceleration & quality nodes for ComfyUI — tuned and ablated on RTX 5060 8GB.
MiniMax H3 加速与画质优化节点套件，专为 RTX 5060 8GB 实测调优。

**末梦秋花系列 · 旗舰三件套**：等风来（无损加速）· 逆风飘（保人物一致）· 顺风飘（跳步加速）

| Node 节点 | Class | What it does 干什么 |
|---|---|---|
| **末梦秋花v4t(等风来)** | `XuanShuH3KernelProbeV4T` | Picks the fastest attention backend on the fly — zero quality loss 自动选最快注意力后端，零画质损失 |
| **末梦秋花(逆风飘)v5x** | `MiniMaxH3TaijiArmorV52FlashY` | Keeps characters consistent with reference images in Ref2VA workflows 保 Ref2VA 参考图人物一致 |
| **末梦秋花v3s1(顺风飘)** | `MiniMaxH3TaijiArmorV3S1` | Skips redundant steps on 20-step standard chains 20 步标准链跳步缓存加速 |

---

## English

### Install

1. Clone or download this repo into `ComfyUI/custom_nodes/`
2. Restart ComfyUI
3. Search `末梦秋花` in the node menu

Or install via [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager) (pending listing).

### Quick start

```
UNET Loader → 末梦秋花v4t(等风来) → KSampler
```

One node, automatic attention-backend probing. First run spends a few seconds benchmarking 3 backends (pytorch / sage / ck_int8); the winner is cached to disk and reused on every later launch.

### 1. v4t (等风来) — Lossless acceleration

Intercepts H3 attention, benchmarks backends live, routes to the fastest. No step-skipping, no latent edits, no approximation.

- **Disk cache**: probe once, remember forever
- **VRAM-aware fallback**: auto-degrades to pytorch under memory pressure instead of OOM-crashing

| Param | Default | Note |
|---|---|---|
| mode | auto_probe | passthrough = do nothing |
| quality_mode | speed_priority | lossless_priority = pytorch first |
| vram_aware | true | auto fallback on low VRAM |
| vram_margin_mb | 200 | safety margin |

### 2. v5x (逆风飘) — Character consistency

For Ref2VA three-reference workflows (scene + girl + boy). Prevents characters from being "sanded flat" by multi-step sampling.

Works on the **condition layer only** (never touches latent): periodically re-measures reference-feature concentration and applies spring-like micro-correction — pulls when drifting, idle when not. v5x adds α clamping and per-segment step-index reset over v5.2s.

| Param | Default | Note |
|---|---|---|
| total_steps | 6 | must match sampler steps (per segment) |
| warmup_steps | 3 | stronger correction early to lock references |
| end_protect_steps | 2 | weaker correction at the end, don't touch the final look |
| conc_high / conc_low | 0.95 / 0.85 | no correction above / hard pull below |
| max_alpha | 0.3 | correction strength cap |

### 3. v3s1 (顺风飘) — Step-skipping acceleration

On 20-step standard chains, detects steps whose outputs barely change and reuses the previous result. **Do not use on 4-step distilled chains** (LightX2V 4step LoRA) — only 1 skippable step, not worth the risk.

Keep `total_steps` aligned with your sampler; defaults are fine otherwise.

### Recommended combos

```
# A. 4-step distilled LoRA + Ref2VA (consistency + speed)
UNET → v4t → v5x → KSampler(4 steps)

# B. 20-step standard chain, max speed (T2V)
UNET → v4t → v3s1 → KSampler(20 steps)
```

The suite's wrappers are mutually exclusive — chaining two cache-type nodes lets the later one override the earlier.

### Requirements

- RTX 5060 8GB (verified) / Blackwell GPUs
- ComfyUI 0.41+

---

## 中文文档

### 安装

1. 下载本仓库，解压到 `ComfyUI/custom_nodes/`
2. 重启 ComfyUI
3. 搜索「末梦秋花」找到节点

国内用户也可从[魔搭镜像](https://modelscope.cn/models/wumingzhibai/comfyui-minimax-h3-momenqiuhua-accelerator)下载。

### 最简用法

```
加载模型 → 末梦秋花v4t(等风来) → 采样器
```

就一个节点，自动选最快的注意力后端，零画质损失。第一次跑会探针几秒钟，之后自动记住。

### 1. 末梦秋花v4t(等风来) —— 无损加速

拦截注意力计算，现场比武选最快的后端（pytorch / sage / ck_int8）。不跳步、不改 latent、不做近似——每次都全算，只是选最快的算路。

- **磁盘缓存**：选过一次就记住，下次启动 ComfyUI 直接用，跳过探针
- **显存感知降级**：长序列显存紧张时自动从 ck_int8 降级到 pytorch，不 OOM

### 2. 末梦秋花(逆风飘)v5x —— 保人物一致

Ref2VA 三参考图（场景+女生+男生）工作流中，防止人物被多步采样磨平。

只在 condition 层定期校正参考图特征权重，弹簧式微抵抗——偏了就拉，不偏不动，不记状态每步重测。v5x 在 v5.2s 基础上修复了 α 越限与分段步号复位两处缺陷。

| 参数 | 默认值 | 说明 |
|---|---|---|
| total_steps | 6 | 单段采样总步数，必须跟采样器对齐 |
| warmup_steps | 3 | 前 N 步校正力度放大，定型参考图 |
| end_protect_steps | 2 | 最后 N 步降低校正力度，别动成片 |
| conc_high | 0.95 | 浓度高于此值不校正（没偏就不动） |
| conc_low | 0.85 | 浓度低于此值硬拉（明显偏了） |
| max_alpha | 0.3 | 校正力度上限（画面变卡就调小到 0.15） |
| shape_protection | true | 形状变化自动分段复位 |

### 3. 末梦秋花v3s1(顺风飘) —— 跳步加速

20 步标准链上，判断哪些步可以跳过不算，直接复用上一步结果。

**注意**：4 步蒸馏链（LightX2V 4step LoRA）上不要用——4 步里只有 1 步可跳，省的时间不值人脸崩的风险。

### 推荐组合

```
# 组合 A：蒸馏 LoRA 4 步跑 Ref2VA（人物一致 + 快）
UNET → v4t(等风来) → v5x(逆风飘) → KSampler(4步)

# 组合 B：20 步标准链全速跑（文生视频）
UNET → v4t(等风来) → v3s1(顺风飘) → KSampler(20步)
```

本套件节点互斥——后串的会覆盖先串的，不要同时串两个缓存类节点。

### 常见问题

**Q: v4t 第一次跑为什么等了几秒？**
A: 在探针测试 3 个后端速度，选最快的。之后存到磁盘，第二次跑直接用。

**Q: 控制台一直刷「显存不足降级 pytorch」？**
A: 把 `vram_margin_mb` 从 200 调到 100。

**Q: v5x 跑出来画面变卡了？**
A: `max_alpha` 太大，默认 0.3 调到 0.15 试试。

**Q: v3s1 在 4 步 LoRA 上为什么没效果？**
A: 4 步里只有 1 步可跳，省的时间约等于零。v3s1 是给 20 步标准链用的。

### 硬件要求

- RTX 5060 8GB（已验证）/ Blackwell 架构 GPU
- ComfyUI 0.41+

---

## 更新日志 Changelog

| 日期 | 内容 |
|---|---|
| 2026-10-20 | 精简为旗舰三件套（v4t/v5x/v3s1）；逆风飘定型为 v5x（α钳制+分段复位修复版）；中英双语 README |
| 2026-10-17 | 新增 v4t（磁盘缓存+显存感知）、v5.2s 逆风飘（condition层微抵抗校正） |
| 2026-10-07 | v3s1/v3t1 定型 |
| 2026-08-24 | v3s 首次发布 |

## License

MIT © 末梦秋华 (momengqiuhua)
