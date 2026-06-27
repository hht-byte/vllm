# COAST 非侵入式适配 vLLM

## 一、背景与问题

### 1.1 问题概述

大型视觉语言模型（LVLMs）在处理高分辨率图像时，视觉 token 数量急剧增长，导致 prefill 阶段计算与显存开销大幅增加、推理延迟上升。现有剪枝方法（如 FastV）依赖浅层 text-to-image 注意力分数进行一次性 scalar 排序剪枝，但这种策略对组合推理不可靠：浅层低注意力的 token 在深层可能变得至关重要（用于解析次要对象、空间关系、上下文线索）。过早剪枝会导致 **Visual Aphasia**（视觉失语症）——模型丧失视觉锚定能力，退化为依赖语言先验。

### 1.2 COAST 方法

COAST（COntrastive Adaptive Semantic Token Pruning，论文 https://arxiv.org/abs/2605.09429）是一种 **免训练（training-free）** 的视觉 token 剪枝框架，其核心思路：

1. **熵驱动动态预算分配**：利用跨模态注意力/相似度的归一化熵 H 估计场景级上下文分散度。低熵→注意力集中→更多预算分配给语义锚点；高熵→注意力分散→更多预算分配给互补空间上下文。
2. **锚点-参照对比语义路由**：选择查询相关的锚点 token（S^last 高分）和低显著度参照 token（S^glo 低分），对候选 token 计算对比分数 Score(c_i) = Sim_A(c_i) - Sim_R(c_i)，保留分数分布的两端：top-n1（锚点对齐证据）和 bottom-n2（互补空间上下文）。
3. **两尾保留**：避免压缩后的视觉序列仅由最显著的查询区域主导，保留边缘上下文证据以防止 Visual Aphasia。

论文实验表明：在 LLaVA-v1.5-7B 上保留 22.2% token（77.8% 剪枝率）可保留 98.64% 原始性能，2.15x 延迟加速。

### 1.3 适配挑战

将 COAST 适配到 vLLM 需要解决以下问题：

- **跨模态注意力获取**：COAST 论文使用 LLM 解码器层的实际注意力概率矩阵，但 vLLM 的 FlashInfer/CUDA 后端不暴露注意力权重。适配方案：在编码器输出层使用视觉自相似度作为代理；在中间层使用隐藏状态的跨模态余弦相似度作为代理。
- **PlaceholderRange 同步**：剪枝后 token 数减少，PlaceholderRange.length 必须与实际保留数一致。
- **M-RoPE 位置重算**：Qwen2.5-VL/Qwen3-VL 使用多维旋转位置编码，剪枝后位置编码需要重新计算。
- **中间层序列缩减**：vLLM 的注意力元数据在 forward 前预计算，无法在中间层动态缩减序列长度。适配方案：将剪枝 token 的信息合并到最近保留 token（加权平均），保持序列长度不变。
- **版本迁移**：vLLM 版本迭代频繁，侵入式修改难以跟随升级。

## 二、关键设计

### 2.1 非侵入式原则

核心设计目标是 **非侵入式（方便迁移到 vLLM 新版本）**，具体体现为：

| 层级 | 内容 | 侵入程度 |
|------|------|---------|
| **算法层** | `vllm/multimodal/coast.py` — 纯算法模块，不涉及模型细节 | 新增文件，0 行修改 |
| **配置层** | `vllm/config/multimodal.py` — 5 个配置字段 + `is_coast_enabled()` | 新增 ~40 行 |
| **管线层** | `arg_utils.py` + `model.py` — CLI 参数注册与传递 | 新增 ~40 行 |
| **初始化层** | `vllm/config/vllm.py` — `__post_init__` 自动检测与应用 patches | 新增 ~15 行 |
| **适配层** | `coast_vllm/` — 全部 monkey-patch，独立包，不修改 vllm 源码 | 0 行修改 vllm |

**vllm 源码总修改量**：约 95 行新增，0 行删除。迁移到新版本时只需重新添加这些新增行。

### 2.2 两层剪枝架构

COAST 可作用于两个层级：

```
┌─────────────────────────────────────────┐
│  1. Encoder Output Level                 │  ← 视觉编码器输出后
│     patch: _process_image_input()        │
│     patch: _process_video_input()        │
│     使用视觉自相似度作为跨模态代理信号    │
│     主要的 token 数缩减发生在此层         │
├─────────────────────────────────────────┤
│  2. Processor Level                      │  ← 预处理阶段
│     patch: _get_prompt_updates()         │
│     更新 PlaceholderRange.length         │
├─────────────────────────────────────────┤
│  3. Intermediate LLM Layer (可选)        │  ← LLM 中间层
│     patch: LlamaModel.forward()          │
│     patch: Qwen2Model.forward()          │
│     使用隐藏状态跨模态相似度进行路由      │
│     剪枝 token 合并到最近保留 token       │
└─────────────────────────────────────────┘
```

### 2.3 COAST 算法详解

#### 编码器输出层算法

由于编码器输出时还没有 LLM 的跨模态注意力，COAST 使用 **视觉 token 自相似度** 作为代理：

1. 计算全局重要性分数 S^glo_j = mean_{i≠j} cos_sim(X_v_i, X_v_j)（每个 token 与其他 token 的平均相似度，捕捉"中心性"）
2. 计算归一化熵 H = -1/log(N_v) * Σ p_j * log(p_j)，其中 p_j = S^glo_j / Σ S^glo
3. 选择 K_anchor = floor(num_retain × anchor_ratio) 个锚点（S^glo 最高的 token）
4. 分配剩余预算：n2 = floor(K_rest × (α_min + (α_max - α_min) × H)), n1 = K_rest - n2
5. 选择 K_R 个参照 token（S^glo 最低的 token）
6. 对候选 token 计算对比分数：Score(c_i) = Sim_A(c_i) - Sim_R(c_i)
7. 保留 top-n1（锚点对齐）+ bottom-n2（互补上下文）+ 锚点
8. 按原始顺序排序保留的 token

#### 中间层算法（可选）

在 LLM 解码器的指定层，使用隐藏状态的 **跨模态余弦相似度** 作为代理：

1. 从 hidden_states 中提取视觉 token 和文本 token 的隐藏状态
2. 计算跨模态相似度矩阵 S^cross = cos_sim(H_t, H_v)
3. S^glo_j = max_i S^cross_ij（最大文本-视觉相似度）
4. S^last_j = S^cross[last_text_idx, j]（最后文本 token 的相似度）
5. 计算熵 H，分配预算，对比路由（同编码器层算法）
6. 保留的 token 保持原值；剪枝 token 的信息合并到最近保留 token（加权平均）
7. 序列长度不变，但剪枝位置被替换为合并后的均值

### 2.4 与 EVS/LRCP 的类比设计

| 方面 | EVS | LRCP | COAST |
|------|-----|------|-------|
| 算法模块位置 | `vllm/multimodal/evs.py` | `vllm/multimodal/lrcp.py` | `vllm/multimodal/coast.py` |
| 核心机制 | 帧间余弦相似度 | PCA 投影残差 | 熵驱动预算 + 对比路由 |
| 信号来源 | 视觉帧间 | 视觉子空间 | 跨模态相似度/视觉自相似度 |
| 保留策略 | Top-K 保留 | Top-K 残差 | 两尾保留（锚点+上下文） |
| 适配包 | 内置 | `lrcp_vllm/` | `coast_vllm/` |
| 适用范围 | 仅视频 | 图像+视频 | 图像+视频 |

### 2.5 自动 Patch 机制

在 `VllmConfig.__post_init__()` 中添加 `_apply_coast_patches()` 方法，使得用户只需通过 CLI 参数 `--coast-retention-ratio 0.222` 启用 COAST，patches 即在配置初始化阶段自动应用：

```python
# vllm/config/vllm.py
def _apply_coast_patches(self):
    if self.model_config.multimodal_config.is_coast_enabled():
        from coast_vllm import apply_patches
        apply_patches()
```

调用时机在 `try_verify_and_update_config()` 之后、模型加载之前。

## 三、实现方案

### 3.1 文件结构与改动明细

```
vllm/                                    ← vllm 源码（约 95 行新增）
├── multimodal/coast.py                   ← COAST 核心算法 (~200 行新增)
├── config/multimodal.py                  ← 5 配置字段 + is_coast_enabled() (~40 行新增)
├── config/model.py                       ← 6 InitVar + __post_init__ + mm_config_kwargs (~24 行新增)
├── config/vllm.py                       ← _apply_coast_patches() (~15 行新增)
├── engine/arg_utils.py                  ← EngineArgs + CLI args + create_model_config (~26 行新增)

coast_vllm/                               ← 独立适配包（不修改 vllm 源码）
├── __init__.py                           ← 导出 apply_patches
├── example.py                            ← 使用示例
├── patches/
│   ├── __init__.py                       ← 导出各 patch 函数
│   ├── apply.py                          ← apply_patches() / unpatch_all()
│   ├── llava_patch.py                    ← LLaVA processor + model patches
│   ├── qwen2_5_vl_patch.py               ← Qwen2.5-VL processor + model patches
│   ├── qwen3_vl_patch.py                 ← Qwen3-VL processor + model patches
│   └── layer_pruning_patch.py            ← 中间层剪枝 patches (Llama/Qwen2)
```

### 3.2 核心算法 (`vllm/multimodal/coast.py`)

详见文件内容，核心函数：

- `compute_coast_retained_tokens_count(original_tokens, retention_ratio)` — 计算保留 token 数
- `coast_prune_visual_tokens(embeddings, num_retain, alpha_min, alpha_max, anchor_ratio)` — 编码器输出层剪枝
- `coast_prune_at_layer(hidden_states, is_visual, num_retain, ...)` — 中间层剪枝（合并模式）

### 3.3 配置管线（6 层参数传递）

```
CLI arg                     EngineArgs field            ModelConfig InitVar         MultiModalConfig field
--coast-retention-ratio  →  coast_retention_ratio      → coast_retention_ratio      → coast_retention_ratio
--coast-alpha-min        →  coast_alpha_min            → coast_alpha_min            → coast_alpha_min
--coast-alpha-max        →  coast_alpha_max            → coast_alpha_max            → coast_alpha_max
--coast-anchor-ratio     →  coast_anchor_ratio         → coast_anchor_ratio         → coast_anchor_ratio
--coast-layer            →  coast_layer                → coast_layer                → coast_layer
```

### 3.4 Processor Patch — PlaceholderRange 更新

在预处理阶段，monkey-patch 各模型的 `_get_prompt_updates()` 方法，将 `num_tokens` 从原始值替换为 COAST 剪枝后的数量：

```python
# llava_patch.py — processor patch
def get_replacement(item_idx: int):
    original_tokens = ...
    num_tokens = compute_coast_retained_tokens_count(original_tokens, retention_ratio)
    return [image_token_id] * num_tokens
```

### 3.5 Model Patch — Encoder 级剪枝

Monkey-patch 模型的 `_process_image_input()` / `_process_video_input()`，在视觉编码器输出后应用 COAST：

```python
# llava_patch.py — model patch
def coast_process_image_input(self, image_input):
    image_embeds = original_process_image_input(self, image_input)
    if not coast_config.is_coast_enabled():
        return image_embeds
    num_retain = compute_coast_retained_tokens_count(original_tokens, retention_ratio)
    compressed, _ = coast_prune_visual_tokens(
        image_embeds, num_retain, alpha_min, alpha_max, anchor_ratio
    )
    return compressed
```

### 3.6 Layer Patch — 中间层剪枝（可选）

Monkey-patch `LlamaModel.forward()` / `Qwen2Model.forward()`，在指定层迭代中插入 COAST 剪枝：

```python
for idx, layer in enumerate(self.layers):
    hidden_states, residual = layer(positions, hidden_states, residual)
    if (actual_start + idx) == coast_layer:
        is_visual = _identify_visual_tokens(input_ids, self)
        true_hidden = hidden_states + residual
        true_hidden = coast_prune_at_layer(
            true_hidden, is_visual, num_retain, alpha_min, alpha_max, anchor_ratio
        )
        hidden_states = true_hidden - residual
```

剪枝 token 的信息合并到最近保留 token，序列长度不变。

## 四、使用方式

### 4.1 服务端启动（推荐）

COAST 已内置自动 hook，直接使用 `vllm serve` 即可：

```bash
vllm serve llava-hf/llava-1.5-7b-hf \
  --port 8086 \
  --dtype bfloat16 \
  --coast-retention-ratio 0.222 \
  --coast-alpha-min 0.05 \
  --coast-alpha-max 0.15 \
  --coast-anchor-ratio 0.8
```

### 4.2 LLM 类方式

```python
from vllm import LLM

llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    coast_retention_ratio=0.222,
    coast_alpha_min=0.05,
    coast_alpha_max=0.15,
    coast_anchor_ratio=0.8,
)
```

### 4.3 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--coast-retention-ratio` | float | None | 保留比例 (0,1]。0.222=保留 22.2% |
| `--coast-alpha-min` | float | 0.05 | α_min: 互补上下文最小比例 |
| `--coast-alpha-max` | float | 0.15 | α_max: 互补上下文最大比例 |
| `--coast-anchor-ratio` | float | 0.8 | 锚点占总预算比例 |
| `--coast-layer` | int | None | 中间层剪枝位置。LLaVA=2, Qwen=14 |

### 4.4 与 EVS 联合使用

COAST可与 EVS 视频剪枝联合使用（对视频先 EVS 剪枝，再 COAST 剋枝）：

```bash
vllm serve Qwen2.5-VL-7B-Instruct \
  --video-pruning-rate 0.5 \
  --coast-retention-ratio 0.222 \
  --coast-alpha-min 0.05 \
  --coast-alpha-max 0.15
```

## 五、版本迁移指南

升级 vLLM 到新版本时，需完成以下步骤：

1. **重新添加 `vllm/multimodal/coast.py`** — 算法文件，与 vLLM 版本无关，直接复制即可。
2. **重新添加配置字段** — 在 `multimodal.py`、`model.py`、`arg_utils.py`、`vllm.py` 中各添加数行（模板化操作，参照 EVS 的对应位置）。
3. **更新 monkey-patches** — 检查模型类的方法签名是否有变化，若变化则更新 `coast_vllm/patches/` 中对应的 patch 文件。
4. **COAST 算法无需修改** — 纯数学计算，与 vLLM 版本无关。

总迁移工作量预计 < 1 小时，且改动均为新增行而非重构。

## 六、与 LRCP 的对比

| 方面 | LRCP | COAST |
|------|------|-------|
| 论文 | arxiv 2605.15621 | arxiv 2605.09429 |
| 核心机制 | PCA 投影残差评分 | 熵驱动预算 + 对比语义路由 |
| 信号来源 | 视觉 token 低秩结构 | 跨模态相似度分布 |
| 保留策略 | Top-K（残差最大） | 两尾保留（锚点+互补上下文） |
| 自适应能力 | 固定保留比例 | 熵驱动动态预算分配 |
| 上下文保护 | Token 合并 | 两尾保留 + 合并 |
| 中间层支持 | 有（PCA 剪枝） | 有（跨模态对比路由） |
| Visual Aphasia | 未专门应对 | 核心贡献：避免 Visual Aphasia |

## 七、总结

本方案将 COAST 论文方法以非侵入式方式适配到 vLLM，核心特点：

- **vllm 源码改动极小**：约 95 行新增（0 行删除），涉及 5 个文件，全部为配置注册与自动 hook
- **COAST 算法完全独立**：位于 `vllm/multimodal/coast.py`，纯数学计算，无模型依赖
- **适配逻辑独立包**：`coast_vllm/` 通过 monkey-patch 实现，不修改 vllm 源码
- **自动生效**：`--coast-retention-ratio` 参数传入后，patches 在 `VllmConfig.__post_init__` 中自动应用
- **支持三大模型族**：LLaVA、Qwen2.5-VL、Qwen3-VL
- **M-RoPE 兼容**：借助 vLLM 现有的 `recompute_mrope_positions()` 机制
- **中间层剪枝可选**：通过 `--coast-layer` 参数在 LLM 中间层追加对比语义路由
- **两尾保留机制**：同时保留锚点对齐证据和互补空间上下文，避免 Visual Aphasia
