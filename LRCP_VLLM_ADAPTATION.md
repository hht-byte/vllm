# LRCP 非侵入式适配 vLLM v0.19.1

## 一、背景与问题

### 1.1 问题概述

大型视觉语言模型（LVLMs）在处理高分辨率图像和长视频时，视觉 token 数量急剧增长，导致 prefill 阶段计算与显存开销大幅增加、推理延迟上升。以 LLaVA-v1.5-7B 为例，单张 336×336 图像产生 576 个视觉 token；LLaVA-NeXT-7B 多裁剪编码可达 2880 个；Video-LLaVA-7B 的 8 帧视频产生 2048 个视觉 token。视觉 token 已成为 prefill 阶段的主要瓶颈。

### 1.2 LRCP 方法

LRCP（Low-Rank Compressibility Guided Pruning，论文 https://arxiv.org/abs/2605.15621）是一种 **免训练（training-free）** 的视觉 token 剪枝框架，其核心发现与思路：

1. **低秩结构**：视觉 token 表征呈现显著低秩结构，存在一个稳定的主导子空间（dominant subspace），即使随机丢弃 80% 的 token，该子空间仍高度稳定。
2. **投影残差评分**：通过 PCA 估计主导子空间后，每个 token 的投影残差 `s_i = ||x_i(I - P_r)||^2` 可作为 token 重要性的代理指标——残差越大意味着该 token 携带的判别性信息越多，越不应被丢弃。
3. **Token 合并**：丢弃的 token 通过余弦相似度分配到最近的保留 token，加权平均以减少信息损失。

论文实验表明：在 LLaVA-v1.5-7B 上保留 64 个 token（88.9% 剪枝率）可保留 94.7% 原始性能；在 Video-LLaVA-7B 上 87.5% 剪枝率可保留 97.8% 性能。

### 1.3 适配挑战

将 LRCP 适配到 vLLM 需要解决以下问题：

- **PlaceholderRange 同步**：vLLM 在预处理阶段为每个多模态数据项创建 `PlaceholderRange(offset, length)` 来定位视觉 token 在序列中的位置。剪枝后 token 数减少，`PlaceholderRange.length` 必须与实际保留数一致，否则 `_gather_mm_embeddings` 和 `embed_input_ids` 会出错。
- **M-RoPE 位置重算**：Qwen2.5-VL/Qwen3-VL 使用多维旋转位置编码（M-RoPE），剪枝后位置编码需要重新计算。
- **模型多样性**：vLLM 支持 82+ 多模态模型，各模型的 `embed_multimodal()` 流程各不相同，逐个修改模型文件既繁杂又侵入性强。
- **版本迁移**：vLLM 版本迭代频繁，侵入式修改难以跟随升级。

## 二、关键设计

### 2.1 非侵入式原则

核心设计目标是 **非侵入式（方便迁移到 vLLM 新版本）**，具体体现为：

| 层级 | 内容 | 侵入程度 |
|------|------|---------|
| **算法层** | `vllm/multimodal/lrcp.py` — 纯算法模块，不涉及模型细节 | 新增文件，0 行修改 |
| **配置层** | `vllm/config/multimodal.py` — 4 个配置字段 + `is_lrcp_enabled()` | 新增 31 行 |
| **管线层** | `arg_utils.py` + `model.py` — CLI 参数注册与传递 | 新增 34 行 |
| **初始化层** | `vllm/config/vllm.py` — `__post_init__` 自动检测与应用 patches | 新增 18 行 |
| **适配层** | `lrcp_vllm/` — 全部 monkey-patch，独立包，不修改 vllm 源码 | 0 行修改 vllm |

**vllm 源码总修改量**：仅 83 行新增，0 行删除。迁移到新版本时只需重新添加这些新增行。

### 2.2 三层剪枝架构

LRCP 可作用于三个层级，对应论文中的两种剪枝位置：

```
┌─────────────────────────────────────────┐
│  1. Encoder Output Level                 │  ← 视觉编码器输出后
│     patch: _process_image_input()        │
│     patch: _process_video_input()        │
│     最主要、效果最佳的剪枝位置            │
├─────────────────────────────────────────┤
│  2. Processor Level                      │  ← 预处理阶段
│     patch: _get_prompt_updates()         │
│     更新 PlaceholderRange.length         │
├─────────────────────────────────────────┤
│  3. Intermediate LLM Layer               │  ← LLM 中间层 (可选)
│     patch: LlamaModel.forward()          │
│     patch: Qwen2Model.forward()          │
│     在指定层 (如 layer 16) 再次剪枝       │
└─────────────────────────────────────────┤
```

### 2.3 与 EVS 的类比设计

LRCP 的适配模式参照 vLLM 已有的 EVS（Efficient Video Sampling）集成方式：

| 方面 | EVS | LRCP |
|------|-----|------|
| 算法模块位置 | `vllm/multimodal/evs.py` | `vllm/multimodal/lrcp.py` |
| 配置字段 | `video_pruning_rate` | `lrcp_retention_ratio` + 3 个 |
| Processor 修改 | 计算 `compute_retained_tokens_count` | 计算 `compute_lrcp_retained_tokens_count` |
| Encoder 修改 | `_postprocess_video_embeds_evs()` | monkey-patch `_process_image_input()` |
| M-RoPE 处理 | 附带 4/5 通道位置信息 | 同样附带位置通道 |
| 适用范围 | 仅视频 | 图像 + 视频 |

### 2.4 自动 Patch 机制

在 `VllmConfig.__post_init__()` 中添加 `_apply_lrcp_patches()` 方法，使得用户只需通过 CLI 参数 `--lrcp-retention-ratio 0.111` 启用 LRCP，patches 即在配置初始化阶段自动应用，无需手动调用 wrapper：

```python
# vllm/config/vllm.py
def _apply_lrcp_patches(self):
    if self.model_config.multimodal_config.is_lrcp_enabled():
        from lrcp_vllm import apply_patches
        apply_patches()
```

调用时机在 `try_verify_and_update_config()` 之后、模型加载之前，确保 patches 在所有进程（API server、engine core、worker）中生效。

## 三、实现方案

### 3.1 文件结构与改动明细

```
vllm/                                    ← vllm 源码（83 行新增）
├── multimodal/lrcp.py                   ← LRCP 核心算法 (146 行新增)
├── config/multimodal.py                 ← 4 配置字段 + is_lrcp_enabled() (31 行新增)
├── config/model.py                      ← 4 InitVar + __post_init__ + mm_config_kwargs (12 行新增)
├── config/vllm.py                       ← _apply_lrcp_patches() 自动 hook (18 行新增)
├── engine/arg_utils.py                  ← EngineArgs + CLI args + create_model_config (22 行新增)

lrcp_vllm/                               ← 独立适配包（不修改 vllm 源码）
├── __init__.py                           ← 导出 apply_patches
├── example.py                            ← 使用示例
├── run_server.py                         ← 服务端启动 wrapper (可选)
├── patches/
│   ├── __init__.py                       ← 导出各 patch 函数
│   ├── apply.py                          ← apply_patches() / unpatch_all()
│   ├── llava_patch.py                    ← LLaVA processor + model patches
│   ├── qwen2_5_vl_patch.py               ← Qwen2.5-VL processor + model patches
│   ├── qwen3_vl_patch.py                 ← Qwen3-VL processor + model patches
│   └── layer_pruning_patch.py            ← 中间层剪枝 patches (Llama/Qwen2)
```

### 3.2 核心算法 (`vllm/multimodal/lrcp.py`)

LRCP 算法分四步实现：

```python
def lrcp_compress(embeddings, num_retain, subspace_dim=4, merge=True):
    # 1. PCA: 估计主导低秩子空间
    centered = embeddings - mean
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    U_r = Vh[:subspace_dim, :].T          # (D, r) 主成分方向

    # 2. 投影残差评分: s_i = ||x_i(I - P_r)||^2
    proj = centered @ U_r                  # (N, r) 子空间投影
    residual = centered - proj @ U_r.T     # (N, D) 残差分量
    scores = (residual ** 2).sum(dim=-1)   # (N,) 投影残差分数

    # 3. 保留残差最大的 K 个 token
    _, top_indices = torch.topk(scores, k=num_retain, largest=True)
    retention_mask[top_indices] = True
    retained = embeddings[retention_mask]  # (K, D)

    # 4. Token 合并: 丢弃 token 按余弦相似度归入最近保留 token，加权平均
    cos_sim = F.cosine_similarity(discarded.unsqueeze(1), retained.unsqueeze(0))
    nearest = cos_sim.argmax(dim=-1)
    retained = (retained + sums) / (1 + counts.unsqueeze(-1))
```

### 3.3 配置管线 (4 层参数传递)

LRCP CLI 参数需通过 vLLM 的 4 层参数管线传递：

```
CLI arg                    EngineArgs field            ModelConfig InitVar         MultiModalConfig field
--lrcp-retention-ratio → lrcp_retention_ratio       → lrcp_retention_ratio      → lrcp_retention_ratio
--lrcp-subspace-dim   → lrcp_subspace_dim           → lrcp_subspace_dim         → lrcp_subspace_dim
--lrcp-merge          → lrcp_merge                  → lrcp_merge                → lrcp_merge
--lrcp-layer          → lrcp_layer                   → lrcp_layer                → lrcp_layer
```

每层需手动注册（`get_kwargs(MultiModalConfig)` 自动生成 argparse kwargs，只需调用 `add_argument`）。

### 3.4 Processor Patch — PlaceholderRange 更新

在预处理阶段，monkey-patch 各模型的 `_get_prompt_updates()` 方法，将 `num_tokens` 从原始值替换为 LRCP 剪枝后的数量：

```python
# llava_patch.py — processor patch
def get_replacement(item_idx):
    original_tokens = self.info.get_num_image_tokens(...)
    num_tokens = compute_lrcp_retained_tokens_count(original_tokens, retention_ratio)
    return [image_token_id] * num_tokens   # PlaceholderRange.length = num_tokens
```

**关键**：`PlaceholderRange` 在创建时即使用剪枝后的 token 数，后续 `_gather_mm_embeddings` 和 `embed_input_ids` 自然匹配，无需额外更新。

### 3.5 Model Patch — Encoder 级剪枝

Monkey-patch 模型的 `_process_image_input()` / `_process_video_input()`，在视觉编码器输出后应用 LRCP：

```python
# llava_patch.py — model patch
def lrcp_process_image_input(self, image_input):
    image_embeds = original_process_image_input(self, image_input)  # 原始编码器输出
    if not lrcp_config.is_lrcp_enabled():
        return image_embeds
    num_retain = compute_lrcp_retained_tokens_count(original_tokens, retention_ratio)
    compressed, _ = lrcp_compress(image_embeds, num_retain, subspace_dim, merge)
    return compressed
```

对于 M-RoPE 模型（Qwen2.5-VL），剪枝后还需附带位置通道：

```python
# qwen2_5_vl_patch.py — M-RoPE 位置处理
positions = compute_mrope_for_media(thw, merge_size).to(emb.device)
positions = positions[:original_tokens][retention_mask]  # 按保留 mask 过滤位置
compressed = torch.cat([compressed, positions], dim=1)   # 附带 4 通道位置信息
```

位置重算由 vLLM 现有的 `recompute_mrope_positions()` 在 `_gather_mm_embeddings` 中处理，无需额外修改。

### 3.6 Layer Patch — 中间层剪枝（可选）

Monkey-patch `LlamaModel.forward()` / `Qwen2Model.forward()`，在指定层迭代中插入 LRCP 剪枝：

```python
for idx, layer in enumerate(self.layers):
    hidden_states, residual = layer(positions, hidden_states, residual)
    if (actual_start + idx) == lrcp_layer:  # 在指定层应用 LRCP
        hidden_states, residual = _apply_layer_lrcp(hidden_states, residual, input_ids, self)
```

通过 `_get_lrcp_layer_config()` 从 `ForwardContext` 获取配置，识别多模态 token 位置后仅对这些 token 执行剪枝。

### 3.7 自动 Patch Hook

在 `VllmConfig.__post_init__()` 中，紧接 `try_verify_and_update_config()` 之后调用 `_apply_lrcp_patches()`：

```python
def __post_init__(self):
    self.try_verify_and_update_config()
    self._apply_lrcp_patches()       # ← 自动检测 LRCP 配置并应用 patches
    ...

def _apply_lrcp_patches(self):
    if self.model_config.multimodal_config.is_lrcp_enabled():
        from lrcp_vllm import apply_patches
        apply_patches()               # monkey-patch 所有模型类方法
```

此 hook 在所有进程中运行（API server、engine core、worker），且在模型加载之前，确保 patches 生效时机正确。

## 四、使用方式

### 4.1 服务端启动（推荐）

LRCP 已内置自动 hook，直接使用 `vllm serve` 即可：

```bash
vllm serve /softwarePlatform/models/Qwen3-VL-4B-sft-v002 \
  --port 8086 \
  --served-model-name Qwen3-VL-4B \
  --allowed-local-media-path / \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 2816 \
  --block-size 128 \
  --max-num-batched-tokens 36988 \
  --gpu-memory-utilization 0.9 \
  --lrcp-retention-ratio 0.111 \
  --lrcp-subspace-dim 4 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

### 4.2 LLM 类方式

```python
from vllm import LLM

llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    lrcp_retention_ratio=0.111,
    lrcp_subspace_dim=4,
    lrcp_merge=True,
)
```

### 4.3 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lrcp-retention-ratio` | float | None | 保留比例 (0,1]。0.111=保留 11.1% token |
| `--lrcp-subspace-dim` | int | 4 | PCA 子空间维度 r。LLaVA=4，Qwen=8 |
| `--lrcp-merge` | bool | True | 是否合并丢弃 token 到最近保留 token |
| `--lrcp-layer` | int | None | 中间层剪枝位置。LLaVA=16，Qwen=14 |

### 4.4 Wrapper 启动（备选）

如果自动 hook 未生效，可使用 wrapper：

```bash
python lrcp_vllm/run_server.py \
  --model ... --lrcp-retention-ratio 0.111 ...
```

## 五、版本迁移指南

升级 vLLM 到新版本时，需完成以下步骤：

1. **重新添加 `vllm/multimodal/lrcp.py`** — 算法文件，与 vLLM 版本无关，直接复制即可。
2. **重新添加配置字段** — 在 `multimodal.py`、`model.py`、`arg_utils.py`、`vllm.py` 中各添加数行（模板化操作，参照 EVS 的对应位置）。
3. **更新 monkey-patches** — 检查模型类的方法签名是否有变化，若变化则更新 `lrcp_vllm/patches/` 中对应的 patch 文件。
4. **LRCP 算法无需修改** — 纯数学计算，与 vLLM 版本无关。

总迁移工作量预计 < 1 小时，且改动均为新增行而非重构。

## 六、总结

本方案将 LRCP 论文方法以非侵入式方式适配到 vLLM v0.19.1，核心特点：

- **vllm 源码改动极小**：仅 83 行新增（0 行删除），涉及 5 个文件，全部为配置注册与自动 hook
- **LRCP 算法完全独立**：位于 `vllm/multimodal/lrcp.py`，纯数学计算，无模型依赖
- **适配逻辑独立包**：`lrcp_vllm/` 通过 monkey-patch 实现，不修改 vllm 源码
- **自动生效**：`--lrcp-retention-ratio` 参数传入后，patches 在 `VllmConfig.__post_init__` 中自动应用
- **支持三大模型族**：LLaVA、Qwen2.5-VL、Qwen3-VL，覆盖固定分辨率与动态分辨率编码
- **M-RoPE 兼容**：借助 vLLM 现有的 `recompute_mrope_positions()` 机制，无需额外处理
- **中间层剪枝可选**：通过 `--lrcp-layer` 参数在 LLM 中间层追加剪枝，进一步增强压缩效果
