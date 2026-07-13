# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from vllm.autogaze.core import build_autogaze_attention_mask


def _qwen_vision_attention(
    attention,
    hidden_states: torch.Tensor,
    *,
    rotary_pos_emb_cos: torch.Tensor,
    rotary_pos_emb_sin: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run a Qwen vision attention module with an explicit AutoGaze mask.

    vLLM's normal vision attention path uses packed independent sequences.
    AutoGaze needs block-causal interaction between frames, so this isolated
    path reuses the pretrained QKV/output projections while delegating the
    masked attention operation to PyTorch SDPA.
    """
    qkv, _ = attention.qkv(hidden_states)
    seq_len, batch_size, _ = qkv.shape
    if batch_size != 1:
        raise ValueError("AutoGaze Qwen vision adapter expects batch size 1")

    qkv = qkv.view(
        seq_len,
        batch_size,
        3,
        attention.num_attention_heads_per_partition,
        attention.hidden_size_per_attention_head,
    ).permute(1, 0, 2, 3, 4)

    qk = qkv[:, :, :2]
    value = qkv[:, :, 2]
    qk_reshaped = qk.permute(2, 0, 1, 3, 4).reshape(
        2 * batch_size,
        seq_len,
        attention.num_attention_heads_per_partition,
        attention.hidden_size_per_attention_head,
    )
    qk_rotated = attention.apply_rotary_emb(
        qk_reshaped.contiguous(),
        rotary_pos_emb_cos,
        rotary_pos_emb_sin,
    ).view(
        2,
        batch_size,
        seq_len,
        attention.num_attention_heads_per_partition,
        attention.hidden_size_per_attention_head,
    )
    query, key = qk_rotated.unbind(dim=0)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    context = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask.view(1, 1, seq_len, seq_len),
        dropout_p=0.0,
        is_causal=False,
        scale=attention.hidden_size_per_attention_head**-0.5,
    )
    context = context.transpose(1, 2).reshape(seq_len, batch_size, -1)
    output, _ = attention.proj(context.contiguous())
    return output


def run_qwen3_5_autogaze_vision(
    visual,
    pixel_values: torch.Tensor,
    *,
    full_grid_thw: Sequence[Sequence[int]],
    gazing_pos: torch.Tensor,
    input_tokens_per_frame: Sequence[int],
    attention_type: str,
    frame_independent_encoding: bool,
) -> torch.Tensor:
    """Encode only AutoGaze-selected patches with a Qwen3.5 vision tower."""
    if gazing_pos.numel() == 0:
        raise ValueError("AutoGaze selected no patches for this video")
    if sum(int(length) for length in input_tokens_per_frame) != gazing_pos.numel():
        raise ValueError(
            "The selected patch count must equal sum(input_tokens_per_frame)"
        )

    pixel_values = pixel_values.to(
        device=visual.device,
        dtype=visual.dtype,
        non_blocking=True,
    )
    gazing_pos = gazing_pos.to(device=visual.device, dtype=torch.long)

    # Positional embeddings are cheap to construct for the full multi-scale
    # layout. The expensive patch projection and transformer blocks only see
    # the selected rows.
    pos_embeds = visual.fast_pos_embed_interpolate(
        [list(grid) for grid in full_grid_thw]
    )[gazing_pos]
    rotary_cos, rotary_sin = visual.rot_pos_emb([list(grid) for grid in full_grid_thw])
    rotary_cos = rotary_cos[gazing_pos]
    rotary_sin = rotary_sin[gazing_pos]

    hidden_states = visual.patch_embed(pixel_values[gazing_pos])
    hidden_states = (hidden_states + pos_embeds).unsqueeze(1)
    attention_mask = build_autogaze_attention_mask(
        input_tokens_per_frame,
        attention_type=attention_type,
        frame_independent_encoding=frame_independent_encoding,
        device=visual.device,
    )

    deepstack_feature_lists = []
    for layer_num, block in enumerate(visual.blocks):
        attention_output = _qwen_vision_attention(
            block.attn,
            block.norm1(hidden_states),
            rotary_pos_emb_cos=rotary_cos,
            rotary_pos_emb_sin=rotary_sin,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))

        if layer_num in visual.deepstack_visual_indexes:
            merger_idx = visual.deepstack_visual_indexes.index(layer_num)
            deepstack_feature_lists.append(
                visual.deepstack_merger_list[merger_idx](hidden_states)
            )

    hidden_states = visual.merger(hidden_states)
    return torch.cat([hidden_states] + deepstack_feature_lists, dim=1)
