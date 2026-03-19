# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class FASACalibrationResult:
    dominant_fc_indices: dict[str, dict[str, list[int]]]
    mean_ca_scores: dict[str, dict[str, list[float]]]
    topk_k: int
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dominant_fc_indices": self.dominant_fc_indices,
            "mean_ca_scores": self.mean_ca_scores,
            "topk_k": self.topk_k,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FASACalibrationResult":
        return cls(
            dominant_fc_indices=payload["dominant_fc_indices"],
            mean_ca_scores=payload["mean_ca_scores"],
            topk_k=payload["topk_k"],
            metadata=payload.get("metadata", {}),
        )


def _validate_even_head_dim(head_dim: int) -> None:
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head dimension must be even, got {head_dim}.")


def split_frequency_chunks(x: torch.Tensor) -> torch.Tensor:
    """Split the last dimension into RoPE frequency chunks of size 2.

    Input shape: [..., head_dim]
    Output shape: [..., num_fc, 2]
    """
    _validate_even_head_dim(x.shape[-1])
    return x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)


def rope_frequency_bases(num_fc: int, head_dim: int, base: float = 10000.0) -> torch.Tensor:
    idx = torch.arange(num_fc, dtype=torch.float32)
    return base ** (-2.0 * idx / head_dim)


def rotate_frequency_chunks(
    chunks: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
) -> torch.Tensor:
    """Apply RoPE to chunked vectors.

    chunks: [..., num_fc, 2]
    positions: shape matching chunks.shape[:-2] or broadcastable to it.
    returns: same shape as chunks
    """
    num_fc = chunks.shape[-2]
    head_dim = num_fc * 2
    theta = rope_frequency_bases(num_fc, head_dim, base=base).to(chunks.device, chunks.dtype)
    angles = positions.to(chunks.device, chunks.dtype)
    while angles.ndim < chunks.ndim - 1:
        angles = angles.unsqueeze(-1)
    angles = angles * theta
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x0 = chunks[..., 0]
    x1 = chunks[..., 1]
    rotated0 = x0 * cos - x1 * sin
    rotated1 = x0 * sin + x1 * cos
    return torch.stack((rotated0, rotated1), dim=-1)


def full_attention_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    query_position: int,
    key_positions: torch.Tensor,
    rope_base: float = 10000.0,
) -> torch.Tensor:
    q_chunks = split_frequency_chunks(query)
    k_chunks = split_frequency_chunks(keys)
    q_rot = rotate_frequency_chunks(
        q_chunks.unsqueeze(0),
        torch.tensor([query_position], device=query.device),
        rope_base,
    )[0]
    k_rot = rotate_frequency_chunks(k_chunks, key_positions.to(query.device), rope_base)
    return (q_rot.unsqueeze(0) * k_rot).sum(dim=(-1, -2))


def single_fc_attention_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    query_position: int,
    key_positions: torch.Tensor,
    fc_index: int,
    rope_base: float = 10000.0,
) -> torch.Tensor:
    q_chunks = split_frequency_chunks(query)[fc_index]
    k_chunks = split_frequency_chunks(keys)[:, fc_index]
    q_rot = rotate_frequency_chunks(q_chunks.unsqueeze(0), torch.tensor([query_position], device=query.device), rope_base)[0]
    k_rot = rotate_frequency_chunks(k_chunks, key_positions.to(query.device), rope_base)
    return (q_rot * k_rot).sum(dim=-1)


def contextual_agreement(
    full_scores: torch.Tensor,
    single_fc_scores: torch.Tensor,
    topk_k: int,
) -> float:
    k = min(topk_k, full_scores.numel(), single_fc_scores.numel())
    if k <= 0:
        return 0.0
    full_idx = torch.topk(full_scores, k=k).indices
    fc_idx = torch.topk(single_fc_scores, k=k).indices
    inter = torch.isin(full_idx, fc_idx).sum().item()
    return inter / k


@torch.no_grad()
def offline_calibration(
    calibration_examples: list[dict[str, torch.Tensor]],
    num_layers: int,
    num_heads: int,
    topk_k: int,
    num_dominant_fcs: int,
    rope_base: float = 10000.0,
) -> FASACalibrationResult:
    """Algorithm 1 from FASA in a minimal tensorized form.

    Each calibration example contains:
      queries: [num_steps, num_layers, num_heads, head_dim]
      keys:    [num_steps, num_layers, num_heads, head_dim]

    At step t, query[t] attends to keys[:t+1].
    """
    score_buckets: dict[tuple[int, int, int], list[float]] = {}

    for example in calibration_examples:
        queries = example["queries"]
        keys = example["keys"]
        num_steps = queries.shape[0]
        head_dim = queries.shape[-1]
        num_fc = head_dim // 2
        positions = torch.arange(num_steps, device=queries.device)

        for t in range(num_steps):
            ctx_positions = positions[: t + 1]
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    q = queries[t, layer_idx, head_idx]
                    k = keys[: t + 1, layer_idx, head_idx]
                    full_scores = full_attention_scores(
                        q, k, t, ctx_positions, rope_base=rope_base
                    )
                    for fc_idx in range(num_fc):
                        fc_scores = single_fc_attention_scores(
                            q, k, t, ctx_positions, fc_idx, rope_base=rope_base
                        )
                        ca = contextual_agreement(full_scores, fc_scores, topk_k)
                        score_buckets.setdefault((layer_idx, head_idx, fc_idx), []).append(ca)

    mean_scores = torch.zeros(num_layers, num_heads, head_dim // 2, dtype=torch.float32)
    dominant_fc_indices: dict[str, dict[str, list[int]]] = {}
    mean_ca_scores: dict[str, dict[str, list[float]]] = {}
    for layer_idx in range(num_layers):
        dominant_fc_indices[str(layer_idx)] = {}
        mean_ca_scores[str(layer_idx)] = {}
        for head_idx in range(num_heads):
            per_fc_scores = []
            for fc_idx in range(head_dim // 2):
                values = score_buckets.get((layer_idx, head_idx, fc_idx), [0.0])
                score = float(sum(values) / len(values))
                mean_scores[layer_idx, head_idx, fc_idx] = score
                per_fc_scores.append(score)
            topk = torch.topk(mean_scores[layer_idx, head_idx], k=min(num_dominant_fcs, head_dim // 2)).indices
            dominant_fc_indices[str(layer_idx)][str(head_idx)] = topk.tolist()
            mean_ca_scores[str(layer_idx)][str(head_idx)] = per_fc_scores

    return FASACalibrationResult(
        dominant_fc_indices=dominant_fc_indices,
        mean_ca_scores=mean_ca_scores,
        topk_k=topk_k,
        metadata={
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "num_frequency_chunks": head_dim // 2,
            "rope_base": rope_base,
        },
    )


def tip_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    query_position: int,
    key_positions: torch.Tensor,
    dominant_fc_indices: list[int],
    rope_base: float = 10000.0,
) -> torch.Tensor:
    scores = []
    for fc_idx in dominant_fc_indices:
        scores.append(
            single_fc_attention_scores(
                query, keys, query_position, key_positions, fc_idx, rope_base=rope_base
            )
        )
    if not scores:
        return torch.zeros(keys.shape[0], dtype=query.dtype, device=query.device)
    return torch.stack(scores, dim=0).sum(dim=0)


def select_top_tokens(scores: torch.Tensor, num_tokens: int) -> torch.Tensor:
    k = min(num_tokens, scores.numel())
    return torch.topk(scores, k=k).indices


def fac_mask_mod_factory(selected_token_indices: torch.Tensor):
    selected = selected_token_indices.to(torch.long)

    def fac_mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        del b, h, q_idx
        return torch.isin(kv_idx, selected)

    return fac_mask_mod


def focused_attention_output(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    selected_token_indices: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    if selected_token_indices.numel() == 0:
        return torch.zeros_like(query)
    k_sel = keys.index_select(0, selected_token_indices)
    v_sel = values.index_select(0, selected_token_indices)
    scale = scale or (query.shape[-1] ** -0.5)
    logits = torch.einsum("hd,thd->ht", query, k_sel) * scale
    probs = torch.softmax(logits, dim=-1)
    return torch.einsum("ht,thd->hd", probs, v_sel)


def save_calibration_result(result: FASACalibrationResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def load_calibration_result(path: str | Path) -> FASACalibrationResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FASACalibrationResult.from_dict(payload)
