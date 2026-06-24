# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# LRCP: Low-Rank Compressibility Guided Visual Token Pruning
# Reference: https://arxiv.org/abs/2605.15621

import torch
import torch.nn.functional as F


def compute_lrcp_retained_tokens_count(
    num_tokens: int,
    retention_ratio: float,
) -> int:
    """Compute the number of tokens to retain after LRCP pruning.

    Args:
        num_tokens: Original number of visual tokens.
        retention_ratio: Fraction of tokens to retain, in range (0, 1].

    Returns:
        Number of tokens to retain (at least 1).
    """
    return max(1, round(num_tokens * retention_ratio))


def bool_mask_to_indices(mask: torch.Tensor) -> torch.Tensor:
    """Convert a boolean mask to integer indices without using nonzero on device.

    NPU/Ascend does not support aclnnNonzeroV2 for boolean tensors.
    This function transfers the mask to CPU, runs nonzero there,
    and transfers the resulting indices back to the original device.

    Args:
        mask: Boolean tensor of shape (N,).

    Returns:
        Long tensor of indices where mask is True, on the same device.
    """
    return mask.cpu().nonzero().squeeze(-1).to(mask.device)


def lrcp_compress(
    embeddings: torch.Tensor,
    num_retain: int,
    subspace_dim: int = 4,
    merge: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LRCP: Low-Rank Compressibility Guided Visual Token Pruning.

    Applies PCA-based token pruning:
    1. Estimate the dominant low-rank subspace via PCA
    2. Score each token by projection residual onto this subspace
    3. Retain tokens with the highest residuals (most discriminative)
    4. Optionally merge discarded tokens into nearest retained neighbors

    All PCA, SVD, and merge computations are performed on CPU to
    avoid NPU/Ascend kernel compatibility issues (aclnnNonzeroV2,
    SVD memory allocation failures on AICPU). For typical visual
    token matrices (N < 3000, D < 4096), CPU computation overhead
    is negligible (< 1ms). Results are transferred back to the
    original device and dtype.

    Args:
        embeddings: Visual token embeddings of shape (N, D).
        num_retain: Number of tokens to retain (K).
        subspace_dim: PCA subspace dimension r. Controls the boundary
            between shared structure and discriminative residuals.
            Default 4 for LLaVA models, 8 for Qwen models.
        merge: Whether to merge discarded tokens into nearest retained
            neighbors via cosine similarity. Reduces information loss.

    Returns:
        Tuple of:
        - compressed_embeddings: (K, D) retained (and optionally merged)
          token embeddings, in the original dtype and device
        - top_indices: (K,) long tensor of retained token indices,
          on the original device
    """
    N, D = embeddings.shape
    orig_dtype = embeddings.dtype
    orig_device = embeddings.device

    if num_retain >= N:
        return embeddings, torch.arange(N, device=orig_device)

    # Move to CPU for all PCA/SVD/merge computation.
    # NPU/Ascend AICPU SVD kernel has memory and compatibility issues.
    emb_cpu = embeddings.float().cpu()

    mean = emb_cpu.mean(dim=0, keepdim=True)
    centered = emb_cpu - mean

    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    U_r = Vh[:subspace_dim, :].T

    proj = centered @ U_r
    residual = centered - proj @ U_r.T
    scores = (residual ** 2).sum(dim=-1)

    sorted_indices = torch.argsort(scores, descending=True)
    top_indices_cpu = sorted_indices[:num_retain]
    retained = emb_cpu[top_indices_cpu]

    if merge and num_retain < N:
        discarded_indices_cpu = sorted_indices[num_retain:]
        discarded = emb_cpu[discarded_indices_cpu]

        cos_sim = F.cosine_similarity(
            discarded.unsqueeze(1),
            retained.unsqueeze(0),
            dim=-1,
        )
        nearest = cos_sim.argmax(dim=-1)

        counts = torch.bincount(nearest, minlength=num_retain).float()
        counts[counts == 0] = 1.0
        sums = torch.zeros_like(retained)
        sums.index_add_(0, nearest, discarded)
        retained = (retained + sums) / (1 + counts.unsqueeze(-1))

    # Transfer results back to original device and dtype
    retained = retained.to(dtype=orig_dtype, device=orig_device)
    top_indices = top_indices_cpu.to(orig_device)

    return retained, top_indices


def lrcp_compress_with_positions(
    embeddings: torch.Tensor,
    positions: torch.Tensor,
    num_retain: int,
    subspace_dim: int = 4,
    merge: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LRCP compression that also prunes and merges mrope positions.

    When position channels are appended to embeddings (as done for
    M-RoPE models like Qwen2.5-VL), this function:
    1. Runs LRCP on the embedding part (first D columns)
    2. Prunes the position part using the same top_indices
    3. Optionally merges position channels alongside embeddings

    Uses integer indexing (top_indices) instead of boolean mask
    indexing, to avoid aclnnNonzeroV2 on NPU/Ascend.

    Args:
        embeddings: Token embeddings of shape (N, D+P) where the
            last P columns are position channels.
        positions: M-RoPE position channels of shape (N, P).
            Can be None if no position channels are present.
        num_retain: Number of tokens to retain.
        subspace_dim: PCA subspace dimension.
        merge: Whether to merge discarded tokens.

    Returns:
        Tuple of:
        - compressed_embeddings: (K, D) retained embeddings
        - compressed_positions: (K, P) retained positions (or None)
    """
    if positions is not None:
        emb_only = embeddings[:, :embeddings.shape[1] - positions.shape[1]]
    else:
        emb_only = embeddings

    compressed, top_indices = lrcp_compress(
        emb_only, num_retain, subspace_dim, merge
    )

    if positions is not None:
        compressed_positions = positions[top_indices]
    else:
        compressed_positions = None

    return compressed, compressed_positions
