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
          token embeddings
        - retention_mask: (N,) boolean mask indicating retained tokens
    """
    N, D = embeddings.shape
    orig_dtype = embeddings.dtype

    if num_retain >= N:
        return embeddings, torch.ones(N, dtype=torch.bool, device=embeddings.device)

    # Cast to float32 for SVD: BFloat16 not supported on NPU/Ascend
    # and float32 provides sufficient precision for PCA computation
    emb_f32 = embeddings.float()
    mean = emb_f32.mean(dim=0, keepdim=True)
    centered = emb_f32 - mean

    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    U_r = Vh[:subspace_dim, :].T  # (D, r) top r principal directions

    proj = centered @ U_r  # (N, r) projection onto subspace
    residual = centered - proj @ U_r.T  # (N, D) residual component
    scores = (residual ** 2).sum(dim=-1)  # (N,) projection residual scores

    _, top_indices = torch.topk(scores, k=num_retain, largest=True, sorted=True)
    retention_mask = torch.zeros(N, dtype=torch.bool, device=embeddings.device)
    retention_mask[top_indices] = True

    retained = emb_f32[retention_mask]  # (K, D) in float32

    if merge and num_retain < N:
        discarded = emb_f32[~retention_mask]  # (N-K, D)

        cos_sim = F.cosine_similarity(
            discarded.unsqueeze(1),
            retained.unsqueeze(0),
            dim=-1,
        )  # (N-K, K)
        nearest = cos_sim.argmax(dim=-1)  # (N-K,)

        counts = torch.bincount(nearest, minlength=num_retain).float()
        counts[counts == 0] = 1.0
        sums = torch.zeros_like(retained)
        sums.index_add_(0, nearest, discarded)
        retained = (retained + sums) / (1 + counts.unsqueeze(-1))

    # Cast back to original dtype (BFloat16, Float16, etc.)
    retained = retained.to(orig_dtype)

    return retained, retention_mask


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
    2. Prunes the position part using the same retention_mask
    3. Optionally merges position channels alongside embeddings

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

    compressed, retention_mask = lrcp_compress(
        emb_only, num_retain, subspace_dim, merge
    )

    if positions is not None:
        compressed_positions = positions[retention_mask]
    else:
        compressed_positions = None

    return compressed, compressed_positions
