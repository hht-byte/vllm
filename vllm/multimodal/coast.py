# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# COAST: COntrastive Adaptive Semantic Token Pruning
# Paper: https://arxiv.org/abs/2605.09429
# "Evading Visual Aphasia: Contrastive Adaptive Semantic Token Pruning
#  for Vision-Language Models"
#
# COAST is a training-free visual token pruning framework that casts
# compression as adaptive semantic routing. It uses cross-modal signals
# to identify query-specific anchors, estimates contextual dispersion via
# attention entropy, and adapts the retention trade-off between semantic
# evidence and spatial context via contrastive routing.

import torch
import torch.nn.functional as F


def compute_coast_retained_tokens_count(
    original_tokens: int,
    retention_ratio: float,
) -> int:
    if retention_ratio <= 0.0:
        return 1
    num_retain = max(1, int(original_tokens * retention_ratio))
    return min(num_retain, original_tokens)


def _compute_visual_global_score(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute global importance score for each visual token using
    pairwise cosine similarity. For encoder-output-level pruning where
    cross-modal attention is unavailable, we use visual self-similarity
    as a proxy: S^glo_j = mean similarity of token j to all others,
    capturing how central each token is to the visual representation."""
    normed = F.normalize(embeddings.float(), dim=-1)
    sim_matrix = normed @ normed.T
    N = embeddings.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)
    sim_matrix = sim_matrix.masked_fill(~mask, 0.0)
    s_glo = sim_matrix.sum(dim=-1) / max(N - 1, 1)
    return s_glo


def _compute_cross_modal_scores(
    visual_hidden: torch.Tensor,
    text_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cross-modal scores using hidden states at intermediate
    LLM layers. S^glo_j = max text-to-visual similarity (Eq.3).
    S^last_j = similarity from the last text token to each visual token."""
    normed_v = F.normalize(visual_hidden.float(), dim=-1)
    normed_t = F.normalize(text_hidden.float(), dim=-1)
    cross_sim = normed_t @ normed_v.T
    s_glo = cross_sim.max(dim=0).values
    s_last = cross_sim[-1]
    return s_glo, s_last


def _compute_entropy(s_glo: torch.Tensor) -> float:
    """Compute normalized entropy of the global attention/similarity
    distribution (Eq.4). Low H -> concentrated attention; high H ->
    dispersed attention."""
    N = s_glo.shape[0]
    if N <= 1:
        return 0.0
    probs = s_glo / s_glo.sum()
    probs = probs.clamp(min=1e-10)
    H = -(probs * probs.log()).sum().item()
    H_normalized = H / torch.log(torch.tensor(N, dtype=torch.float32)).item()
    return min(max(H_normalized, 0.0), 1.0)


def _contrastive_routing(
    candidate_embeds: torch.Tensor,
    anchor_embeds: torch.Tensor,
    reference_embeds: torch.Tensor,
) -> torch.Tensor:
    """Compute contrastive routing score for each candidate token
    (Eq.8-10). Score(c_i) = Sim_A(c_i) - Sim_R(c_i), where
    Sim_A = max cosine similarity to anchors, Sim_R = mean cosine
    similarity to references."""
    normed_c = F.normalize(candidate_embeds.float(), dim=-1)
    normed_a = F.normalize(anchor_embeds.float(), dim=-1)
    normed_r = F.normalize(reference_embeds.float(), dim=-1)

    sim_a = normed_c @ normed_a.T
    sim_A = sim_a.max(dim=-1).values

    sim_r = normed_c @ normed_r.T
    sim_R = sim_r.mean(dim=-1)

    score = sim_A - sim_R
    return score


def coast_prune_visual_tokens(
    embeddings: torch.Tensor,
    num_retain: int,
    alpha_min: float = 0.05,
    alpha_max: float = 0.15,
    anchor_ratio: float = 0.8,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    """COAST pruning at encoder output level.

    Uses visual self-similarity as a proxy for cross-modal attention
    (since text embeddings are not yet available at this stage).
    Applies entropy-driven dynamic budgeting and contrastive semantic
    routing with two-tail retention.

    Args:
        embeddings: Visual token embeddings (N_v, d).
        num_retain: Number of tokens to retain after pruning.
        alpha_min: Minimum fraction of K_rest for complementary context.
        alpha_max: Maximum fraction of K_rest for complementary context.
        anchor_ratio: Fraction of total budget allocated to anchors.

    Returns:
        Tuple of (compressed_embeddings, retention_mask).
        compressed_embeddings: (num_retain, d) retained token embeddings
            sorted in original order.
        retention_mask: (N_v,) boolean mask indicating which tokens were
            retained.
    """
    N_v = embeddings.shape[0]
    if num_retain >= N_v:
        mask = torch.ones(N_v, dtype=torch.bool, device=embeddings.device)
        return embeddings, mask

    s_glo = _compute_visual_global_score(embeddings)
    H = _compute_entropy(s_glo)

    K_anchor = max(1, min(int(num_retain * anchor_ratio), num_retain))
    K_rest = num_retain - K_anchor

    if K_rest <= 0:
        K_anchor = num_retain
        K_rest = 0
        n1 = 0
        n2 = 0
    elif K_rest == 1:
        n2 = 1
        n1 = 0
    else:
        n2 = max(1, min(int(K_rest * (alpha_min + (alpha_max - alpha_min) * H)), K_rest - 1))
        n1 = K_rest - n2

    anchor_indices = torch.topk(s_glo, k=K_anchor).indices
    K_R = max(1, min(N_v // 10, N_v - K_anchor - num_retain))
    remaining_after_anchor = N_v - K_anchor
    if K_R > remaining_after_anchor:
        K_R = max(1, remaining_after_anchor)
    reference_indices = torch.topk(s_glo, k=K_R, largest=False).indices

    all_selected = anchor_indices
    candidate_mask = torch.ones(N_v, dtype=torch.bool, device=embeddings.device)
    candidate_mask[all_selected] = False
    candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]

    if candidate_indices.shape[0] == 0:
        retention_mask = torch.zeros(N_v, dtype=torch.bool, device=embeddings.device)
        retention_mask[anchor_indices] = True
        retained = embeddings[retention_mask]
        return retained, retention_mask

    candidate_embeds = embeddings[candidate_indices]
    anchor_embeds = embeddings[anchor_indices]
    reference_embeds = embeddings[reference_indices]

    scores = _contrastive_routing(candidate_embeds, anchor_embeds, reference_embeds)

    actual_n1 = min(n1, candidate_indices.shape[0])
    actual_n2 = min(n2, candidate_indices.shape[0] - actual_n1)

    top_n1_idx = torch.topk(scores, k=actual_n1).indices
    bottom_scores_idx = torch.topk(scores, k=actual_n2, largest=False).indices

    selected_candidate_idx = torch.cat([top_n1_idx, bottom_scores_idx])
    selected_indices = candidate_indices[selected_candidate_idx]

    all_kept = torch.cat([anchor_indices, selected_indices])
    all_kept = all_kept.sort()[0]

    retention_mask = torch.zeros(N_v, dtype=torch.bool, device=embeddings.device)
    retention_mask[all_kept] = True

    retained = embeddings[retention_mask]
    return retained, retention_mask


def coast_prune_at_layer(
    hidden_states: torch.Tensor,
    is_visual: torch.BoolTensor,
    num_retain: int,
    alpha_min: float = 0.05,
    alpha_max: float = 0.15,
    anchor_ratio: float = 0.8,
) -> torch.Tensor:
    """COAST pruning at intermediate LLM layer using cross-modal
    hidden state similarity.

    Uses the true cross-modal signal from hidden states: text-to-visual
    similarity serves as S^glo, and last-text-token similarity serves
    as S^last. Applies full COAST algorithm with entropy-driven budgeting
    and contrastive routing.

    Since vllm's attention backend cannot dynamically reduce sequence
    length mid-forward, pruned tokens are merged into their nearest
    retained neighbor via weighted averaging, preserving sequence length
    while concentrating information into retained positions.

    Args:
        hidden_states: Full sequence hidden states (N, d).
        is_visual: Boolean mask (N,) indicating visual token positions.
        num_retain: Number of visual tokens to retain.
        alpha_min: Minimum fraction for complementary context.
        alpha_max: Maximum fraction for complementary context.
        anchor_ratio: Fraction of total budget for anchors.

    Returns:
        hidden_states with pruned visual tokens merged into retained
        neighbors. Shape unchanged (N, d).
    """
    N = hidden_states.shape[0]
    visual_indices = is_visual.nonzero(as_tuple=True)[0]
    N_v = visual_indices.shape[0]

    if num_retain >= N_v or N_v <= 1:
        return hidden_states

    text_indices = (~is_visual).nonzero(as_tuple=True)[0]
    N_t = text_indices.shape[0]

    if N_t == 0:
        s_glo = _compute_visual_global_score(hidden_states[visual_indices])
        s_last = s_glo
    else:
        visual_hidden = hidden_states[visual_indices]
        text_hidden = hidden_states[text_indices]
        s_glo, s_last = _compute_cross_modal_scores(visual_hidden, text_hidden)

    H = _compute_entropy(s_glo)

    K_anchor = max(1, min(int(num_retain * anchor_ratio), num_retain))
    K_rest = num_retain - K_anchor

    if K_rest <= 0:
        K_anchor = num_retain
        K_rest = 0
        n1 = 0
        n2 = 0
    elif K_rest == 1:
        n2 = 1
        n1 = 0
    else:
        n2 = max(1, min(int(K_rest * (alpha_min + (alpha_max - alpha_min) * H)), K_rest - 1))
        n1 = K_rest - n2

    anchor_local = torch.topk(s_last, k=K_anchor).indices
    K_R = max(1, min(N_v // 10, N_v - K_anchor - num_retain))
    remaining_after_anchor = N_v - K_anchor
    if K_R > remaining_after_anchor:
        K_R = max(1, remaining_after_anchor)
    reference_local = torch.topk(s_glo, k=K_R, largest=False).indices

    anchor_indices = visual_indices[anchor_local]
    reference_indices = visual_indices[reference_local]

    all_selected_local = anchor_local
    candidate_mask = torch.ones(N_v, dtype=torch.bool, device=hidden_states.device)
    candidate_mask[all_selected_local] = False
    candidate_local = candidate_mask.nonzero(as_tuple=True)[0]

    if candidate_local.shape[0] == 0:
        retained_local = anchor_local
    else:
        candidate_embeds = hidden_states[visual_indices[candidate_local]]
        anchor_embeds = hidden_states[anchor_indices]
        reference_embeds = hidden_states[reference_indices]

        scores = _contrastive_routing(
            candidate_embeds, anchor_embeds, reference_embeds
        )

        actual_n1 = min(n1, candidate_local.shape[0])
        actual_n2 = min(n2, candidate_local.shape[0] - actual_n1)

        top_n1_local = candidate_local[torch.topk(scores, k=actual_n1).indices]
        bottom_n2_local = candidate_local[
            torch.topk(scores, k=actual_n2, largest=False).indices
        ]

        retained_local = torch.cat([anchor_local, top_n1_local, bottom_n2_local])

    retained_global = visual_indices[retained_local].sort()[0]
    retained_local_mask = torch.zeros(N_v, dtype=torch.bool, device=hidden_states.device)
    retained_local_mask[retained_local] = True
    pruned_global = visual_indices[~retained_local_mask]

    if pruned_global.shape[0] == 0:
        return hidden_states

    retained_embeds = hidden_states[retained_global]
    pruned_embeds = hidden_states[pruned_global]

    normed_retained = F.normalize(retained_embeds.float(), dim=-1)
    normed_pruned = F.normalize(pruned_embeds.float(), dim=-1)

    sim = normed_pruned @ normed_retained.T
    nearest = sim.argmax(dim=-1)

    result = hidden_states.clone()
    counts = torch.zeros(retained_global.shape[0], device=hidden_states.device)
    sums = retained_embeds.clone()

    for p_idx, r_idx in enumerate(nearest.tolist()):
        sums[r_idx] += pruned_embeds[p_idx]
        counts[r_idx] += 1

    merged = sums / (1 + counts.unsqueeze(-1))
    result[retained_global] = merged.to(hidden_states.dtype)

    discard_value = merged.mean(dim=0)
    result[pruned_global] = discard_value.to(hidden_states.dtype)

    return result
