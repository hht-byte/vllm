import torch

from vllm.attention_utils.fasa import (
    contextual_agreement,
    fac_mask_mod_factory,
    focused_attention_output,
    full_attention_scores,
    offline_calibration,
    select_top_tokens,
    single_fc_attention_scores,
    tip_scores,
)


def _build_synthetic_dataset():
    torch.manual_seed(0)
    num_steps = 6
    num_layers = 1
    num_heads = 1
    head_dim = 8  # 4 FCs
    queries = torch.randn(num_steps, num_layers, num_heads, head_dim)
    keys = torch.randn(num_steps, num_layers, num_heads, head_dim)

    dominant_fc = 2
    for t in range(num_steps):
        keys[t, 0, 0, dominant_fc * 2 : dominant_fc * 2 + 2] = queries[
            t, 0, 0, dominant_fc * 2 : dominant_fc * 2 + 2
        ]
    return [{"queries": queries, "keys": keys}], dominant_fc


def test_contextual_agreement_range():
    full_scores = torch.tensor([5.0, 4.0, 3.0, 2.0])
    fc_scores = torch.tensor([5.0, 1.0, 3.0, 0.0])
    ca = contextual_agreement(full_scores, fc_scores, topk_k=2)
    assert 0.0 <= ca <= 1.0
    assert ca == 1.0


def test_offline_calibration_finds_dominant_fc():
    dataset, dominant_fc = _build_synthetic_dataset()
    result = offline_calibration(
        calibration_examples=dataset,
        num_layers=1,
        num_heads=1,
        topk_k=2,
        num_dominant_fcs=1,
    )
    assert result.dominant_fc_indices["0"]["0"] == [dominant_fc]


def test_tip_scores_prioritize_selected_dominant_fc():
    torch.manual_seed(1)
    head_dim = 8
    q = torch.randn(head_dim)
    ks = torch.randn(5, head_dim)
    dominant_fc = 1
    ks[3, dominant_fc * 2 : dominant_fc * 2 + 2] = q[dominant_fc * 2 : dominant_fc * 2 + 2]
    positions = torch.arange(5)
    scores = tip_scores(q, ks, 4, positions, [dominant_fc])
    top_idx = select_top_tokens(scores, 1)
    assert top_idx.tolist() == [3]


def test_fac_mask_mod_factory_masks_only_selected_indices():
    mask_mod = fac_mask_mod_factory(torch.tensor([1, 3, 5]))
    kv = torch.tensor([0, 1, 2, 3, 4, 5])
    out = mask_mod(torch.tensor(0), torch.tensor(0), torch.tensor(0), kv)
    assert out.tolist() == [False, True, False, True, False, True]


def test_focused_attention_matches_manual_subset_attention():
    torch.manual_seed(2)
    query = torch.randn(2, 4)
    keys = torch.randn(6, 2, 4)
    values = torch.randn(6, 2, 4)
    selected = torch.tensor([1, 4, 5])
    out = focused_attention_output(query, keys, values, selected)

    k_sel = keys[selected]
    v_sel = values[selected]
    logits = torch.einsum("hd,thd->ht", query, k_sel) * (query.shape[-1] ** -0.5)
    probs = torch.softmax(logits, dim=-1)
    expected = torch.einsum("ht,thd->hd", probs, v_sel)
    torch.testing.assert_close(out, expected)


def test_single_fc_and_full_attention_shapes():
    torch.manual_seed(3)
    q = torch.randn(8)
    k = torch.randn(4, 8)
    positions = torch.arange(4)
    full = full_attention_scores(q, k, 3, positions)
    one_fc = single_fc_attention_scores(q, k, 3, positions, 0)
    assert full.shape == (4,)
    assert one_fc.shape == (4,)
