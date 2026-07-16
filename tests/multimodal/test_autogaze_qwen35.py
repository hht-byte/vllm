# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature

import vllm.autogaze.processor as autogaze_processor
from vllm.autogaze.config import AutoGazeConfig
from vllm.autogaze.core import (
    build_autogaze_attention_mask,
    coalesce_gazing_to_tubelets,
    expand_gazing_for_qwen_merge,
)
from vllm.autogaze.vision import run_qwen3_5_autogaze_vision


def test_coalesce_gazing_to_qwen_tubelets_preserves_stable_union():
    gazing_pos = torch.tensor([[0, 1, 0, 6, 7, 5, 10, 12, 10, 17, 18, 15]])
    if_padded = torch.tensor(
        [[False, False, True] * 4],
        dtype=torch.bool,
    )

    tubelets = coalesce_gazing_to_tubelets(
        gazing_pos,
        if_padded,
        torch.tensor([3, 3, 3, 3]),
        tokens_per_frame=5,
        temporal_patch_size=2,
    )

    assert tubelets == [[0, 1, 2], [0, 2, 3]]


def test_expand_gaze_keeps_complete_qwen_merge_groups():
    selected, counts, mrope, grids = expand_gazing_for_qwen_merge(
        [[0, 4, 19], [1]],
        [(4, 4), (8, 8)],
        spatial_merge_size=2,
    )

    assert selected.tolist() == [
        0,
        1,
        2,
        3,
        16,
        17,
        18,
        19,
        76,
        77,
        78,
        79,
        84,
        85,
        86,
        87,
    ]
    assert counts == [3, 1]
    assert mrope.tolist() == [
        [0, 1, 1],
        [0, 0, 0],
        [0, 3, 3],
        [1, 1, 3],
    ]
    assert grids == [[1, 4, 4], [1, 8, 8]] * 2


def test_block_causal_attention_is_bidirectional_within_frame():
    mask = build_autogaze_attention_mask(
        [2, 3],
        attention_type="block_causal",
    )

    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


def test_frame_independent_attention_removes_cross_frame_edges():
    mask = build_autogaze_attention_mask(
        [2, 1],
        attention_type="bidirectional",
        frame_independent_encoding=True,
    )
    assert mask.tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, True],
    ]


def test_autogaze_config_from_env(monkeypatch):
    monkeypatch.setenv("AUTOGAZE_ENABLED", "1")
    monkeypatch.setenv("AUTOGAZE_SCALES", "64+128+256")
    monkeypatch.setenv("AUTOGAZE_TASK_LOSS", "none")
    monkeypatch.setenv("AUTOGAZE_ATTN_TYPE", "causal")

    config = AutoGazeConfig.from_env()

    assert config.enabled
    assert config.scales == (64, 128, 256)
    assert config.task_loss_requirement is None
    assert config.attention_type == "causal"


def test_autogaze_config_rejects_unsorted_scales():
    with pytest.raises(ValueError, match="unique and increasing"):
        AutoGazeConfig(scales=(128, 64)).validate()


def test_image_input_keeps_one_temporal_grid(monkeypatch):
    class _VisionConfig:
        patch_size = 16
        spatial_merge_size = 2

    class _Config:
        vision_config = _VisionConfig()

    class _Info:
        def get_hf_config(self):
            return _Config()

    class _Processor:
        info = _Info()

    def fake_run_autogaze(video, **kwargs):
        assert video.shape == (1, 32, 32, 3)
        return {
            "gazing_pos": torch.tensor([[0]]),
            "num_gazing_each_frame": torch.tensor([1]),
            "if_padded_gazing": torch.tensor([[False]]),
        }

    def fake_patchify(*args, **kwargs):
        return torch.zeros(4, 8), [(2, 2)]

    monkeypatch.setattr(autogaze_processor, "_run_autogaze", fake_run_autogaze)
    monkeypatch.setattr(
        autogaze_processor,
        "_qwen_patchify_image_scales",
        fake_patchify,
    )

    outputs = autogaze_processor.build_qwen3_5_autogaze_outputs(
        _Processor(),
        mm_data={"images": [torch.zeros(3, 32, 32, dtype=torch.uint8)]},
        mm_kwargs={},
        tok_kwargs={},
        original_outputs=BatchFeature(
            {
                "pixel_values": torch.zeros(4, 8),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
        ),
        config=AutoGazeConfig(scales=(32,), device="cpu"),
    )

    assert outputs["autogaze_image_num_output_tokens"].tolist() == [1]
    assert outputs["autogaze_image_gazing_pos_length"].tolist() == [4]
    assert outputs["autogaze_image_full_patch_counts"].tolist() == [4]
    assert outputs["image_grid_thw"].tolist() == [[1, 2, 2]]


class _FakeParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        return self.linear(inputs), None


class _FakeAttention(nn.Module):
    num_attention_heads_per_partition = 2
    hidden_size_per_attention_head = 2

    def __init__(self):
        super().__init__()
        self.qkv = _FakeParallelLinear(4, 12)
        self.proj = _FakeParallelLinear(4, 4)
        self.apply_rotary_emb = lambda inputs, cos, sin: inputs


class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _FakeAttention()
        self.norm1 = nn.LayerNorm(4)
        self.norm2 = nn.LayerNorm(4)
        self.mlp = nn.Linear(4, 4)


class _FakeMerger(nn.Module):
    def forward(self, inputs):
        return inputs.reshape(-1, 4, 4).mean(dim=1)


class _FakeVisual(nn.Module):
    spatial_merge_size = 2
    deepstack_visual_indexes: list[int] = []

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock()])
        self.deepstack_merger_list = nn.ModuleList()
        self.merger = _FakeMerger()
        self.patch_embed = nn.Identity()

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    def fast_pos_embed_interpolate(self, grid_thw):
        length = sum(t * h * w for t, h, w in grid_thw)
        return torch.zeros(length, 4)

    def rot_pos_emb(self, grid_thw):
        length = sum(t * h * w for t, h, w in grid_thw)
        return torch.zeros(length, 2), torch.zeros(length, 2)


def test_qwen_vision_adapter_runs_only_selected_merge_groups():
    torch.manual_seed(0)
    visual = _FakeVisual()
    pixels = torch.randn(12, 4)

    output = run_qwen3_5_autogaze_vision(
        visual,
        pixels,
        full_grid_thw=[[1, 2, 2], [1, 2, 2], [1, 2, 2]],
        gazing_pos=torch.tensor([0, 1, 2, 3, 8, 9, 10, 11]),
        input_tokens_per_frame=[4, 4],
        attention_type="block_causal",
        frame_independent_encoding=False,
    )

    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()
