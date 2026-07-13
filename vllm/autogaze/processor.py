# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
from transformers.video_utils import VideoMetadata

from vllm.autogaze.config import AutoGazeConfig
from vllm.autogaze.core import (
    coalesce_gazing_to_tubelets,
    expand_gazing_for_qwen_merge,
)
from vllm.multimodal.processing import BaseMultiModalProcessor

_runner_key: tuple[str, str, int] | None = None
_autogaze_model = None
_autogaze_processor = None


def _get_autogaze_runner(config: AutoGazeConfig):
    global _autogaze_model, _autogaze_processor, _runner_key

    largest_scale = config.scales[-1]
    key = (config.model_id, config.device, largest_scale)
    if _runner_key == key:
        return _autogaze_model, _autogaze_processor

    try:
        from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
    except ImportError as exc:
        raise RuntimeError(
            "AutoGaze support is enabled, but the official AutoGaze package "
            "is not installed. Install NVlabs/AutoGaze in the vLLM "
            "environment."
        ) from exc

    model = AutoGaze.from_pretrained(config.model_id, device_map=None)
    model = model.to(config.device).eval()
    processor = AutoGazeImageProcessor.from_pretrained(
        config.model_id,
        size={"height": largest_scale, "width": largest_scale},
    )

    _runner_key = key
    _autogaze_model = model
    _autogaze_processor = processor
    return model, processor


def _to_thwc_video(video: object) -> torch.Tensor:
    tensor = torch.as_tensor(video)
    if tensor.ndim != 4:
        raise ValueError("AutoGaze expects video input with four dimensions")
    if tensor.shape[-1] == 3:
        return tensor
    if tensor.shape[1] == 3:
        return tensor.permute(0, 2, 3, 1)
    raise ValueError("AutoGaze expects RGB video input")


def _sample_and_resize_video(
    video: object,
    *,
    num_frames: int,
    size: int,
) -> np.ndarray:
    frames = _to_thwc_video(video)
    if frames.shape[0] == 0:
        raise ValueError("AutoGaze cannot process an empty video")

    indices = torch.linspace(0, frames.shape[0] - 1, num_frames).round().long()
    frames = frames[indices]
    frames = frames.permute(0, 3, 1, 2).to(dtype=torch.float32)
    frames = F.interpolate(
        frames,
        size=(size, size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    frames = frames.clamp(0, 255).round().to(dtype=torch.uint8)
    return frames.permute(0, 2, 3, 1).cpu().numpy()


def _run_autogaze(
    video: np.ndarray,
    *,
    config: AutoGazeConfig,
    target_patch_size: int,
) -> dict[str, torch.Tensor]:
    model, processor = _get_autogaze_runner(config)
    inputs = processor(
        videos=list(video),
        size={"height": config.scales[-1], "width": config.scales[-1]},
        return_tensors="pt",
    )["pixel_values"]
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.as_tensor(inputs)
    inputs = inputs.to(config.device)

    with torch.inference_mode():
        output = model(
            {"video": inputs},
            gazing_ratio=config.gazing_ratio,
            task_loss_requirement=config.task_loss_requirement,
            target_scales=list(config.scales),
            target_patch_size=target_patch_size,
            generate_only=True,
        )

    def as_cpu_tensor(value: object) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return torch.as_tensor(value)

    return {
        "gazing_pos": as_cpu_tensor(output["gazing_pos"]),
        "num_gazing_each_frame": as_cpu_tensor(output["num_gazing_each_frame"]),
        "if_padded_gazing": as_cpu_tensor(output["if_padded_gazing"]),
    }


def _video_metadata_for_processed_frames(
    metadata: Mapping[str, Any],
    num_frames: int,
) -> VideoMetadata:
    values = {
        key: value for key, value in metadata.items() if key != "do_sample_frames"
    }
    fps = float(values.get("fps", 1.0))
    values.update(
        total_num_frames=num_frames,
        frames_indices=list(range(num_frames)),
        duration=num_frames / fps,
    )
    return VideoMetadata(**values)


def _qwen_patchify_scales(
    processor,
    video: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    config: AutoGazeConfig,
    mm_kwargs: Mapping[str, object],
    tok_kwargs: Mapping[str, object],
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    clean_kwargs = dict(mm_kwargs)
    for key in (
        "fps",
        "num_frames",
        "size",
        "min_pixels",
        "max_pixels",
    ):
        clean_kwargs.pop(key, None)
    clean_kwargs.update(do_resize=False, do_sample_frames=False)

    metadata_obj = _video_metadata_for_processed_frames(metadata, len(video))
    pixels_per_scale: list[torch.Tensor] = []
    scale_grids: list[tuple[int, int]] = []
    grid_t: int | None = None

    for scale in config.scales:
        if scale == config.scales[-1]:
            scaled_video = video
        else:
            frames = torch.from_numpy(video).permute(0, 3, 1, 2).float()
            frames = F.interpolate(
                frames,
                size=(scale, scale),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            scaled_video = (
                frames.clamp(0, 255)
                .round()
                .to(dtype=torch.uint8)
                .permute(0, 2, 3, 1)
                .numpy()
            )

        outputs = BaseMultiModalProcessor._call_hf_processor(
            processor,
            prompt="<|vision_start|><|video_pad|><|vision_end|>",
            mm_data={
                "videos": [[scaled_video]],
                "video_metadata": [[metadata_obj]],
            },
            mm_kwargs=clean_kwargs,
            tok_kwargs=tok_kwargs,
        )
        grid = outputs["video_grid_thw"][0]
        current_t, height, width = (int(value) for value in grid.tolist())
        if grid_t is None:
            grid_t = current_t
        elif grid_t != current_t:
            raise ValueError("All AutoGaze scales must have the same temporal grid")
        pixels_per_scale.append(outputs["pixel_values_videos"])
        scale_grids.append((height, width))

    assert grid_t is not None
    frame_major_pixels: list[torch.Tensor] = []
    for frame_idx in range(grid_t):
        for pixels, (height, width) in zip(pixels_per_scale, scale_grids):
            patches_per_frame = height * width
            start = frame_idx * patches_per_frame
            frame_major_pixels.append(pixels[start : start + patches_per_frame])
    return torch.cat(frame_major_pixels), scale_grids


def _pad_tensors(tensors: list[torch.Tensor], value: int = 0) -> torch.Tensor:
    if not tensors:
        raise ValueError("Cannot pad an empty tensor list")
    max_shape = tuple(
        max(tensor.shape[dim] for tensor in tensors) for dim in range(tensors[0].ndim)
    )
    output = torch.full(
        (len(tensors), *max_shape),
        value,
        dtype=tensors[0].dtype,
    )
    for idx, tensor in enumerate(tensors):
        slices = (idx, *(slice(0, size) for size in tensor.shape))
        output[slices] = tensor
    return output


def build_qwen3_5_autogaze_outputs(
    processor,
    *,
    mm_data: Mapping[str, object],
    mm_kwargs: Mapping[str, object],
    tok_kwargs: Mapping[str, object],
    original_outputs: BatchFeature,
    config: AutoGazeConfig,
) -> BatchFeature:
    """Replace standard Qwen video inputs with AutoGaze multi-scale inputs."""
    videos = list(mm_data.get("videos", []))
    if not videos:
        return original_outputs

    hf_config = processor.info.get_hf_config()
    vision_config = hf_config.vision_config
    patch_size = int(vision_config.patch_size)
    merge_size = int(vision_config.spatial_merge_size)
    temporal_patch_size = int(vision_config.temporal_patch_size)
    target_patch_size = patch_size * merge_size
    if any(scale % target_patch_size for scale in config.scales):
        raise ValueError(
            "Every AutoGaze scale must be divisible by Qwen patch_size * "
            "spatial_merge_size"
        )

    timestamps = original_outputs["timestamps"]
    original_grid_thw = original_outputs["video_grid_thw"]
    per_video: list[dict[str, torch.Tensor]] = []

    for video_idx, item in enumerate(videos):
        raw_video, metadata = item
        num_tubelets = int(original_grid_thw[video_idx, 0])
        num_source_frames = num_tubelets * temporal_patch_size
        video = _sample_and_resize_video(
            raw_video,
            num_frames=num_source_frames,
            size=config.scales[-1],
        )
        gaze = _run_autogaze(
            video,
            config=config,
            target_patch_size=target_patch_size,
        )

        tokens_per_source_frame = sum(
            (scale // target_patch_size) ** 2 for scale in config.scales
        )
        tubelet_positions = coalesce_gazing_to_tubelets(
            gaze["gazing_pos"],
            gaze["if_padded_gazing"],
            gaze["num_gazing_each_frame"],
            tokens_per_frame=tokens_per_source_frame,
            temporal_patch_size=temporal_patch_size,
        )
        if len(tubelet_positions) != num_tubelets:
            raise ValueError(
                "AutoGaze and Qwen produced different temporal grid lengths"
            )

        full_pixels, scale_grids = _qwen_patchify_scales(
            processor,
            video,
            metadata,
            config=config,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        (
            selected_rows,
            output_tokens_per_frame,
            mrope_positions,
            full_grid_thw,
        ) = expand_gazing_for_qwen_merge(
            tubelet_positions,
            scale_grids,
            spatial_merge_size=merge_size,
        )
        if not output_tokens_per_frame or min(output_tokens_per_frame) <= 0:
            raise ValueError("AutoGaze must retain at least one token per tubelet")

        per_video.append(
            {
                "pixels": full_pixels,
                "selected_rows": selected_rows,
                "tokens_per_frame": torch.tensor(
                    output_tokens_per_frame, dtype=torch.long
                ),
                "mrope_positions": mrope_positions,
                "full_grid_thw": torch.tensor(full_grid_thw, dtype=torch.long),
                "video_grid_thw": torch.tensor(
                    [
                        num_tubelets,
                        config.scales[-1] // patch_size,
                        config.scales[-1] // patch_size,
                    ],
                    dtype=torch.long,
                ),
            }
        )

    outputs = dict(original_outputs)
    outputs.update(
        pixel_values_videos=torch.cat([item["pixels"] for item in per_video]),
        video_grid_thw=torch.stack([item["video_grid_thw"] for item in per_video]),
        timestamps=timestamps,
        autogaze_gazing_pos=_pad_tensors([item["selected_rows"] for item in per_video]),
        autogaze_gazing_pos_length=torch.tensor(
            [item["selected_rows"].numel() for item in per_video],
            dtype=torch.long,
        ),
        autogaze_num_tokens_per_frame=_pad_tensors(
            [item["tokens_per_frame"] for item in per_video]
        ),
        autogaze_num_frames=torch.tensor(
            [item["tokens_per_frame"].numel() for item in per_video],
            dtype=torch.long,
        ),
        autogaze_mrope_positions=_pad_tensors(
            [item["mrope_positions"] for item in per_video]
        ),
        autogaze_num_output_tokens=torch.tensor(
            [item["mrope_positions"].shape[0] for item in per_video],
            dtype=torch.long,
        ),
        autogaze_full_grid_thw=_pad_tensors(
            [item["full_grid_thw"] for item in per_video]
        ),
        autogaze_num_grids=torch.tensor(
            [item["full_grid_thw"].shape[0] for item in per_video],
            dtype=torch.long,
        ),
        autogaze_full_patch_counts=torch.tensor(
            [item["pixels"].shape[0] for item in per_video],
            dtype=torch.long,
        ),
    )
    return BatchFeature(outputs)
