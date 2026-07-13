# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import torch

from vllm.autogaze.config import AutoGazeConfig
from vllm.autogaze.processor import build_qwen3_5_autogaze_outputs
from vllm.autogaze.vision import run_qwen3_5_autogaze_vision

logger = logging.getLogger(__name__)

_config = AutoGazeConfig()
_applied = False
_original_call_hf_processor = None
_original_get_mm_fields_config = None
_original_get_prompt_updates = None
_original_parse_video_input = None
_original_process_video_input = None
_original_get_mrope_input_positions = None

_AUTOGAZE_FIELDS = (
    "autogaze_gazing_pos",
    "autogaze_gazing_pos_length",
    "autogaze_num_tokens_per_frame",
    "autogaze_num_frames",
    "autogaze_mrope_positions",
    "autogaze_num_output_tokens",
    "autogaze_full_grid_thw",
    "autogaze_num_grids",
    "autogaze_full_patch_counts",
)


def _is_qwen3_5_processor(processor) -> bool:
    return processor.info.__class__.__name__ in {
        "Qwen3_5ProcessingInfo",
        "Qwen3_5MoeProcessingInfo",
    }


def _field_data(feature, key: str):
    field = feature.data.get(key)
    return None if field is None else field.data


def _patch_processor() -> None:
    global _original_call_hf_processor
    global _original_get_mm_fields_config
    global _original_get_prompt_updates

    from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
    from vllm.multimodal.inputs import MultiModalFieldConfig
    from vllm.multimodal.processing import PromptReplacement

    if _original_call_hf_processor is None:
        _original_call_hf_processor = Qwen3VLMultiModalProcessor._call_hf_processor
        _original_get_mm_fields_config = (
            Qwen3VLMultiModalProcessor._get_mm_fields_config
        )
        _original_get_prompt_updates = Qwen3VLMultiModalProcessor._get_prompt_updates

    def autogaze_call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ):
        outputs = _original_call_hf_processor(
            self,
            prompt,
            mm_data,
            mm_kwargs,
            tok_kwargs,
        )
        if not _config.enabled or not _is_qwen3_5_processor(self):
            return outputs
        return build_qwen3_5_autogaze_outputs(
            self,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
            original_outputs=outputs,
            config=_config,
        )

    def autogaze_get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs,
    ):
        fields = dict(
            _original_get_mm_fields_config(
                self,
                hf_inputs,
                hf_processor_mm_kwargs,
            )
        )
        if "autogaze_full_patch_counts" not in hf_inputs:
            return fields

        fields["pixel_values_videos"] = MultiModalFieldConfig.flat_from_sizes(
            "video", hf_inputs["autogaze_full_patch_counts"]
        )
        for key in _AUTOGAZE_FIELDS:
            fields[key] = MultiModalFieldConfig.batched(
                "video",
                keep_on_cpu=key != "autogaze_gazing_pos",
            )
        return fields

    def autogaze_get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs,
        out_mm_kwargs,
    ):
        updates = list(
            _original_get_prompt_updates(
                self,
                mm_items,
                hf_processor_mm_kwargs,
                out_mm_kwargs,
            )
        )
        if (
            not _config.enabled
            or not _is_qwen3_5_processor(self)
            or "video" not in out_mm_kwargs
            or not out_mm_kwargs["video"]
            or "autogaze_num_tokens_per_frame" not in out_mm_kwargs["video"][0]
        ):
            return updates

        tokenizer = self.info.get_tokenizer()
        hf_config = self.info.get_hf_config()

        def get_video_replacement(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            counts = out_item["autogaze_num_tokens_per_frame"].data
            num_frames = int(out_item["autogaze_num_frames"].data)
            tokens_per_frame = [int(value) for value in counts[:num_frames].tolist()]
            timestamps = out_item["timestamps"].data
            return Qwen3VLMultiModalProcessor.get_video_repl(
                tokens_per_frame=tokens_per_frame,
                timestamps=timestamps,
                tokenizer=tokenizer,
                vision_start_token_id=hf_config.vision_start_token_id,
                vision_end_token_id=hf_config.vision_end_token_id,
                video_token_id=hf_config.video_token_id,
                select_token_id=True,
            )

        updates = [update for update in updates if update.modality != "video"]
        updates.append(
            PromptReplacement(
                modality="video",
                target="<|vision_start|><|video_pad|><|vision_end|>",
                replacement=get_video_replacement,
            )
        )
        return updates

    Qwen3VLMultiModalProcessor._call_hf_processor = autogaze_call_hf_processor
    Qwen3VLMultiModalProcessor._get_mm_fields_config = autogaze_get_mm_fields_config
    Qwen3VLMultiModalProcessor._get_prompt_updates = autogaze_get_prompt_updates


def _patch_model() -> None:
    global _original_parse_video_input
    global _original_process_video_input
    global _original_get_mrope_input_positions

    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from vllm.multimodal.evs import compute_mrope_for_media

    if _original_parse_video_input is None:
        _original_parse_video_input = (
            Qwen3VLForConditionalGeneration._parse_and_validate_video_input
        )
        _original_process_video_input = (
            Qwen3VLForConditionalGeneration._process_video_input
        )
        _original_get_mrope_input_positions = (
            Qwen3VLForConditionalGeneration.get_mrope_input_positions
        )

    def autogaze_parse_video_input(self, **kwargs: object):
        video_input = _original_parse_video_input(self, **kwargs)
        if video_input is None or "autogaze_gazing_pos" not in kwargs:
            return video_input
        for key in _AUTOGAZE_FIELDS:
            video_input[key] = kwargs.get(key)  # type: ignore[literal-required]
        return video_input

    def autogaze_process_video_input(self, video_input):
        if "autogaze_gazing_pos" not in video_input:
            return _original_process_video_input(self, video_input)
        if video_input["type"] != "pixel_values_videos":
            raise ValueError("AutoGaze requires Qwen pixel video inputs")

        pixels = video_input["pixel_values_videos"]
        full_patch_counts = video_input["autogaze_full_patch_counts"]
        selected_rows = video_input["autogaze_gazing_pos"]
        selected_lengths = video_input["autogaze_gazing_pos_length"]
        tokens_per_frame = video_input["autogaze_num_tokens_per_frame"]
        num_frames = video_input["autogaze_num_frames"]
        full_grids = video_input["autogaze_full_grid_thw"]
        num_grids = video_input["autogaze_num_grids"]

        merge_unit = self.visual.spatial_merge_size**2
        outputs: list[torch.Tensor] = []
        pixel_offset = 0
        for item_idx, patch_count_value in enumerate(full_patch_counts.tolist()):
            patch_count = int(patch_count_value)
            item_pixels = pixels[pixel_offset : pixel_offset + patch_count]
            pixel_offset += patch_count

            gaze_length = int(selected_lengths[item_idx])
            item_gaze = selected_rows[item_idx, :gaze_length]
            frame_count = int(num_frames[item_idx])
            output_counts = [
                int(value)
                for value in tokens_per_frame[item_idx, :frame_count].tolist()
            ]
            input_counts = [count * merge_unit for count in output_counts]
            grid_count = int(num_grids[item_idx])
            grid_list = full_grids[item_idx, :grid_count].tolist()

            embeddings = run_qwen3_5_autogaze_vision(
                self.visual,
                item_pixels,
                full_grid_thw=grid_list,
                gazing_pos=item_gaze,
                input_tokens_per_frame=input_counts,
                attention_type=_config.attention_type,
                frame_independent_encoding=_config.frame_independent_encoding,
            )
            if embeddings.shape[0] != sum(output_counts):
                raise RuntimeError(
                    "Qwen AutoGaze merger output does not match gaze token count"
                )
            outputs.append(embeddings)
        return tuple(outputs)

    def autogaze_get_mrope_input_positions(self, input_tokens, mm_features):
        if not any(
            _field_data(feature, "autogaze_mrope_positions") is not None
            for feature in mm_features
        ):
            return _original_get_mrope_input_positions(
                self,
                input_tokens,
                mm_features,
            )

        sequence_length = len(input_tokens)
        positions = np.zeros((3, sequence_length), dtype=np.int64)
        cursor = 0
        next_position = 0

        def fill_text(start: int, end: int) -> None:
            nonlocal next_position
            if end <= start:
                return
            values = np.arange(next_position, next_position + end - start)
            positions[:, start:end] = values
            next_position += end - start

        for feature in sorted(mm_features, key=lambda item: item.mm_position.offset):
            if feature.modality == "image":
                token_id = self.config.image_token_id
                grid = _field_data(feature, "image_grid_thw")
                local_positions = compute_mrope_for_media(
                    grid,
                    self.visual.spatial_merge_size,
                )[:, :3].cpu()
                groups = [local_positions]
            elif feature.modality == "video":
                token_id = self.config.video_token_id
                autogaze_positions = _field_data(feature, "autogaze_mrope_positions")
                if autogaze_positions is None:
                    grid = _field_data(feature, "video_grid_thw")
                    all_positions = compute_mrope_for_media(
                        grid,
                        self.visual.spatial_merge_size,
                    )[:, :3].cpu()
                    tokens_per_frame = all_positions.shape[0] // int(grid[0])
                    groups = list(all_positions.split(tokens_per_frame))
                else:
                    output_tokens = int(
                        _field_data(feature, "autogaze_num_output_tokens")
                    )
                    autogaze_positions = autogaze_positions[:output_tokens].cpu()
                    frame_count = int(_field_data(feature, "autogaze_num_frames"))
                    counts_tensor = _field_data(
                        feature, "autogaze_num_tokens_per_frame"
                    )
                    counts = [int(value) for value in counts_tensor[:frame_count]]
                    groups = list(autogaze_positions.split(counts))
            else:
                raise ValueError(f"Unsupported modality: {feature.modality}")

            search_from = max(cursor, int(feature.mm_position.offset))
            for group in groups:
                count = group.shape[0]
                media_indices: list[int] = []
                while search_from < sequence_length and len(media_indices) < count:
                    if input_tokens[search_from] == token_id:
                        media_indices.append(search_from)
                    search_from += 1
                if len(media_indices) != count:
                    raise ValueError(
                        "Could not align AutoGaze embeddings with Qwen placeholders"
                    )

                fill_text(cursor, media_indices[0])
                if media_indices != list(
                    range(media_indices[0], media_indices[0] + count)
                ):
                    raise ValueError("Qwen media tokens must be contiguous per frame")

                local = group.numpy().T
                positions[:, media_indices[0] : media_indices[-1] + 1] = (
                    local + next_position
                )
                next_position += int(local.max()) + 1 if local.size else 0
                cursor = media_indices[-1] + 1

        fill_text(cursor, sequence_length)
        position_tensor = torch.from_numpy(positions)
        delta = int(positions.max() + 1 - sequence_length)
        return position_tensor, delta

    Qwen3_5ForConditionalGeneration._parse_and_validate_video_input = (
        autogaze_parse_video_input
    )
    Qwen3_5ForConditionalGeneration._process_video_input = autogaze_process_video_input
    Qwen3_5ForConditionalGeneration.get_mrope_input_positions = (
        autogaze_get_mrope_input_positions
    )


def apply_patches(config: AutoGazeConfig | None = None) -> None:
    """Install the dormant Qwen3.5 AutoGaze adapter patches."""
    global _applied, _config

    if config is None:
        config = AutoGazeConfig.from_env()
    config.validate()
    _config = config
    if not config.enabled:
        return
    if _applied:
        logger.info("AutoGaze patches already applied; runtime config updated")
        return

    _patch_processor()
    _patch_model()
    _applied = True
    logger.info("Enabled non-invasive AutoGaze adapter for Qwen3.5")
