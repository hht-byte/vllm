# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence

import torch


def coalesce_gazing_to_tubelets(
    gazing_pos: torch.Tensor,
    if_padded_gazing: torch.Tensor,
    num_gazing_each_frame: torch.Tensor,
    *,
    tokens_per_frame: int,
    temporal_patch_size: int,
) -> list[list[int]]:
    """Convert frame-level AutoGaze output to Qwen temporal tubelets.

    Qwen3.5 patch embedding consumes ``temporal_patch_size`` frames in one
    tubelet. A spatial patch is kept when AutoGaze selected it in any source
    frame in that tubelet. Selection order is stable, which preserves the
    autoregressive gaze order as far as possible while removing duplicates.

    The function operates on one video. AutoGaze can return a batched tensor,
    but callers split that batch before adapting it to vLLM media items.
    """
    if gazing_pos.ndim == 2:
        if gazing_pos.shape[0] != 1:
            raise ValueError("Expected gazing output for exactly one video")
        gazing_pos = gazing_pos[0]
    if if_padded_gazing.ndim == 2:
        if_padded_gazing = if_padded_gazing[0]

    frame_lengths = [int(length) for length in num_gazing_each_frame.tolist()]
    if sum(frame_lengths) != gazing_pos.numel():
        raise ValueError("sum(num_gazing_each_frame) must match the gazing_pos length")
    if temporal_patch_size <= 0:
        raise ValueError("temporal_patch_size must be positive")

    positions_by_frame: list[list[int]] = []
    offset = 0
    for frame_idx, length in enumerate(frame_lengths):
        frame_pos = gazing_pos[offset : offset + length]
        frame_padding = if_padded_gazing[offset : offset + length]
        offset += length

        local_positions: list[int] = []
        for position in frame_pos[~frame_padding].tolist():
            local_position = int(position) - frame_idx * tokens_per_frame
            if not 0 <= local_position < tokens_per_frame:
                raise ValueError(
                    f"AutoGaze position {position} is outside frame {frame_idx}"
                )
            local_positions.append(local_position)
        positions_by_frame.append(local_positions)

    tubelets: list[list[int]] = []
    for start in range(0, len(positions_by_frame), temporal_patch_size):
        seen: set[int] = set()
        tubelet: list[int] = []
        for frame in positions_by_frame[start : start + temporal_patch_size]:
            for position in frame:
                if position not in seen:
                    seen.add(position)
                    tubelet.append(position)
        tubelets.append(tubelet)
    return tubelets


def expand_gazing_for_qwen_merge(
    tubelet_positions: Sequence[Sequence[int]],
    scale_patch_grids: Sequence[tuple[int, int]],
    *,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, list[int], torch.Tensor, list[list[int]]]:
    """Map AutoGaze macro-patches to Qwen patch rows and MRoPE positions.

    AutoGaze selects one downstream token at ``patch_size * merge_size``
    granularity. Qwen's merger, however, expects every selected token to be a
    complete ``merge_size x merge_size`` group. This function expands every
    gaze to that full group so the pretrained merger layout is preserved.

    Returns the selected Qwen patch-row indices, output tokens per tubelet,
    normalized ``(t, h, w)`` MRoPE coordinates, and the full frame-major grid
    list used for Qwen positional embedding interpolation.
    """
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")

    merge_unit = spatial_merge_size**2
    macro_grids: list[tuple[int, int]] = []
    for height, width in scale_patch_grids:
        if height % spatial_merge_size or width % spatial_merge_size:
            raise ValueError(
                "Every Qwen patch grid must be divisible by spatial_merge_size"
            )
        macro_grids.append((height // spatial_merge_size, width // spatial_merge_size))

    macro_counts = [height * width for height, width in macro_grids]
    raw_counts = [height * width for height, width in scale_patch_grids]
    tokens_per_frame = sum(macro_counts)
    rows_per_frame = sum(raw_counts)
    max_macro_h = max(height for height, _ in macro_grids)
    max_macro_w = max(width for _, width in macro_grids)

    macro_offsets = [0]
    raw_offsets = [0]
    for macro_count, raw_count in zip(macro_counts, raw_counts):
        macro_offsets.append(macro_offsets[-1] + macro_count)
        raw_offsets.append(raw_offsets[-1] + raw_count)

    selected_rows: list[int] = []
    output_tokens_per_frame: list[int] = []
    mrope_positions: list[tuple[int, int, int]] = []
    full_grid_thw: list[list[int]] = []

    for frame_idx, frame_positions in enumerate(tubelet_positions):
        full_grid_thw.extend([1, height, width] for height, width in scale_patch_grids)
        output_tokens_per_frame.append(len(frame_positions))
        for position in frame_positions:
            if not 0 <= position < tokens_per_frame:
                raise ValueError(
                    f"AutoGaze position {position} exceeds the multi-scale grid"
                )

            scale_idx = next(
                idx
                for idx in range(len(macro_counts))
                if position < macro_offsets[idx + 1]
            )
            local_macro = position - macro_offsets[scale_idx]
            macro_h, macro_w = macro_grids[scale_idx]
            macro_row, macro_col = divmod(local_macro, macro_w)

            row_start = (
                frame_idx * rows_per_frame
                + raw_offsets[scale_idx]
                + local_macro * merge_unit
            )
            selected_rows.extend(range(row_start, row_start + merge_unit))

            normalized_row = min(
                max_macro_h - 1,
                ((2 * macro_row + 1) * max_macro_h) // (2 * macro_h),
            )
            normalized_col = min(
                max_macro_w - 1,
                ((2 * macro_col + 1) * max_macro_w) // (2 * macro_w),
            )
            mrope_positions.append((frame_idx, normalized_row, normalized_col))

    return (
        torch.tensor(selected_rows, dtype=torch.long),
        output_tokens_per_frame,
        torch.tensor(mrope_positions, dtype=torch.long).reshape(-1, 3),
        full_grid_thw,
    )


def build_autogaze_attention_mask(
    input_tokens_per_frame: Sequence[int],
    *,
    attention_type: str,
    frame_independent_encoding: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build the boolean attention mask described in AutoGaze INTEGRATION.md."""
    lengths = [int(length) for length in input_tokens_per_frame]
    if any(length < 0 for length in lengths):
        raise ValueError("Frame token counts must be non-negative")
    total_tokens = sum(lengths)

    if attention_type == "bidirectional":
        mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool, device=device)
    elif attention_type == "causal":
        mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool, device=device)
        mask = torch.tril(mask)
    elif attention_type == "block_causal":
        mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)
        query_start = 0
        for length in lengths:
            query_end = query_start + length
            mask[query_start:query_end, :query_end] = True
            query_start = query_end
    else:
        raise ValueError(f"Unsupported AutoGaze attention type: {attention_type}")

    if frame_independent_encoding:
        independent = torch.zeros_like(mask)
        start = 0
        for length in lengths:
            end = start + length
            independent[start:end, start:end] = True
            start = end
        mask &= independent
    return mask
