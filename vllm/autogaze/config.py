# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

AutoGazeAttentionType = Literal["block_causal", "causal", "bidirectional"]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AutoGazeConfig:
    """Opt-in configuration for the Qwen3.5 AutoGaze adapter.

    AutoGaze remains an external dependency. Keeping this configuration in a
    plugin makes the stock Qwen3.5 processor and model path completely dormant
    unless ``enabled`` is explicitly set.
    """

    enabled: bool = False
    model_id: str = "nvidia/AutoGaze"
    scales: tuple[int, ...] = (64, 128, 224, 448)
    gazing_ratio: float = 0.1
    task_loss_requirement: float | None = 0.7
    attention_type: AutoGazeAttentionType = "block_causal"
    frame_independent_encoding: bool = False
    device: str = "cuda"

    @classmethod
    def from_env(cls) -> AutoGazeConfig:
        scales = tuple(
            int(scale)
            for scale in os.getenv("AUTOGAZE_SCALES", "64+128+224+448").split("+")
        )
        task_loss_raw = os.getenv("AUTOGAZE_TASK_LOSS", "0.7").strip()
        task_loss = (
            None
            if task_loss_raw.lower() in {"", "none", "null"}
            else float(task_loss_raw)
        )
        config = cls(
            enabled=_env_bool("AUTOGAZE_ENABLED", False),
            model_id=os.getenv("AUTOGAZE_MODEL_ID", "nvidia/AutoGaze"),
            scales=scales,
            gazing_ratio=float(os.getenv("AUTOGAZE_GAZING_RATIO", "0.1")),
            task_loss_requirement=task_loss,
            attention_type=os.getenv(  # type: ignore[arg-type]
                "AUTOGAZE_ATTN_TYPE", "block_causal"
            ).lower(),
            frame_independent_encoding=_env_bool(
                "AUTOGAZE_FRAME_INDEPENDENT", False
            ),
            device=os.getenv("AUTOGAZE_DEVICE", "cuda"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.scales:
            raise ValueError("AutoGaze scales must not be empty")
        if tuple(sorted(set(self.scales))) != self.scales:
            raise ValueError("AutoGaze scales must be unique and increasing")
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("AutoGaze scales must be positive")
        if not 0 < self.gazing_ratio <= 1:
            raise ValueError("AutoGaze gazing_ratio must be in (0, 1]")
        if self.attention_type not in {
            "block_causal",
            "causal",
            "bidirectional",
        }:
            raise ValueError(
                "AutoGaze attention_type must be block_causal, causal, or bidirectional"
            )
