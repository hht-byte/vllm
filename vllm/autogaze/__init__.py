# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.autogaze.config import AutoGazeConfig

__all__ = ["AutoGazeConfig", "apply_patches"]


def apply_patches(config: AutoGazeConfig | None = None) -> None:
    # Keep importing vLLM model/processor modules off the normal startup path
    # when the adapter is installed but disabled.
    from vllm.autogaze.patches import apply_patches as _apply_patches

    _apply_patches(config)


def register_plugin() -> None:
    """vLLM general-plugin entry point."""
    config = AutoGazeConfig.from_env()
    if config.enabled:
        apply_patches(config)
