#!/usr/bin/env python3
"""LRCP-enhanced vLLM OpenAI-compatible server wrapper.

This script applies LRCP monkey-patches before starting the vLLM
server, enabling visual token pruning via the LRCP method.

Usage:
    python lrcp_vllm/run_server.py \
        --model llava-hf/llava-1.5-7b-hf \
        --lrcp-retention-ratio 0.111 \
        --lrcp-subspace-dim 4 \
        --lrcp-merge \
        [other vllm server args...]

This is equivalent to:
    1. from lrcp_vllm import apply_patches; apply_patches()
    2. python -m vllm.entrypoints.openai.api_server [args...]

All standard vLLM server arguments are supported in addition to LRCP args.
"""

import sys


def main():
    from lrcp_vllm import apply_patches
    apply_patches()

    from vllm.entrypoints.openai.api_server import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
