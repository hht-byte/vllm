#!/usr/bin/env python3
"""Example: Using COAST with vLLM for visual token pruning.

COAST (COntrastive Adaptive Semantic Token Pruning) reduces visual token
count by using entropy-driven dynamic budgeting and contrastive semantic
routing. It preserves both query-aligned semantic evidence and complementary
spatial context through two-tail retention.

Usage with vLLM OpenAI-compatible server:

    # 1. Start vllm server with COAST config (auto-patch is applied)
    vllm serve llava-hf/llava-1.5-7b-hf \
        --coast-retention-ratio 0.222 \
        --coast-alpha-min 0.05 \
        --coast-alpha-max 0.15 \
        --coast-anchor-ratio 0.8

Usage with vLLM LLM class:

    from vllm import LLM

    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        coast_retention_ratio=0.222,
        coast_alpha_min=0.05,
        coast_alpha_max=0.15,
        coast_anchor_ratio=0.8,
    )

Config parameters:
    coast_retention_ratio: Fraction of visual tokens to retain.
        - 0.222: retain ~22.2% (77.8% reduction, default for LLaVA)
        - 0.111: retain ~11.1% (88.9% reduction, aggressive)
    coast_alpha_min: Min fraction of non-anchor budget for context (0.05).
    coast_alpha_max: Max fraction of non-anchor budget for context (0.15).
    coast_anchor_ratio: Fraction of total budget for anchors (0.8).

Architecture overview:

    The adaptation is non-invasive, using monkey-patching:
    - Core COAST algorithm: vllm/multimodal/coast.py (like evs.py)
    - Config fields: vllm/config/multimodal.py (4 new fields + is_coast_enabled)
    - Runtime patches: coast_vllm/patches/ (separate package)
    - Auto-hook: VllmConfig.__post_init__ calls _apply_coast_patches()
"""

import sys


def main():
    print("COAST patches are auto-applied via VllmConfig.__post_init__.")
    print()
    print("Example command:")
    print(
        "  vllm serve llava-hf/llava-1.5-7b-hf "
        "--coast-retention-ratio 0.222 "
        "--coast-alpha-min 0.05 "
        "--coast-alpha-max 0.15 "
        "--coast-anchor-ratio 0.8"
    )


if __name__ == "__main__":
    main()
