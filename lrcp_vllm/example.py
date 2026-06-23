#!/usr/bin/env python3
"""Example: Using LRCP with vLLM for visual token pruning.

LRCP (Low-Rank Compressibility Guided Pruning) reduces visual token count
by scoring tokens via PCA projection residual onto the dominant low-rank
subspace. This achieves high performance retention under aggressive pruning.

Usage with vLLM OpenAI-compatible server:

    # 1. Apply patches before starting vllm
    python -c "from lrcp_vllm import apply_patches; apply_patches()"

    # 2. Start vllm server with LRCP config
    python -m vllm.entrypoints.openai.api_server \
        --model llava-hf/llava-1.5-7b-hf \
        --lrcp-retention-ratio 0.111 \
        --lrcp-subspace-dim 4 \
        --lrcp-merge True

Usage with vLLM LLM class:

    from lrcp_vllm import apply_patches
    apply_patches()

    from vllm import LLM

    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        lrcp_retention_ratio=0.111,
        lrcp_subspace_dim=4,
        lrcp_merge=True,
    )

Config parameters:
    lrcp_retention_ratio: Fraction of visual tokens to retain.
        - 0.111: retain ~11.1% (88.9% reduction, best for LLaVA-v1.5-7B)
        - 0.222: retain ~22.2% (77.8% reduction)
        - 0.333: retain ~33.3% (66.7% reduction)
    lrcp_subspace_dim: PCA subspace dimension r.
        - 4 for LLaVA-based models (default)
        - 8 for Qwen-based models
    lrcp_merge: Whether to merge discarded tokens (default True).
        Merging improves performance by averaging discarded tokens
        with their nearest retained neighbors.
    lrcp_layer: Intermediate LLM layer for additional pruning.
        - 16 for LLaVA models (recommended)
        - 14 for Qwen models (recommended)
        - None: only apply at encoder output level (default)

Architecture overview:

    The adaptation is non-invasive, using monkey-patching:
    - Core LRCP algorithm: vllm/multimodal/lrcp.py (like evs.py)
    - Config fields: vllm/config/multimodal.py (3 new fields + is_lrcp_enabled)
    - Runtime patches: lrcp_vllm/patches/ (separate package)

    When migrating to a new vllm version:
    1. Re-add lrcp.py and config fields (well-defined, minimal)
    2. Update monkey-patches if model method signatures changed
    3. LRCP algorithm stays unchanged

    The patch flow:
    - Processor level: Updates PlaceholderRange.length to reflect
      pruned token count (computed from retention_ratio)
    - Encoder level: Wraps model's embed_multimodal/_process_image_input
      to apply LRCP after visual encoder output
    - Layer level (optional): Wraps LLM forward to apply LRCP at
      intermediate layers during prefill

    For M-RoPE models (Qwen2.5-VL, Qwen3-VL):
    - LRCP appends position channels to embeddings (like EVS)
    - vllm's existing recompute_mrope_positions handles position
      correction after pruning
"""

import sys


def main():
    # Apply LRCP patches before importing vllm LLM
    from lrcp_vllm import apply_patches
    apply_patches()

    # Now vllm can be used with LRCP config
    print("LRCP patches applied. vLLM is ready to use with LRCP config.")
    print()
    print("Example command:")
    print(
        "  python -m vllm.entrypoints.openai.api_server "
        "--model llava-hf/llava-1.5-7b-hf "
        "--lrcp-retention-ratio 0.111 "
        "--lrcp-subspace-dim 4"
    )


if __name__ == "__main__":
    main()
