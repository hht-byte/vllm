import logging

logger = logging.getLogger(__name__)

_applied = False


def apply_patches():
    global _applied
    if _applied:
        logger.info("LRCP patches already applied, skipping.")
        return

    from lrcp_vllm.patches.llava_patch import patch_llava
    from lrcp_vllm.patches.qwen2_5_vl_patch import patch_qwen2_5_vl
    from lrcp_vllm.patches.qwen3_vl_patch import patch_qwen3_vl
    from lrcp_vllm.patches.layer_pruning_patch import patch_layer_pruning

    patch_llava()
    logger.info("LRCP: LLaVA patches applied.")

    patch_qwen2_5_vl()
    logger.info("LRCP: Qwen2.5-VL patches applied.")

    patch_qwen3_vl()
    logger.info("LRCP: Qwen3-VL patches applied (if model exists).")

    patch_layer_pruning()
    logger.info("LRCP: Intermediate-layer pruning patches applied.")

    _applied = True
    logger.info("LRCP: All patches applied successfully.")


def unpatch_all():
    global _applied
    if not _applied:
        logger.info("LRCP patches not applied, skipping unpatch.")
        return

    from lrcp_vllm.patches.llava_patch import unpatch_llava
    from lrcp_vllm.patches.qwen2_5_vl_patch import unpatch_qwen2_5_vl
    from lrcp_vllm.patches.qwen3_vl_patch import unpatch_qwen3_vl
    from lrcp_vllm.patches.layer_pruning_patch import unpatch_layer_pruning

    unpatch_llava()
    unpatch_qwen2_5_vl()
    unpatch_qwen3_vl()
    unpatch_layer_pruning()

    _applied = False
    logger.info("LRCP: All patches removed.")
