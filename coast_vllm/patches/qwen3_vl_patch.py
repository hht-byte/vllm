import torch

from vllm.multimodal.coast import (
    compute_coast_retained_tokens_count,
    coast_prune_visual_tokens,
)


def patch_qwen3_vl_processor():
    try:
        from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
    except ImportError:
        return None

    original_get_prompt_updates = Qwen3VLMultiModalProcessor._get_prompt_updates

    def coast_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        coast_config = self.info.ctx.get_mm_config()
        if not coast_config.is_coast_enabled():
            return original_get_prompt_updates(
                self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs
            )

        return original_get_prompt_updates(
            self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs
        )

    Qwen3VLMultiModalProcessor._get_prompt_updates = coast_get_prompt_updates
    return original_get_prompt_updates


def patch_qwen3_vl_model():
    try:
        from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
    except ImportError:
        return None, None

    original_process_image_input = Qwen3VLForConditionalGeneration._process_image_input
    original_process_video_input = Qwen3VLForConditionalGeneration._process_video_input

    def coast_process_image_input(self, image_input):
        image_embeddings = original_process_image_input(self, image_input)

        coast_config = self.multimodal_config
        if not coast_config.is_coast_enabled():
            return image_embeddings

        retention_ratio = coast_config.coast_retention_ratio
        alpha_min = coast_config.coast_alpha_min
        alpha_max = coast_config.coast_alpha_max
        anchor_ratio = coast_config.coast_anchor_ratio

        compressed_list = []
        for emb in image_embeddings:
            original_tokens = emb.shape[0]
            num_retain = compute_coast_retained_tokens_count(
                original_tokens, retention_ratio
            )
            compressed, _ = coast_prune_visual_tokens(
                emb, num_retain, alpha_min, alpha_max, anchor_ratio
            )
            compressed_list.append(compressed)

        return tuple(compressed_list)

    def coast_process_video_input(self, video_input):
        video_embeddings = original_process_video_input(self, video_input)

        coast_config = self.multimodal_config
        if not coast_config.is_coast_enabled():
            return video_embeddings

        return video_embeddings

    Qwen3VLForConditionalGeneration._process_image_input = coast_process_image_input
    Qwen3VLForConditionalGeneration._process_video_input = coast_process_video_input
    return original_process_image_input, original_process_video_input


_original_qwen3_processor_method = None
_original_qwen3_model_methods = None


def patch_qwen3_vl():
    global _original_qwen3_processor_method, _original_qwen3_model_methods
    _original_qwen3_processor_method = patch_qwen3_vl_processor()
    _original_qwen3_model_methods = patch_qwen3_vl_model()


def unpatch_qwen3_vl():
    try:
        from vllm.model_executor.models.qwen3_vl import (
            Qwen3VLMultiModalProcessor,
            Qwen3VLForConditionalGeneration,
        )
    except ImportError:
        return

    if _original_qwen3_processor_method is not None:
        Qwen3VLMultiModalProcessor._get_prompt_updates = (
            _original_qwen3_processor_method
        )
    if _original_qwen3_model_methods is not None:
        if _original_qwen3_model_methods[0] is not None:
            Qwen3VLForConditionalGeneration._process_image_input = (
                _original_qwen3_model_methods[0]
            )
        if _original_qwen3_model_methods[1] is not None:
            Qwen3VLForConditionalGeneration._process_video_input = (
                _original_qwen3_model_methods[1]
            )
