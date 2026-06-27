import torch

from vllm.multimodal.coast import (
    compute_coast_retained_tokens_count,
    coast_prune_visual_tokens,
)


def patch_llava_processor():
    from vllm.model_executor.models.llava import BaseLlavaMultiModalProcessor

    original_get_prompt_updates = BaseLlavaMultiModalProcessor._get_prompt_updates

    def coast_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        coast_config = self.info.ctx.get_mm_config()
        if not coast_config.is_coast_enabled():
            return original_get_prompt_updates(
                self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs
            )

        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index
        retention_ratio = coast_config.coast_retention_ratio

        def get_replacement(item_idx: int):
            from vllm.multimodal.parse import ImageEmbeddingItems, ImageProcessorItems

            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )
            if isinstance(images, ImageEmbeddingItems):
                original_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                original_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )
            num_tokens = compute_coast_retained_tokens_count(
                original_tokens, retention_ratio
            )
            return [image_token_id] * num_tokens

        from vllm.multimodal.processing.processor import PromptReplacement

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]

    BaseLlavaMultiModalProcessor._get_prompt_updates = coast_get_prompt_updates
    return original_get_prompt_updates


def patch_llava_model():
    from vllm.model_executor.models.llava import LlavaForConditionalGeneration

    original_process_image_input = LlavaForConditionalGeneration._process_image_input

    def coast_process_image_input(self, image_input):
        image_embeds = original_process_image_input(self, image_input)

        coast_config = self.multimodal_config
        if not coast_config.is_coast_enabled():
            return image_embeds

        retention_ratio = coast_config.coast_retention_ratio
        alpha_min = coast_config.coast_alpha_min
        alpha_max = coast_config.coast_alpha_max
        anchor_ratio = coast_config.coast_anchor_ratio

        if isinstance(image_embeds, torch.Tensor):
            original_tokens = image_embeds.shape[0]
            num_retain = compute_coast_retained_tokens_count(
                original_tokens, retention_ratio
            )
            compressed, _ = coast_prune_visual_tokens(
                image_embeds, num_retain, alpha_min, alpha_max, anchor_ratio
            )
            return compressed
        else:
            compressed_list = []
            for emb in image_embeds:
                original_tokens = emb.shape[0]
                num_retain = compute_coast_retained_tokens_count(
                    original_tokens, retention_ratio
                )
                compressed, _ = coast_prune_visual_tokens(
                    emb, num_retain, alpha_min, alpha_max, anchor_ratio
                )
                compressed_list.append(compressed)
            return tuple(compressed_list)

    LlavaForConditionalGeneration._process_image_input = coast_process_image_input
    return original_process_image_input


_original_llava_processor_method = None
_original_llava_model_method = None


def patch_llava():
    global _original_llava_processor_method, _original_llava_model_method
    _original_llava_processor_method = patch_llava_processor()
    _original_llava_model_method = patch_llava_model()


def unpatch_llava():
    from vllm.model_executor.models.llava import (
        BaseLlavaMultiModalProcessor,
        LlavaForConditionalGeneration,
    )

    if _original_llava_processor_method is not None:
        BaseLlavaMultiModalProcessor._get_prompt_updates = (
            _original_llava_processor_method
        )
    if _original_llava_model_method is not None:
        LlavaForConditionalGeneration._process_image_input = (
            _original_llava_model_method
        )
