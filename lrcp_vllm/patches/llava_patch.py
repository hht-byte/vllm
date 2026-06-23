import torch

from vllm.multimodal.lrcp import compute_lrcp_retained_tokens_count, lrcp_compress


def patch_llava_processor():
    from vllm.model_executor.models.llava import BaseLlavaMultiModalProcessor

    original_get_prompt_updates = BaseLlavaMultiModalProcessor._get_prompt_updates

    def lrcp_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        lrcp_config = self.info.ctx.get_mm_config()
        if not lrcp_config.is_lrcp_enabled():
            return original_get_prompt_updates(
                self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs
            )

        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index
        retention_ratio = lrcp_config.lrcp_retention_ratio

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )
            from vllm.multimodal.parse import ImageEmbeddingItems, ImageProcessorItems

            if isinstance(images, ImageEmbeddingItems):
                original_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                original_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

            num_tokens = compute_lrcp_retained_tokens_count(
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

    BaseLlavaMultiModalProcessor._get_prompt_updates = lrcp_get_prompt_updates
    return original_get_prompt_updates


def patch_llava_model():
    from vllm.model_executor.models.llava import LlavaForConditionalGeneration

    original_process_image_input = LlavaForConditionalGeneration._process_image_input

    def lrcp_process_image_input(self, image_input):
        image_embeds = original_process_image_input(self, image_input)

        lrcp_config = self.multimodal_config
        if not lrcp_config.is_lrcp_enabled():
            return image_embeds

        retention_ratio = lrcp_config.lrcp_retention_ratio
        subspace_dim = lrcp_config.lrcp_subspace_dim
        merge = lrcp_config.lrcp_merge

        if isinstance(image_embeds, torch.Tensor):
            original_tokens = image_embeds.shape[0]
            num_retain = compute_lrcp_retained_tokens_count(
                original_tokens, retention_ratio
            )
            compressed, _ = lrcp_compress(
                image_embeds, num_retain, subspace_dim, merge
            )
            return compressed
        else:
            compressed_list = []
            for emb in image_embeds:
                original_tokens = emb.shape[0]
                num_retain = compute_lrcp_retained_tokens_count(
                    original_tokens, retention_ratio
                )
                compressed, _ = lrcp_compress(emb, num_retain, subspace_dim, merge)
                compressed_list.append(compressed)
            return tuple(compressed_list)

    LlavaForConditionalGeneration._process_image_input = lrcp_process_image_input
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
        BaseLlavaMultiModalProcessor._get_prompt_updates = _original_llava_processor_method
    if _original_llava_model_method is not None:
        LlavaForConditionalGeneration._process_image_input = _original_llava_model_method
