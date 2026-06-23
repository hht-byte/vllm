import torch

from vllm.multimodal.lrcp import compute_lrcp_retained_tokens_count, lrcp_compress
from vllm.multimodal.evs import compute_mrope_for_media


def patch_qwen2_5_vl_processor():
    from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLMultiModalProcessor

    original_get_prompt_updates = Qwen2_5_VLMultiModalProcessor._get_prompt_updates

    def lrcp_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        lrcp_config = self.info.ctx.get_mm_config()
        if not lrcp_config.is_lrcp_enabled():
            return original_get_prompt_updates(
                self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs
            )

        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        placeholder = {
            "image": vocab[hf_processor.image_token],
            "video": vocab[hf_processor.video_token],
        }
        merge_length = image_processor.merge_size ** 2
        retention_ratio = lrcp_config.lrcp_retention_ratio

        from functools import partial
        from vllm.multimodal.processing.processor import PromptReplacement

        def get_replacement_qwen2vl(item_idx: int, modality: str):
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens_original = int(grid_thw.prod()) // merge_length

            video_pruning_rate = lrcp_config.video_pruning_rate
            if (
                modality == "video"
                and video_pruning_rate is not None
                and video_pruning_rate > 0.0
            ):
                from vllm.multimodal.evs import compute_retained_tokens_count
                T, H, W = map(int, grid_thw)
                tokens_per_frame = (H // image_processor.merge_size) * (
                    W // image_processor.merge_size
                )
                num_tokens = compute_retained_tokens_count(
                    tokens_per_frame, T, video_pruning_rate,
                )
            else:
                num_tokens = compute_lrcp_retained_tokens_count(
                    num_tokens_original, retention_ratio
                )

            return [placeholder[modality]] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement_qwen2vl, modality=modality),
            )
            for modality in ("image", "video")
        ]

    Qwen2_5_VLMultiModalProcessor._get_prompt_updates = lrcp_get_prompt_updates
    return original_get_prompt_updates


def patch_qwen2_5_vl_model():
    from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

    original_process_image_input = Qwen2_5_VLForConditionalGeneration._process_image_input
    original_process_video_input = Qwen2_5_VLForConditionalGeneration._process_video_input

    def lrcp_process_image_input(self, image_input):
        image_embeddings = original_process_image_input(self, image_input)

        lrcp_config = self.multimodal_config
        if not lrcp_config.is_lrcp_enabled():
            return image_embeddings

        retention_ratio = lrcp_config.lrcp_retention_ratio
        subspace_dim = lrcp_config.lrcp_subspace_dim
        merge = lrcp_config.lrcp_merge

        grid_thw = image_input["image_grid_thw"]
        merge_size = self.visual.spatial_merge_size

        compressed_list = []
        for emb, thw in zip(image_embeddings, grid_thw.tolist()):
            original_tokens = emb.shape[0]
            num_retain = compute_lrcp_retained_tokens_count(
                original_tokens, retention_ratio
            )
            compressed, _ = lrcp_compress(emb, num_retain, subspace_dim, merge)

            if self.is_multimodal_pruning_enabled:
                positions = compute_mrope_for_media(
                    thw, merge_size,
                ).to(emb.device)
                positions = positions[:original_tokens]
                _, retention_mask = lrcp_compress(
                    emb, num_retain, subspace_dim, merge=False
                )
                positions = positions[retention_mask]
                compressed = torch.cat([compressed, positions], dim=1)

            compressed_list.append(compressed)

        return tuple(compressed_list)

    def lrcp_process_video_input(self, video_input):
        video_embeddings = original_process_video_input(self, video_input)

        lrcp_config = self.multimodal_config
        if not lrcp_config.is_lrcp_enabled():
            return video_embeddings

        retention_ratio = lrcp_config.lrcp_retention_ratio
        subspace_dim = lrcp_config.lrcp_subspace_dim
        merge = lrcp_config.lrcp_merge

        grid_thw = video_input["video_grid_thw"]
        merge_size = self.visual.spatial_merge_size

        second_per_grid_ts = video_input.get("second_per_grid_ts")
        tokens_per_second = self.config.vision_config.tokens_per_second

        compressed_list = []
        for emb_idx, (emb, thw) in enumerate(zip(video_embeddings, grid_thw.tolist())):
            original_tokens = emb.shape[0]

            if self.is_multimodal_pruning_enabled and self.video_pruning_rate is not None:
                from vllm.multimodal.evs import compute_retention_mask
                evs_mask = compute_retention_mask(
                    emb, thw,
                    spatial_merge_size=self.visual.spatial_merge_size,
                    q=self.video_pruning_rate,
                )
                emb = emb[evs_mask]
                original_tokens_retained = emb.shape[0]
                num_retain = compute_lrcp_retained_tokens_count(
                    original_tokens_retained, retention_ratio
                )
                compressed, retention_mask = lrcp_compress(
                    emb, num_retain, subspace_dim, merge
                )

                video_second_per_grid_t = (
                    second_per_grid_ts[emb_idx].item()
                    if second_per_grid_ts is not None else 1.0
                )
                positions = compute_mrope_for_media(
                    thw, merge_size,
                    tokens_per_second=tokens_per_second,
                    video_second_per_grid=video_second_per_grid_t,
                ).to(emb.device)
                positions = positions[:original_tokens][evs_mask]
                positions = positions[retention_mask]
                compressed = torch.cat([compressed, positions], dim=1)
            else:
                num_retain = compute_lrcp_retained_tokens_count(
                    original_tokens, retention_ratio
                )
                compressed, retention_mask = lrcp_compress(
                    emb, num_retain, subspace_dim, merge
                )

                if self.is_multimodal_pruning_enabled:
                    positions = compute_mrope_for_media(
                        thw, merge_size,
                        tokens_per_second=tokens_per_second,
                        video_second_per_grid=(
                            second_per_grid_ts[emb_idx].item()
                            if second_per_grid_ts is not None else 1.0
                        ),
                    ).to(emb.device)
                    positions = positions[:original_tokens][retention_mask]
                    compressed = torch.cat([compressed, positions], dim=1)

            compressed_list.append(compressed)

        return tuple(compressed_list)

    Qwen2_5_VLForConditionalGeneration._process_image_input = lrcp_process_image_input
    Qwen2_5_VLForConditionalGeneration._process_video_input = lrcp_process_video_input
    return original_process_image_input, original_process_video_input


_original_qwen_processor_method = None
_original_qwen_model_methods = None


def patch_qwen2_5_vl():
    global _original_qwen_processor_method, _original_qwen_model_methods
    _original_qwen_processor_method = patch_qwen2_5_vl_processor()
    _original_qwen_model_methods = patch_qwen2_5_vl_model()


def unpatch_qwen2_5_vl():
    from vllm.model_executor.models.qwen2_5_vl import (
        Qwen2_5_VLMultiModalProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )

    if _original_qwen_processor_method is not None:
        Qwen2_5_VLMultiModalProcessor._get_prompt_updates = _original_qwen_processor_method
    if _original_qwen_model_methods is not None:
        Qwen2_5_VLForConditionalGeneration._process_image_input = _original_qwen_model_methods[0]
        Qwen2_5_VLForConditionalGeneration._process_video_input = _original_qwen_model_methods[1]
