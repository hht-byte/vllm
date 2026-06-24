import torch

from vllm.multimodal.lrcp import (
    compute_lrcp_retained_tokens_count,
    lrcp_compress,
    bool_mask_to_indices,
)


_original_forward_methods = {}


def patch_layer_pruning():
    lrcp_layer_patch_llama()
    lrcp_layer_patch_qwen2()


def lrcp_layer_patch_llama():
    try:
        from vllm.model_executor.models.llama import LlamaModel
    except ImportError:
        return

    original_forward = LlamaModel.forward

    def lrcp_forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds=None,
        **extra_layer_kwargs,
    ):
        from vllm.config.multimodal import MultiModalConfig
        from vllm.distributed.parallel_state import get_pp_group

        # Get LRCP config from the outer model
        lrcp_layer = _get_lrcp_layer_config(self)
        if lrcp_layer is None:
            return original_forward(
                self, input_ids, positions, intermediate_tensors,
                inputs_embeds=inputs_embeds, **extra_layer_kwargs,
            )

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        from itertools import islice

        aux_hidden_states = self._maybe_add_hidden_state(
            [], 0, hidden_states, residual
        )

        actual_start = self.start_layer
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(
                positions, hidden_states, residual, **extra_layer_kwargs
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

            # Apply LRCP at the specified intermediate layer
            if (actual_start + idx) == lrcp_layer:
                hidden_states, residual = _apply_layer_lrcp(
                    hidden_states, residual, input_ids, self,
                )

        if not get_pp_group().is_last_rank:
            from vllm.sequence import IntermediateTensors
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    LlamaModel.forward = lrcp_forward
    _original_forward_methods["LlamaModel"] = original_forward


def lrcp_layer_patch_qwen2():
    try:
        from vllm.model_executor.models.qwen2 import Qwen2Model
    except ImportError:
        return

    original_forward = Qwen2Model.forward

    def lrcp_forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
    ):
        lrcp_layer = _get_lrcp_layer_config(self)
        if lrcp_layer is None:
            return original_forward(
                self, input_ids, positions, intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

        from vllm.distributed.parallel_state import get_pp_group

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        from itertools import islice

        aux_hidden_states = self._maybe_add_hidden_state(
            [], 0, hidden_states, residual
        )

        actual_start = self.start_layer
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(
                positions, hidden_states, residual
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

            if (actual_start + idx) == lrcp_layer:
                hidden_states, residual = _apply_layer_lrcp(
                    hidden_states, residual, input_ids, self,
                )

        if not get_pp_group().is_last_rank:
            from vllm.sequence import IntermediateTensors
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    Qwen2Model.forward = lrcp_forward
    _original_forward_methods["Qwen2Model"] = original_forward


def _get_lrcp_layer_config(model):
    try:
        parent = model
        while hasattr(parent, "language_model"):
            parent = parent.language_model
        if hasattr(parent, "model") and parent.model is model:
            outer = parent
            while hasattr(outer, "vllm_config"):
                mm_config = outer.vllm_config.model_config.multimodal_config
                if mm_config.is_lrcp_enabled() and mm_config.lrcp_layer is not None:
                    return mm_config.lrcp_layer
                return None
    except Exception:
        pass

    try:
        from vllm.forward_context import get_forward_context
        ctx = get_forward_context()
        if hasattr(ctx, "vllm_config"):
            mm_config = ctx.vllm_config.model_config.multimodal_config
            if mm_config.is_lrcp_enabled() and mm_config.lrcp_layer is not None:
                return mm_config.lrcp_layer
    except Exception:
        pass

    return None


def _apply_layer_lrcp(hidden_states, residual, input_ids, model):
    lrcp_layer = _get_lrcp_layer_config(model)
    if lrcp_layer is None:
        return hidden_states, residual

    try:
        from vllm.forward_context import get_forward_context
        ctx = get_forward_context()
        mm_config = ctx.vllm_config.model_config.multimodal_config
    except Exception:
        return hidden_states, residual

    retention_ratio = mm_config.lrcp_retention_ratio
    subspace_dim = mm_config.lrcp_subspace_dim
    merge = mm_config.lrcp_merge

    # Identify multimodal token positions from input_ids
    # Use integer indices instead of boolean mask indexing
    # to avoid aclnnNonzeroV2 on NPU/Ascend
    if input_ids is not None:
        mm_token_ids = _get_mm_token_ids(model)
        if mm_token_ids:
            is_mm = torch.zeros_like(input_ids, dtype=torch.bool)
            for tid in mm_token_ids:
                is_mm |= (input_ids == tid)

            if is_mm.any():
                mm_indices = bool_mask_to_indices(is_mm)
                mm_hidden = hidden_states[mm_indices]
                num_retain = compute_lrcp_retained_tokens_count(
                    mm_hidden.shape[0], retention_ratio
                )
                compressed, top_indices = lrcp_compress(
                    mm_hidden, num_retain, subspace_dim, merge
                )
                retained_indices = mm_indices[top_indices]
                hidden_states[retained_indices] = compressed

                if residual is not None:
                    mm_residual = residual[mm_indices]
                    mm_residual_compressed, _ = lrcp_compress(
                        mm_residual, num_retain, subspace_dim, merge
                    )
                    residual[retained_indices] = mm_residual_compressed

    return hidden_states, residual


def _get_mm_token_ids(model):
    try:
        parent = model
        while hasattr(parent, "language_model"):
            parent = parent.language_model
        if hasattr(parent, "config"):
            config = parent.config
            ids = []
            if hasattr(config, "image_token_index"):
                ids.append(config.image_token_index)
            if hasattr(config, "vision_start_token_id"):
                ids.append(config.vision_start_token_id)
            if hasattr(config, "video_token_id"):
                ids.append(config.video_token_id)
            if hasattr(config, "image_token_id"):
                ids.append(config.image_token_id)
            return ids
    except Exception:
        pass
    return []


def unpatch_layer_pruning():
    try:
        from vllm.model_executor.models.llama import LlamaModel
        if "LlamaModel" in _original_forward_methods:
            LlamaModel.forward = _original_forward_methods["LlamaModel"]
    except ImportError:
        pass

    try:
        from vllm.model_executor.models.qwen2 import Qwen2Model
        if "Qwen2Model" in _original_forward_methods:
            Qwen2Model.forward = _original_forward_methods["Qwen2Model"]
    except ImportError:
        pass
