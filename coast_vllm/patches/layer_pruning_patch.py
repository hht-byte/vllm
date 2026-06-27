import torch

from vllm.multimodal.coast import (
    compute_coast_retained_tokens_count,
    coast_prune_at_layer,
)

_original_forward_methods = {}


def patch_layer_pruning():
    coast_layer_patch_llama()
    coast_layer_patch_qwen2()


def coast_layer_patch_llama():
    try:
        from vllm.model_executor.models.llama import LlamaModel
    except ImportError:
        return

    original_forward = LlamaModel.forward

    def coast_forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds=None,
        **extra_layer_kwargs,
    ):
        coast_config = _get_coast_layer_config(self)
        if coast_config is None:
            return original_forward(
                self, input_ids, positions, intermediate_tensors,
                inputs_embeds=inputs_embeds, **extra_layer_kwargs,
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

        coast_layer = coast_config.coast_layer
        retention_ratio = coast_config.coast_retention_ratio
        alpha_min = coast_config.coast_alpha_min
        alpha_max = coast_config.coast_alpha_max
        anchor_ratio = coast_config.coast_anchor_ratio

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

            if (actual_start + idx) == coast_layer:
                is_visual = _identify_visual_tokens(input_ids, self)
                if is_visual is not None and is_visual.any():
                    true_hidden = (
                        hidden_states + residual
                        if residual is not None
                        else hidden_states
                    )
                    N_v = is_visual.sum().item()
                    num_retain = compute_coast_retained_tokens_count(
                        N_v, retention_ratio
                    )
                    true_hidden = coast_prune_at_layer(
                        true_hidden, is_visual, num_retain,
                        alpha_min, alpha_max, anchor_ratio,
                    )
                    if residual is not None:
                        hidden_states = true_hidden - residual
                    else:
                        hidden_states = true_hidden
                        residual = torch.zeros_like(hidden_states)

        if not get_pp_group().is_last_rank:
            from vllm.sequence import IntermediateTensors
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    LlamaModel.forward = coast_forward
    _original_forward_methods["LlamaModel"] = original_forward


def coast_layer_patch_qwen2():
    try:
        from vllm.model_executor.models.qwen2 import Qwen2Model
    except ImportError:
        return

    original_forward = Qwen2Model.forward

    def coast_forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
    ):
        coast_config = _get_coast_layer_config(self)
        if coast_config is None:
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

        coast_layer = coast_config.coast_layer
        retention_ratio = coast_config.coast_retention_ratio
        alpha_min = coast_config.coast_alpha_min
        alpha_max = coast_config.coast_alpha_max
        anchor_ratio = coast_config.coast_anchor_ratio

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

            if (actual_start + idx) == coast_layer:
                is_visual = _identify_visual_tokens(input_ids, self)
                if is_visual is not None and is_visual.any():
                    true_hidden = (
                        hidden_states + residual
                        if residual is not None
                        else hidden_states
                    )
                    N_v = is_visual.sum().item()
                    num_retain = compute_coast_retained_tokens_count(
                        N_v, retention_ratio
                    )
                    true_hidden = coast_prune_at_layer(
                        true_hidden, is_visual, num_retain,
                        alpha_min, alpha_max, anchor_ratio,
                    )
                    if residual is not None:
                        hidden_states = true_hidden - residual
                    else:
                        hidden_states = true_hidden
                        residual = torch.zeros_like(hidden_states)

        if not get_pp_group().is_last_rank:
            from vllm.sequence import IntermediateTensors
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    Qwen2Model.forward = coast_forward
    _original_forward_methods["Qwen2Model"] = original_forward


def _get_coast_layer_config(model):
    try:
        parent = model
        while hasattr(parent, "language_model"):
            parent = parent.language_model
        if hasattr(parent, "model") and parent.model is model:
            outer = parent
            while hasattr(outer, "vllm_config"):
                mm_config = outer.vllm_config.model_config.multimodal_config
                if mm_config.is_coast_enabled():
                    return mm_config
                return None
    except Exception:
        pass

    try:
        from vllm.forward_context import get_forward_context
        ctx = get_forward_context()
        if hasattr(ctx, "vllm_config"):
            mm_config = ctx.vllm_config.model_config.multimodal_config
            if mm_config.is_coast_enabled():
                return mm_config
    except Exception:
        pass

    return None


def _identify_visual_tokens(input_ids, model):
    if input_ids is None:
        return None

    mm_token_ids = _get_mm_token_ids(model)
    if not mm_token_ids:
        return None

    is_visual = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    for tid in mm_token_ids:
        is_visual |= (input_ids == tid)

    return is_visual


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
