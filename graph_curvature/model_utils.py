"""Input capture and reusable V/O projection costs for the A-only runner."""

import torch

from .curvature_utils.curv_model_utils import _operation_distance_matrix_torch


class _InputCaptured(Exception):
    pass


@torch.no_grad()
def capture_layer_inputs(model, tokenizer, prompts, max_length=None):
    """Capture real first-layer inputs without running the remaining model."""
    inputs = []

    def capture(module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        inputs.append(hidden.detach().cpu())
        raise _InputCaptured

    handle = model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt", truncation=max_length is not None,
                               max_length=max_length)
            tokens = tokens.to(model.model.embed_tokens.weight.device)
            try:
                model.model(**tokens, use_cache=False)
            except _InputCaptured:
                pass
    finally:
        handle.remove()
    return inputs


def build_a_layer_cache(model, layer_id, device="cpu"):
    """Copy V/O costs once per layer; reuse them across all prompts and key tokens."""
    return {
        f"{name}__dist": _operation_distance_matrix_torch(model, name, layer_id, device).cpu()
        for name in ("v_proj", "o_proj")
    }
