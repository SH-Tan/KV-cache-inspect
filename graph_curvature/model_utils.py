"""Reusable input capture and graph-cache selection for decoder-only LLMs."""

import torch

from .curvature_utils.graph_relation import GRAPH


class _InputCaptured(Exception):
    pass


@torch.no_grad()
def capture_layer_inputs(model, tokenizer, prompts, max_length):
    """Capture real first-layer inputs without running the remaining model."""
    inputs = []

    def capture(module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        inputs.append(hidden.detach().cpu())
        raise _InputCaptured

    handle = model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=max_length)
            tokens = tokens.to(model.model.embed_tokens.weight.device)
            try:
                model.model(**tokens, use_cache=False)
            except _InputCaptured:
                pass
    finally:
        handle.remove()
    return inputs


def required_cache_names(target_ops, operations):
    names = set()
    for short in target_ops:
        names.add(short)
        relation = GRAPH[short]
        names.update(relation.get("prev", []))
        names.update(relation.get("next", []))
        names.update(relation.get("residual", []))
    # Gate/up shortest paths use the minimum cost across the two branches.
    names.update(("gate_proj", "up_proj"))
    return names.intersection(operations)
