import numpy as np
import torch

from graph_curvature.curvature_utils.curv_dtype_utils import curvature_torch_dtype
from graph_curvature.curvature_utils.curv_tensor_utils import _min_plus_torch


def _get_a_head_dims(operations):
    """Return query/V head counts and head width, checking the GQA layout."""
    v_hidden_dim = operations["v_proj"].shape[-1]
    attn_out_hidden_dim = operations["Att_out"].shape[-1]
    num_q_heads = operations["A"].shape[1]

    assert attn_out_hidden_dim % num_q_heads == 0
    head_dim = attn_out_hidden_dim // num_q_heads
    assert v_hidden_dim % head_dim == 0

    num_v_heads = v_hidden_dim // head_dim
    assert num_q_heads % num_v_heads == 0
    return num_q_heads, num_v_heads, head_dim


def _attention_costs(attention, device="cpu"):
    return attention.detach().to(
        device=device, dtype=curvature_torch_dtype(), copy=True,
    ).abs_().reciprocal_()


def _get_a_head_costs(operations, device="cpu"):
    """Return final-query attention costs, their CPU array, and valid-head mask."""
    head_costs = _attention_costs(operations["A"][0, :, 0], device=device)
    head_costs_np = head_costs.cpu().numpy()
    valid_heads = np.isfinite(head_costs_np) & (head_costs_np > 0)
    return head_costs, head_costs_np, valid_heads


def _build_a_head_path_costs(
    prev_dist, next_dist, head_costs, head_dim,
    prev_active, next_active, chunk_k=256, chunk_p=256,
):
    """Exact sparse min-plus DP for fixed neighbors: A→O, then X→A→O.

    Each A output has exactly one V-feature source. Reduce over the repeated
    query heads first, then perform min-plus with the active input neighbors.
    Reduce features to head sets only at the boundaries, after complete paths
    have been composed. Infinite edges never block alternative finite paths.
    """
    num_v_heads = prev_dist.shape[1] // head_dim
    repeat = head_costs.numel() // num_v_heads
    device = prev_dist.device
    next_active = torch.as_tensor(next_active, dtype=torch.long, device=device)
    next_cost = next_dist[:, next_active]
    next_cost.add_(head_costs.repeat_interleave(head_dim).unsqueeze(1))
    curr_to_next = next_cost.reshape(
        num_v_heads, repeat, head_dim, next_active.numel(),
    ).amin(dim=1).reshape(prev_dist.shape[1], next_active.numel())
    del next_cost

    prev_active = torch.as_tensor(prev_active, dtype=torch.long, device=device)
    prev_cost = prev_dist[prev_active]
    prev_to_next = _min_plus_torch(
        prev_cost, curr_to_next, chunk_k=chunk_k, chunk_p=chunk_p,
    )
    prev_to_source = prev_cost.reshape(
        prev_active.numel(), num_v_heads, head_dim,
    ).amin(dim=-1)
    source_to_next = curr_to_next.reshape(
        num_v_heads, head_dim, len(next_active),
    ).amin(dim=1)
    return (prev_to_next.cpu().numpy(), prev_to_source.cpu().numpy(),
            source_to_next.cpu().numpy())
