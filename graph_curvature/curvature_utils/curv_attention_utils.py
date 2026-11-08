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


def _get_a_feature_map(operations, device="cpu"):
    """Return GQA feature/head indices, V/output widths, and per-head width."""
    num_q_heads, num_v_heads, head_dim = _get_a_head_dims(operations)
    v_hidden_dim = num_v_heads * head_dim
    attn_out_hidden_dim = num_q_heads * head_dim
    repeat = num_q_heads // num_v_heads

    out_idx = torch.arange(attn_out_hidden_dim, device=device)
    q_head = out_idx // head_dim
    v_idx = (q_head // repeat) * head_dim + out_idx % head_dim
    return v_idx, q_head, v_hidden_dim, attn_out_hidden_dim, head_dim


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


def _get_a_curr_dist(operations, device="cpu", seq_idx=0):
    """Build [V features, Att_out features] costs for one key token of the last query."""
    v_idx, q_head, in_dim, out_dim, _ = _get_a_feature_map(operations, device=device)
    head_costs = _attention_costs(operations["A"][0, :, 0, seq_idx], device=device)
    curr_dist = torch.full(
        (in_dim, out_dim),
        float("inf"),
        dtype=curvature_torch_dtype(),
        device=device,
    )

    out_idx = torch.arange(q_head.numel(), device=device)
    curr_dist[v_idx, out_idx] = head_costs[q_head]
    return curr_dist


def _build_a_seq_path_costs(
    prev_dist, next_dist, head_costs, head_dim,
    prev_active, next_active, source_nodes, chunk_k=256, chunk_p=256,
):
    """Build only the required path blocks for one key token, exploiting sparse A.

    Each A output has exactly one V-feature source. Reduce over the repeated
    query heads first, then perform min-plus with the active input neighbors.
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
    prev_to_next = _min_plus_torch(
        prev_dist[prev_active], curr_to_next, chunk_k=chunk_k, chunk_p=chunk_p,
    )
    source_nodes = torch.as_tensor(source_nodes, dtype=torch.long, device=device)
    return (
        prev_to_next.cpu().numpy(),
        curr_to_next[source_nodes].cpu().numpy(),
    )


def _build_a_head_path_costs(
    prev_dist, next_dist, head_costs, head_dim,
    prev_active, next_active, chunk_k=256, chunk_p=256,
):
    """Keep exact X-to-O feature paths; use shortest distances to/from head sets.

    A head is a set of feature nodes, not a route between different features.
    Reduce boundary distances only after constructing the feature-level paths.
    """
    num_v_heads = prev_dist.shape[1] // head_dim
    source_nodes = np.arange(prev_dist.shape[1])
    prev_to_next, feature_to_next = _build_a_seq_path_costs(
        prev_dist, next_dist, head_costs, head_dim,
        prev_active, next_active, source_nodes, chunk_k, chunk_p,
    )
    prev_active = torch.as_tensor(prev_active, dtype=torch.long, device=prev_dist.device)
    prev_to_source = prev_dist[prev_active].reshape(
        prev_active.numel(), num_v_heads, head_dim,
    ).amin(dim=-1)
    source_to_next = feature_to_next.reshape(
        num_v_heads, head_dim, len(next_active),
    ).min(axis=1)
    return prev_to_next, prev_to_source.cpu().numpy(), source_to_next
