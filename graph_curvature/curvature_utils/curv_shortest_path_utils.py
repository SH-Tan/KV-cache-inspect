import torch
import numpy as np

from graph_curvature.curvature_utils.curv_attention_utils import _get_a_curr_dist
from graph_curvature.curvature_utils.graph_relation import _resolve_graph_sets
from graph_curvature.curvature_utils.curv_distribution_utils import _min_reduce_blocks
from graph_curvature.curvature_utils.curv_tensor_utils import (
    _all_cost_matrices,
    _get_matrix_torch,
    _get_qk_next_cost,
    adaptive_chunksize,
    _min_plus_torch,
)

SP_CACHE = {}


def _gate_up_shortest_path_dist(layer_cache, short_name, curr_dist, device):
    if short_name not in {"gate_proj", "up_proj"}:
        return curr_dist

    other_name = "up_proj" if short_name == "gate_proj" else "gate_proj"
    other_dist = _get_matrix_torch(layer_cache, other_name, device=device)
    if other_dist is None or other_dist.shape != curr_dist.shape:
        return curr_dist
    return torch.minimum(curr_dist, other_dist)


def _residual_to_curr_out(residual_matrix, curr_dist, chunk_k=256, chunk_p=256):
    if (
        residual_matrix.shape[0] == curr_dist.shape[0]
        and residual_matrix.shape[1] == curr_dist.shape[0]
    ):
        return curr_dist + torch.diagonal(residual_matrix).reshape(-1, 1)
    return _min_plus_torch(residual_matrix, curr_dist, chunk_k=chunk_k, chunk_p=chunk_p)


def build_shortest_path_cache(
    operations,
    layer_cache,
    short_name,
    sp_cache=None,
    device="cuda",
    graph_data=None,
    model_meta=None,
    include_qk_next=True,
    seq_idx=0,
):
    """
    Cache is flat: sp_cache[short_name] = ...
    If you want to rebuild for the next layer, clear sp_cache at the layer boundary.
    """
    if sp_cache is None:
        sp_cache = SP_CACHE

    if graph_data is None:
        graph_data = _resolve_graph_sets(operations, short_name)

    if (short_name not in {"q_proj", "k_proj", "A"}) and (short_name in sp_cache):
        return sp_cache[short_name], graph_data

    # Current costs have shape [in_features, out_features].
    if short_name == "A":
        curr_dist = _get_a_curr_dist(operations, device=device, seq_idx=seq_idx)
    else:
        curr_dist = _get_matrix_torch(layer_cache, short_name, device=device)
    if curr_dist is None:
        return None, graph_data

    # For A, the prev dists is W_v
    prev_dists = _all_cost_matrices(layer_cache, graph_data["prev_cost_names"], device=device)

    if short_name in {"q_proj", "k_proj"} and include_qk_next:
            cost_n = graph_data["next_cost_names"][0]
            cost = operations.get(cost_n)
            if model_meta is None:
                raise ValueError(f"model_meta is required for {short_name}")
            next_dists = {
                name: _get_qk_next_cost(cost, short_name, model_meta, device=device)
                for name in graph_data["next_cost_names"]
            }
    # A will go into this branch, next is W_o, prev is W_v
    else:
        next_names = [] if short_name in {"q_proj", "k_proj"} else graph_data["next_cost_names"]
        next_dists = _all_cost_matrices(layer_cache, next_names, device=device)

    chunk_k, chunk_p = adaptive_chunksize(device=device)

    curr_dist_np = curr_dist.cpu().contiguous().numpy()
    curr_path_dist = _gate_up_shortest_path_dist(layer_cache, short_name, curr_dist, device)
    curr_out_to_next_np = (
        _min_reduce_blocks([v.cpu().contiguous().numpy() for v in next_dists.values()])
        if next_dists
        else None
    )

    prev_to_curr_out_all = {}
    curr_in_to_next_all = {}
    prev_to_next_all = {}
    prev_to_curr_in_np = None

    if residual_dists:
        prev_in_matrices = list(prev_dists.values()) + list(residual_dists.values())
        if prev_in_matrices:
            total_prev_in_rows = sum(int(matrix.shape[0]) for matrix in prev_in_matrices)
            prev_to_curr_in_np = np.empty(
                (total_prev_in_rows, curr_dist.shape[0]),
                dtype=curr_dist_np.dtype,
            )
            row_start = 0
            for matrix in prev_in_matrices:
                row_end = row_start + matrix.shape[0]
                prev_to_curr_in_np[row_start:row_end] = matrix.cpu().contiguous().numpy()
                row_start = row_end
        del prev_in_matrices
    else:
        prev_to_curr_in_np = (
            _min_reduce_blocks([v.cpu().contiguous().numpy() for v in prev_dists.values()])
            if prev_dists
            else None
        )

    for name, next_matrix in next_dists.items():
        curr_in_to_next = _min_plus_torch(
            curr_path_dist, next_matrix, chunk_k=chunk_k, chunk_p=chunk_p
        )
        curr_in_to_next_all[name] = curr_in_to_next.cpu().contiguous().numpy()
        del curr_in_to_next

    if residual_dists:
        total_prev_rows = sum(
            int(prev_matrix.shape[0]) for prev_matrix in prev_dists.values()
        ) + sum(
            int(residual_matrix.shape[0]) for residual_matrix in residual_dists.values()
        )
        prev_to_curr_out = torch.empty(
            (total_prev_rows, curr_dist.shape[1]),
            dtype=curr_dist.dtype,
            device=curr_dist.device,
        )
        row_start = 0
        for prev_matrix in prev_dists.values():
            block = _min_plus_torch(
                prev_matrix, curr_path_dist, chunk_k=chunk_k, chunk_p=chunk_p
            )
            row_end = row_start + block.shape[0]
            prev_to_curr_out[row_start:row_end] = block
            row_start = row_end
            del block
        for residual_matrix in residual_dists.values():
            block = _residual_to_curr_out(
                residual_matrix, curr_path_dist, chunk_k=chunk_k, chunk_p=chunk_p
            )
            row_end = row_start + block.shape[0]
            prev_to_curr_out[row_start:row_end] = block
            row_start = row_end
            del block
        for next_name, next_matrix in next_dists.items():
            key = f"prev_with_residual->{next_name}"
            prev_to_next = _min_plus_torch(
                prev_to_curr_out, next_matrix, chunk_k=chunk_k, chunk_p=chunk_p
            )
            prev_to_next_all[key] = prev_to_next.cpu().contiguous().numpy()
            del prev_to_next
        prev_to_curr_out_all["prev_with_residual"] = prev_to_curr_out.cpu().contiguous().numpy()
        del prev_to_curr_out
    else:
        for prev_name, prev_matrix in prev_dists.items():
            prev_to_curr_out = _min_plus_torch(
                prev_matrix, curr_path_dist, chunk_k=chunk_k, chunk_p=chunk_p
            )
            for next_name, next_matrix in next_dists.items():
                key = f"{prev_name}->{next_name}"
                prev_to_next = _min_plus_torch(
                    prev_to_curr_out, next_matrix, chunk_k=chunk_k, chunk_p=chunk_p
                )
                prev_to_next_all[key] = prev_to_next.cpu().contiguous().numpy()
                del prev_to_next
            prev_to_curr_out_all[prev_name] = prev_to_curr_out.cpu().contiguous().numpy()
            del prev_to_curr_out

    del prev_dists, residual_dists
    if curr_path_dist is not curr_dist:
        del curr_path_dist
    del curr_dist
    del next_dists
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    sp = {
        "curr_dist": curr_dist_np,
        "prev_to_curr_in": prev_to_curr_in_np,
        "curr_out_to_next": curr_out_to_next_np,
        "prev_to_curr_out_all": prev_to_curr_out_all,
        "curr_in_to_next_all": curr_in_to_next_all,
        "prev_to_next_all": prev_to_next_all,
    }

    sp_cache[short_name] = sp
    return sp, graph_data
