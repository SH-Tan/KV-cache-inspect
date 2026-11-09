"""One curvature per last-query attention coefficient, for each prompt and layer."""

from contextlib import nullcontext
from multiprocessing import get_context
from types import SimpleNamespace

import numpy as np
import ot
import torch

from .curv_analysis_utils import finite_cost_statistics
from .curv_attention_utils import (
    _build_a_head_path_costs,
    _get_a_head_dims,
    _get_a_head_costs,
)
from .curv_distribution_utils import _build_node_distribution, _edge_distribution
from .curv_dtype_utils import curvature_np_dtype
from .curv_tensor_utils import _get_matrix_torch, adaptive_chunksize

# Fork workers inherit these read-only CPU arrays for one prompt.
_WORKER_STATE = None
CURVATURE_ALGORITHM_VERSION = 2


def _transport_cost(mu, nu, cost):
    """Solve OT on shortest-path distances; remaining infinities are unreachable pairs."""
    if mu[-1] == 0:
        mu, cost = mu[:-1], cost[:-1]
    if nu[-1] == 0:
        nu, cost = nu[:-1], cost[:, :-1]
    finite = np.isfinite(cost)
    if finite.all():
        distance, log = ot.emd2(mu, nu, cost, log=True, numThreads=1)
        return float(distance) if not log.get('warning') else np.nan
    if not finite.any():
        return np.nan

    # A residual transport cycle has at most m+n edges. With allowed costs
    # scaled to [0,1], m+n+1 excludes forbidden edges whenever transport is feasible.
    scale = max(float(cost[finite].max()), 1.0)
    bounded = np.full(cost.shape, sum(cost.shape) + 1.0, dtype=np.float64)
    bounded[finite] = cost[finite] / scale
    plan, log = ot.emd(mu, nu, bounded, log=True, numThreads=1)
    if log.get('warning') or plan[~finite].sum() > 1e-12:
        return np.nan
    return float(np.sum(plan[finite] * cost[finite], dtype=np.float64))


def _edge_cost_matrix_base(cost, prev_to_source, source_to_next, edge_cost):
    """Reuse the token's OT matrix, updating only the current edge's row/column."""
    cost[:-1, -1] = prev_to_source + edge_cost
    cost[-1, :-1] = source_to_next
    cost[-1, -1] = edge_cost
    return cost


def _sequence_task(seq_idx):
    """Build one token's path blocks locally, sharing the prompt's read-only inputs."""
    state = _WORKER_STATE
    layer_input, prev_dist, next_dist, head_costs, head_dim = state.path_inputs
    chunk_k, chunk_p = adaptive_chunksize(device=prev_dist.device)
    distribution = _build_node_distribution(
        layer_input[:, seq_idx:seq_idx + 1], 'layer_input', state.alpha,
    )
    mu, prev_active = _edge_distribution(distribution[0], state.alpha)
    prev_to_next, prev_to_source, source_to_next = _build_a_head_path_costs(
        prev_dist, next_dist, head_costs[:, seq_idx], head_dim,
        prev_active, state.next_active, chunk_k, chunk_p,
    )
    cost = np.empty((len(prev_active) + 1, len(state.next_active) + 1), dtype=curvature_np_dtype())
    cost[:-1, :-1] = prev_to_next
    return seq_idx, mu, cost, prev_to_source, source_to_next


def _compute_sequence(task):
    """Evaluate every attention head for one key token, returning one score per A."""
    state = _WORKER_STATE
    if state.cpu_paths:
        task = _sequence_task(task)
    seq_idx, mu, cost, prev_to_source, source_to_next = task
    scores = np.full(state.head_costs.shape[0], np.nan, dtype=curvature_np_dtype())
    valid = np.flatnonzero(state.valid_heads[:, seq_idx])
    # With zero center mass, only the shared X-to-O block contributes to OT.
    reuse_transport = state.alpha == 0 and cost.shape[0] > 1 and cost.shape[1] > 1
    transport = None

    for head in valid:
        edge_cost = float(state.head_costs[head, seq_idx])
        source_head = head // state.repeat
        _edge_cost_matrix_base(
            cost, prev_to_source[:, source_head], source_to_next[source_head], edge_cost,
        )
        if transport is None or not reuse_transport:
            transport = _transport_cost(mu, state.nu, cost)
        if np.isfinite(transport):
            scores[head] = (1.0 - transport / edge_cost) / (1.0 - state.alpha)

    count = len(valid)
    return seq_idx, scores, count, count * (cost.shape[0] - 1), count * len(state.next_active)


def compute_a_curvature(
    operations,
    layer_cache,
    alpha=0.0,
    device='cpu',
    workers=1,
):
    """Return [query_heads, key_tokens] curvature for this prompt's last query.

    Key s uses A[0, :, 0, s] and input neighbors x[s]. Each A connects a V-head
    feature set to a query-head feature set. Center masses are independent of
    V/Att_out values; distances to/from a set are minima over its features.
    Outgoing neighbors always belong to the final query. Numerical block helpers own
    the GQA mapping, distribution reduction, and sparse shortest-path costs.
    CPU workers build path costs and solve OT locally; only token indices and
    compact results cross the process queues. CUDA path costs stay in the parent.
    """
    global _WORKER_STATE
    try:
        num_heads, num_v_heads, head_dim = _get_a_head_dims(operations)
        head_costs, head_costs_np, valid_heads = _get_a_head_costs(operations, device=device)

        prev_cpu = _get_matrix_torch(layer_cache, 'v_proj', device='cpu').numpy()
        next_cpu = _get_matrix_torch(layer_cache, 'o_proj', device='cpu').numpy()
        prev_dist = torch.as_tensor(prev_cpu, device=device)
        next_dist = torch.as_tensor(next_cpu, device=device)
        
        next_distribution = _build_node_distribution(operations['o_proj'][:, -1:], 'o_proj', alpha)
        nu, next_active = _edge_distribution(next_distribution[0], alpha)
        
        _WORKER_STATE = SimpleNamespace(
            head_costs=head_costs_np, valid_heads=valid_heads,
            repeat=num_heads // num_v_heads,
            nu=nu, next_active=next_active, alpha=alpha,
            cpu_paths=prev_dist.device.type == 'cpu',
            path_inputs=(operations['layer_input'], prev_dist, next_dist, head_costs, head_dim),
        )

        curvature = np.full(head_costs_np.shape, np.nan, dtype=curvature_np_dtype())
        evaluations = in_neighbors = out_neighbors = 0
        seq_indices = np.flatnonzero(valid_heads.any(axis=0))
        if len(seq_indices):
            tasks = (seq_indices if _WORKER_STATE.cpu_paths else
                     (_sequence_task(seq_idx) for seq_idx in seq_indices))
            worker_count = min(workers, len(seq_indices))
            context = (get_context('fork').Pool(
                processes=worker_count, initializer=torch.set_num_threads, initargs=(1,),
            ) if worker_count > 1 else nullcontext())
            with context as pool:
                results = (map(_compute_sequence, tasks) if pool is None else
                           pool.imap_unordered(_compute_sequence, tasks, chunksize=1))
                for seq_idx, scores, count, in_count, out_count in results:
                    curvature[:, seq_idx] = scores
                    evaluations += count
                    in_neighbors += in_count
                    out_neighbors += out_count

        metrics = {
            'costs': {
                'prev': finite_cost_statistics(prev_cpu),
                'cur': finite_cost_statistics(head_costs_np),
                'next': finite_cost_statistics(next_cpu),
            },
            'edge_evaluations': evaluations,
            'in_neighbor_sum': in_neighbors,
            'out_neighbor_sum': out_neighbors,
        }
        return torch.from_numpy(curvature), metrics
    finally:
        _WORKER_STATE = None
