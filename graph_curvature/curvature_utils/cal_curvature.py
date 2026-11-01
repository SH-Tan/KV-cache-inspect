import numpy as np
import torch
import ot
import os

import graph_curvature.curvature_utils.curv_analysis_utils as analysis_utils
from graph_curvature.curvature_utils.curv_dtype_utils import curvature_np_dtype, curvature_torch_dtype
from graph_curvature.curvature_utils.curv_filter_utils import sliding_median_low_pass
from graph_curvature.curvature_utils.curv_distribution_utils import (
    _build_node_distribution,
    _build_distribution_node_values,
    _build_node_distribution_row_from_values,
    _build_qk_out_node_distribution,
    _edge_distribution,
    _min_reduce_blocks,
)

from graph_curvature.curvature_utils.curv_sequence_utils import (
    _build_att_out_to_o_cost,
    _build_oproj_to_att_in_value_map,
    _build_vproj_to_att_out_cost,
    _build_vproj_to_att_out_value_map,
    _precompute_oproj_prev_distributions,
    _precompute_vproj_next_distributions,
)
from graph_curvature.curvature_utils.curv_shortest_path_utils import build_shortest_path_cache
from graph_curvature.curvature_utils.curv_shared_utils import _from_shared_numpy, _to_shared_numpy, _load_worker_seq_distribution, _to_shared_seq_metas
import graph_curvature.curvature_utils.curv_metric_utils as metric_utils
from graph_curvature.curvature_utils.curv_tensor_utils import _build_v_to_att_out_template, _build_x_to_out_cost

from multiprocessing import get_context
import time

proc = int(os.environ.get("CURV_WORKERS", "1"))

_SHARED_CURR_DIST = None
_SHARED_PREV_IN = None
_SHARED_NEXT_OUT = None
_SHARED_SP = None
_SHARED_ALPHA = 0.0
_A = None
_Q_to_A = None
_V_COST = None
_SPK = None
_SHARED_SOURCE_NODE_VALUES = None
_SHARED_TARGET_NODE_VALUES = None
_SHARED_GATE_BETA_VALUES = None
_SHARED_PREV_NODE_VALUES = None
_SHARED_NEXT_NODE_VALUES = None
_SHARED_RESIDUAL_NODE_META = None
_SHARED_RESIDUAL_STARTS = None
_SHARED_RESIDUAL_ENDS = None
_SHARED_OT_PREV_SCORE = None
_SHARED_OT_NEXT_SCORE = None

_SHARED_SHORT_NAME = None
_SHARED_SAMPLE_IDX = None
_SHARED_MODEL_META = None
_SHARED_SEQ_LEN = 1
_SHARED_TOP_K = 5
_SHARED_SEQ_SELECT = "top"
_SHARED_LPF_WINDOW = 0
_SHARED_SAVE_PARAMETER_LOGS = False
_WORKER_CURR_DIST_SHM = None
_WORKER_PREV_IN_SHM = None
_WORKER_NEXT_OUT_SHM = None
_WORKER_SOURCE_NODE_SHM = None
_WORKER_TARGET_NODE_SHM = None
_WORKER_GATE_BETA_SHM = None
_WORKER_PREV_NODE_SHM = None
_WORKER_NEXT_NODE_SHM = None
_WORKER_SCORE_SHMS = []
_WORKER_PREV_SEQ_METAS = None
_WORKER_NEXT_SEQ_METAS = None
_WORKER_PREV_SEQ_SHMS = {}
_WORKER_NEXT_SEQ_SHMS = {}
_WORKER_PREV_SEQ_CACHE = {}
_WORKER_NEXT_SEQ_CACHE = {}


def _as_index_array(active):
    if active is None:
        return np.empty((0,), dtype=np.int64)
    return np.asarray(active, dtype=np.int64).reshape(-1)


def _apply_residual_source_cost(cost, prev_active, u_idx):
    if cost is None or _SHARED_RESIDUAL_STARTS is None:
        return cost
    prev_active = _as_index_array(prev_active)
    if prev_active.size == 0:
        return cost

    residual_rows = np.zeros(prev_active.shape, dtype=bool)
    for start, end in zip(_SHARED_RESIDUAL_STARTS, _SHARED_RESIDUAL_ENDS):
        rows = (prev_active >= start) & (prev_active < end)
        residual_rows |= rows & ((prev_active - start) == int(u_idx))
    if np.any(residual_rows):
        row_idx = np.nonzero(residual_rows)[0]
        if cost.shape[1] > 1:
            cost[row_idx, :-1] = 1.0 + cost[-1, :-1]
        cost[row_idx, -1] = 1.0 + cost[-1, -1]
    return cost


def _edge_cost_matrix_base(u_idx, v_idx, prev_active, next_active, sp_uv):
    """
    Static seq-local cost matrix.
    """
    prev_active = _as_index_array(prev_active)
    next_active = _as_index_array(next_active)

    prev_count = int(len(prev_active))
    next_count = int(len(next_active))

    cost = np.full((prev_count + 1, next_count + 1), np.inf, dtype=curvature_np_dtype())
    sp = _SHARED_SP

    prev_to_next_all = sp.get("prev_to_next_all", {})
    prev_to_curr_out_all = sp.get("prev_to_curr_out_all", {})
    curr_in_to_next_all = sp.get("curr_in_to_next_all", {})

    if prev_count > 0 and next_count > 0 and prev_to_next_all:
        block = _min_reduce_blocks([
            matrix[np.ix_(prev_active, next_active)]
            for matrix in prev_to_next_all.values()
        ])
        if block is not None:
            cost[:-1, :-1] = block

    if prev_count > 0 and prev_to_curr_out_all:
        block = _min_reduce_blocks([
            matrix[prev_active, v_idx]
            for matrix in prev_to_curr_out_all.values()
        ])
        if block is not None:
            cost[:-1, -1] = block

    if next_count > 0 and curr_in_to_next_all:
        block = _min_reduce_blocks([
            matrix[u_idx, next_active]
            for matrix in curr_in_to_next_all.values()
        ])
        if block is not None:
            cost[-1, :-1] = block

    cost[-1, -1] = sp_uv

    return _apply_residual_source_cost(cost, prev_active, u_idx)


def _seq_node_value(node_values, seq_idx, node_idx):
    if node_values is None:
        return None
    if seq_idx >= node_values.shape[0] or node_idx >= node_values.shape[1]:
        return None
    return float(node_values[int(seq_idx), int(node_idx)])


def _source_node_value(seq_idx, u_idx):
    return _seq_node_value(_SHARED_SOURCE_NODE_VALUES, seq_idx, u_idx)


def _target_node_value(seq_idx, v_idx):
    return _seq_node_value(_SHARED_TARGET_NODE_VALUES, seq_idx, v_idx)


def _gate_beta_value(seq_idx, v_idx):
    return _seq_node_value(_SHARED_GATE_BETA_VALUES, seq_idx, v_idx)


def _out_neighbor_node_values(seq_idx, next_active, nu, cost):
    if _SHARED_NEXT_NODE_VALUES is None or len(next_active) == 0:
        return None
    next_probs = np.asarray(nu[:-1], dtype=curvature_np_dtype()).reshape(-1)
    next_cost = np.asarray(cost[-1, :-1], dtype=curvature_np_dtype()).reshape(-1)
    values = []
    for pos, node_idx in enumerate(np.asarray(next_active, dtype=np.int64).reshape(-1)):
        node_idx = int(node_idx)
        values.append({
            "node_idx": node_idx,
            "raw_value": _seq_node_value(_SHARED_NEXT_NODE_VALUES, seq_idx, node_idx),
            "normalized_probability": float(next_probs[pos]),
            "cost_from_source": float(next_cost[pos]),
        })
    return values


def _weight_magnitude_from_cost(cost_value):
    if cost_value is None or not np.isfinite(cost_value) or cost_value <= 0.0:
        return 0.0
    return float(1.0 / cost_value)


def _mu_node_values(seq_idx, u_idx, prev_active, mu):
    if _SHARED_PREV_NODE_VALUES is None or len(prev_active) == 0:
        return None

    prev_probs = np.asarray(mu[:-1], dtype=curvature_np_dtype()).reshape(-1)
    prev_to_curr_in = _SHARED_SP.get("prev_to_curr_in") if _SHARED_SP is not None else None
    values = []
    for pos, node_idx in enumerate(np.asarray(prev_active, dtype=np.int64).reshape(-1)):
        node_idx = int(node_idx)
        raw_value = _seq_node_value(_SHARED_PREV_NODE_VALUES, seq_idx, node_idx)
        cost_to_u = None
        if prev_to_curr_in is not None:
            cost_to_u = float(prev_to_curr_in[node_idx, int(u_idx)])
        weight_magnitude = _weight_magnitude_from_cost(cost_to_u)
        node_times_weight = None if raw_value is None else float(raw_value * weight_magnitude)
        probability = float(prev_probs[pos])
        item = {
            "node_idx": node_idx,
            "raw_value": raw_value,
            "normalized_probability": probability,
            "cost_to_u": cost_to_u,
            "original_weight_magnitude_to_u": weight_magnitude,
            "node_times_weight_magnitude": node_times_weight,
        }
        if _SHARED_RESIDUAL_NODE_META:
            for meta in _SHARED_RESIDUAL_NODE_META:
                start = int(meta["start"])
                end = int(meta["end"])
                if start <= node_idx < end:
                    item["residual_name"] = meta["name"]
                    item["residual_local_idx"] = node_idx - start
                    break
        if node_times_weight is not None:
            item["probability_times_node_weight"] = float(probability * node_times_weight)
        values.append(item)
    return values


def _with_importance(values):
    values = [dict(item) for item in values if item is not None]
    total = sum(
        abs(float(item.get("node_times_weight_magnitude", 0.0)))
        for item in values
        if item.get("node_times_weight_magnitude") is not None
    )
    for item in values:
        contribution = item.get("node_times_weight_magnitude")
        item["importance"] = 0.0 if contribution is None or total == 0.0 else abs(float(contribution)) / total
    return values


def _residual_node_detail(meta, seq_idx, node_idx, probability=None, u_idx=None):
    raw_value = _seq_node_value(_SHARED_PREV_NODE_VALUES, seq_idx, node_idx)
    item = {
        "name": meta["name"],
        "node_idx": int(node_idx),
        "local_idx": int(node_idx) - int(meta["start"]),
        "raw_value": raw_value,
    }
    if probability is not None:
        item["normalized_probability"] = float(probability)
    if u_idx is not None and _SHARED_SP is not None:
        prev_to_curr_in = _SHARED_SP.get("prev_to_curr_in")
        if prev_to_curr_in is None:
            return item
        cost_to_u = float(prev_to_curr_in[int(node_idx), int(u_idx)])
        weight_magnitude = _weight_magnitude_from_cost(cost_to_u)
        item["cost_to_u"] = cost_to_u
        item["original_weight_magnitude_to_u"] = weight_magnitude
        if raw_value is not None:
            node_times_weight = float(raw_value * weight_magnitude)
            item["node_times_weight_magnitude"] = node_times_weight
            if probability is not None:
                item["probability_times_node_weight"] = float(float(probability) * node_times_weight)
    return item


def _matching_residual_node_indices(u_idx):
    if _SHARED_RESIDUAL_STARTS is None:
        return np.empty((0,), dtype=np.int64)
    u_idx = int(u_idx)
    widths = _SHARED_RESIDUAL_ENDS - _SHARED_RESIDUAL_STARTS
    return (_SHARED_RESIDUAL_STARTS[u_idx < widths] + u_idx).astype(np.int64, copy=False)


def _matching_residual_nodes(u_idx):
    if not _SHARED_RESIDUAL_NODE_META or _SHARED_RESIDUAL_STARTS is None:
        return []

    node_indices = _matching_residual_node_indices(u_idx)
    if node_indices.size == 0:
        return []
    meta_idx = np.nonzero(int(u_idx) < (_SHARED_RESIDUAL_ENDS - _SHARED_RESIDUAL_STARTS))[0]
    return [
        (_SHARED_RESIDUAL_NODE_META[int(meta_pos)], int(node_idx))
        for meta_pos, node_idx in zip(meta_idx, node_indices)
    ]


def _residual_source_node_values(seq_idx, u_idx, mu=None, prev_active=None):
    if _SHARED_PREV_NODE_VALUES is None or not _SHARED_RESIDUAL_NODE_META:
        return None
    prev_probs = None if mu is None else np.asarray(mu[:-1], dtype=curvature_np_dtype()).reshape(-1)
    prev_active = None if prev_active is None else np.asarray(prev_active, dtype=np.int64).reshape(-1)
    values = []
    for meta, node_idx in _matching_residual_nodes(u_idx):
        probability = 0.0
        if prev_probs is not None and prev_active is not None:
            match = np.nonzero(prev_active == node_idx)[0]
            if match.size > 0:
                probability = float(prev_probs[int(match[0])])
        values.append(_residual_node_detail(meta, seq_idx, node_idx, probability, u_idx))
    return values or None


def _edge_prev_distribution_with_residual(seq_idx, u_idx):
    if _SHARED_PREV_NODE_VALUES is None or not _SHARED_RESIDUAL_NODE_META:
        return None
    base_width = int(_SHARED_RESIDUAL_NODE_META[0]["start"])
    residual_nodes = _matching_residual_node_indices(u_idx)
    if base_width <= 0 and residual_nodes.size == 0:
        return None

    seq_values = _SHARED_PREV_NODE_VALUES[int(seq_idx)]
    row_values = np.empty(base_width + residual_nodes.size, dtype=curvature_np_dtype())
    if base_width > 0:
        row_values[:base_width] = seq_values[:base_width]
    if residual_nodes.size > 0:
        row_values[base_width:] = seq_values[residual_nodes]

    row = _build_node_distribution_row_from_values(
        row_values,
        _SHARED_ALPHA,
    )
    mu, active = _edge_distribution(row, _SHARED_ALPHA)
    active = np.asarray(active, dtype=np.int64)
    if active.size > 0:
        active = active.copy()
        residual_active = active >= base_width
        if np.any(residual_active):
            active[residual_active] = residual_nodes[active[residual_active] - base_width]
    return mu, active


def _filter_residual_prev_distribution(u_idx, mu, prev_active):
    if (
        _SHARED_PREV_NODE_VALUES is None
        or _SHARED_RESIDUAL_STARTS is None
        or len(prev_active) == 0
    ):
        return mu, prev_active

    prev_active = np.asarray(prev_active, dtype=np.int64).reshape(-1)
    prev_probs = np.asarray(mu[:-1], dtype=curvature_np_dtype()).reshape(-1)
    keep = np.ones(prev_active.shape, dtype=bool)
    is_residual = np.zeros(prev_active.shape, dtype=bool)
    matches_source = np.zeros(prev_active.shape, dtype=bool)
    for start, end in zip(_SHARED_RESIDUAL_STARTS, _SHARED_RESIDUAL_ENDS):
        rows = (prev_active >= start) & (prev_active < end)
        is_residual |= rows
        matches_source |= rows & ((prev_active - start) == int(u_idx))
    keep[is_residual] = matches_source[is_residual]

    if keep.all():
        return mu, prev_active

    kept_active = prev_active[keep]
    kept_probs = prev_probs[keep]
    prob_sum = float(kept_probs.sum())
    if prob_sum > 0.0:
        kept_probs = kept_probs * (float(1.0 - _SHARED_ALPHA) / prob_sum)
        kept_mu = np.hstack((kept_probs, np.array([_SHARED_ALPHA], dtype=curvature_np_dtype())))
    else:
        kept_mu = np.array([1.0], dtype=curvature_np_dtype())
        kept_active = np.empty((0,), dtype=np.int64)
    return kept_mu.astype(curvature_np_dtype(), copy=False), kept_active


def _append_residual_prev_nodes(operations, prev_node_tensor, residual_names):
    residual_tensors = [operations[name] for name in residual_names if name in operations]
    if not residual_tensors:
        return prev_node_tensor
    tensors = []
    if prev_node_tensor is not None:
        tensors.append(prev_node_tensor)
    tensors.extend(residual_tensors)
    return torch.cat(tensors, dim=-1) if len(tensors) > 1 else tensors[0]


def _build_residual_node_meta(operations, residual_names, prev_width):
    if not residual_names:
        return None

    metas = []
    offset = int(prev_width or 0)
    for name in residual_names:
        residual = operations.get(name)
        if residual is None:
            continue
        width = int(residual.shape[-1])
        metas.append({"name": name, "start": offset, "end": offset + width})
        offset += width

    return metas or None


def _set_shared_residual_node_meta(meta):
    global _SHARED_RESIDUAL_NODE_META, _SHARED_RESIDUAL_STARTS, _SHARED_RESIDUAL_ENDS

    _SHARED_RESIDUAL_NODE_META = meta
    if not meta:
        _SHARED_RESIDUAL_STARTS = None
        _SHARED_RESIDUAL_ENDS = None
        return

    _SHARED_RESIDUAL_STARTS = np.asarray(
        [int(item["start"]) for item in meta],
        dtype=np.int64,
    )
    _SHARED_RESIDUAL_ENDS = np.asarray(
        [int(item["end"]) for item in meta],
        dtype=np.int64,
    )


def _node_value_is_zero(value):
    return value is not None and np.isfinite(value) and value == 0.0


def _edge_duv_beta(source_value, target_value, weight_magnitude=1.0):
    if source_value is None or target_value is None:
        return 1.0
    if source_value == 0.0 or target_value == 0.0:
        return 0.0
    beta = abs(source_value) * abs(weight_magnitude) / abs(target_value) if target_value != 0.0 else 1.0
    return float(beta) if np.isfinite(beta) and beta > 0.0 else 1.0


def _finite_array_stats(values):
    values = np.asarray(values, dtype=curvature_np_dtype()).reshape(-1)
    finite = np.isfinite(values)
    if not np.any(finite):
        return None

    vals = values[finite].astype(np.float64, copy=False)
    return {
        "count": int(vals.size),
        "sum": float(vals.sum()),
        "sum_sq": float(np.dot(vals, vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def _finite_cost_stats(values):
    if values is None:
        return None
    values = np.asarray(values, dtype=curvature_np_dtype()).reshape(-1)
    finite = np.isfinite(values)
    if not np.any(finite):
        return None

    vals = values[finite].astype(np.float64, copy=False)
    return {
        "count": int(vals.size),
        "sum": float(vals.sum()),
        "sum_sq": float(np.dot(vals, vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "median_sum": float(np.median(vals)),
        "median_count": 1,
    }


def _example_cost_stats(curr_dist_np, sp):
    return {
        "prev": _finite_cost_stats(sp.get("prev_to_curr_in")),
        "cur": _finite_cost_stats(curr_dist_np),
        "next": _finite_cost_stats(sp.get("curr_out_to_next")),
    }


def _score_value(score, idx, seq_idx, other_idx=None):
    if score is None:
        return None
    if np.isscalar(score):
        return float(score)
    score = np.asarray(score, dtype=curvature_np_dtype())
    if score.ndim == 3:
        if other_idx is None:
            return None
        head_dim = int(_SHARED_MODEL_META["head_dim"])
        head_idx = int(other_idx) // head_dim
        if int(idx) >= score.shape[1] or head_idx >= score.shape[2]:
            return None
        return float(score[int(seq_idx), int(idx), head_idx])
    if score.ndim >= 2:
        idx = int(idx)
        if idx >= score.shape[1]:
            return None
        return float(score[int(seq_idx), idx])
    return float(score[int(seq_idx)])


def _get_min_QK_A_cost(v_idx, u_idx, s, prev_active):
    prev_active = _as_index_array(prev_active)

    head_dim = _SHARED_MODEL_META["head_dim"]
    repeat = _SHARED_MODEL_META["repeat"]
    seq_len = _SHARED_SEQ_LEN

    d = v_idx % head_dim
    kv_head = v_idx // head_dim

    # repeated q-heads that share this kv head
    q_start = kv_head * repeat
    q_end = (kv_head + 1) * repeat

    shared_q_heads = np.arange(q_start, q_end)
    # seq @ repeat
    shared_out_idx = shared_q_heads * head_dim + d   # shape [repeat]

    # k cost, _SPK["prev_to_next_all"] = [input, seq * q head]
    k_start = kv_head * seq_len * repeat
    k_end = k_start + repeat * seq_len
    shared_k_heads = np.arange(k_start, k_end)    # [repeat * seq_len]

    # all seq @ repeat
    k_down = _V_COST[q_start:q_end, s, d]

    # Q only overlaps K on out_seq = s
    shared_start = s * repeat
    shared_end = (s + 1) * repeat
    merged_width = seq_len * repeat

    prev_merged = np.empty((0, merged_width), dtype=curvature_np_dtype())

    if len(prev_active) > 0:
        # previous input -> Q-local-out
        q_prev = _Q_to_A["prev_to_next_all"][np.ix_(prev_active, shared_out_idx)]     # [P, repeat]

        # previous input -> K-local-node over all seq/repeated heads
        k_prev_prefix = _SPK["prev_to_next_all"][np.ix_(prev_active, shared_k_heads)]   # [P, seq*repeat]
        # Reorder from [r0_s0, ..., r0_sN, r1_s0, ..., rM_sN] to [s0_r0, ..., sN_rM].
        k_prev_prefix = k_prev_prefix.reshape(len(prev_active), repeat, seq_len).transpose(0, 2, 1)

        # previous input -> K -> V-out, broadcast repeated-head cost across all seq
        k_prev = k_prev_prefix + k_down[None, None, :]                             # [P, seq, repeat]
        prev_merged = k_prev.reshape(len(prev_active), merged_width)

        prev_merged[:, shared_start:shared_end] = np.minimum(
            q_prev,
            prev_merged[:, shared_start:shared_end]
        )


    # current input -> Q-local-out
    q_curr = _Q_to_A["curr_in_to_next_all"][u_idx, shared_out_idx]             # [repeat]

    # current input -> K-local-node over all seq/repeated heads
    k_curr_prefix = _SPK["curr_in_to_next_all"][u_idx, shared_k_heads]         # [seq*repeat]
    # Reorder from [r0_s0, ..., r0_sN, r1_s0, ..., rM_sN] to [s0_r0, ..., sN_rM].
    k_curr_prefix = k_curr_prefix.reshape(repeat, seq_len).transpose(1, 0)     # [seq, repeat]

    # current input -> K -> V-out, broadcast repeated-head cost across all seq
    k_curr = k_curr_prefix + k_down[None, :]                                     # [seq, repeat]

    curr_merged = k_curr.reshape(merged_width)
    curr_merged[shared_start:shared_end] = np.minimum(
        q_curr,
        curr_merged[shared_start:shared_end]
    )

    return {
        "prev_to_out": prev_merged,
        "curr_to_out": curr_merged,
    }


def _edge_cost_matrix_seq_aware(u_idx, v_idx, prev_active, next_active, sp_uv, seq = 0):
    prev_active = _as_index_array(prev_active)
    next_active = _as_index_array(next_active)

    short_name = _SHARED_SHORT_NAME
    if short_name == "q_proj":
        cost = _edge_cost_matrix_base(u_idx, v_idx, prev_active, next_active, sp_uv)
        return cost

    if short_name not in {"v_proj", "o_proj"}:
        cost = _edge_cost_matrix_base(u_idx, v_idx, prev_active, next_active, sp_uv)
        return cost

    prev_count = int(len(prev_active))
    next_count = int(len(next_active))
    cost = np.full((prev_count + 1, next_count + 1), np.inf, dtype=curvature_np_dtype())

    sp = _SHARED_SP
    A = _A
    meta = _SHARED_MODEL_META
    head_dim = meta["head_dim"]
    repeat = meta["repeat"]

    prev_to_curr_out_all = sp.get("prev_to_curr_out_all", {})
    curr_in_to_next_all = sp.get("curr_in_to_next_all", {})

    # For seq-aware ops, both attention weights and attention metadata must exist.
    assert A is not None and meta is not None

    cost[-1, -1] = sp_uv

    # Static prev -> current output-node column
    if prev_count > 0 and prev_to_curr_out_all:
        block = _min_reduce_blocks([
            matrix[prev_active, v_idx]
            for matrix in prev_to_curr_out_all.values()
        ])
        if block is not None:
            cost[:-1, -1] = block

    # Static current input-node row -> next
    if next_count > 0 and curr_in_to_next_all:
        block = _min_reduce_blocks([
            matrix[u_idx, next_active]
            for matrix in curr_in_to_next_all.values()
        ])
        if block is not None:
            cost[-1, :-1] = block


    # Dynamic attention-coupled part
    if short_name == "v_proj" and next_count > 0:
        merge_cost = _get_min_QK_A_cost(v_idx, u_idx, seq, prev_active)

        dynamic_next = _build_vproj_to_att_out_cost(
            a=A,
            seq_len=_SHARED_SEQ_LEN,
            s_in=seq,
            v_idx=v_idx,
            head_dim=head_dim,
            repeat=repeat,
        )

        # if next_active is a subset, select aligned entries first
        if dynamic_next.shape[0] != next_count:
            dynamic_next = dynamic_next[next_active]
        if dynamic_next.shape[0] != next_count:
            raise ValueError(
                f"dynamic_next shape mismatch: got {dynamic_next.shape}, expected ({next_count},)"
            )

        merge_prev = merge_cost["prev_to_out"]
        merge_curr = merge_cost["curr_to_out"]

        if merge_prev.shape[1] != next_count and merge_prev.shape[1] > 0:
            merge_prev = merge_prev[:, next_active]
        if merge_curr.shape[0] != next_count:
            merge_curr = merge_curr[next_active]

        # Base dynamic path uses current v_proj edge then attention-to-output.
        base_prev = cost[:-1, -1][:, None] + dynamic_next[None, :]
        base_curr = cost[-1, -1] + dynamic_next

        if prev_count > 0 and merge_prev.shape[1] == next_count:
            cost[:-1, :-1] = np.minimum(base_prev, merge_prev)
        else:
            cost[:-1, :-1] = base_prev

        # Current input row uses the better of direct v->A->out and merged Q/K->A->out.
        cost[-1, :-1] = np.minimum(base_curr, merge_curr)

    elif short_name == "o_proj" and prev_count > 0:
        dynamic_prev = _build_att_out_to_o_cost(
            a=A,
            s_out=seq,
            out_idx=u_idx,
            head_dim=head_dim,
        )

        # if prev_active is a subset, select aligned entries first
        if dynamic_prev.shape[0] != prev_count:
            dynamic_prev = dynamic_prev[prev_active]
        if dynamic_prev.shape[0] != prev_count:
            raise ValueError(
                f"dynamic_prev shape mismatch: got {dynamic_prev.shape}, expected ({prev_count},)"
            )

        # prev nodes -> next nodes through current endpoint
        cost[:-1, :] = dynamic_prev[:, None] + cost[-1, :][None, :]

    return _apply_residual_source_cost(cost, prev_active, u_idx)


def _edge_seq_distributions(edge_info, seq_info):
    u_idx, v_idx = edge_info

    sp_uv = float(_SHARED_CURR_DIST[u_idx, v_idx])
    next_active_offset = 0

    # For seq-aware nodes, pick the row for this sequence.
    if _SHARED_SHORT_NAME == "v_proj":
        next_row = None if _SHARED_NEXT_OUT is None else _SHARED_NEXT_OUT[v_idx]
    elif _SHARED_SHORT_NAME in {"q_proj", "k_proj"} and _SHARED_NEXT_OUT is not None:
        head_dim = _SHARED_MODEL_META["head_dim"]
        repeat = _SHARED_MODEL_META["repeat"]
        next_width = _SHARED_NEXT_OUT.shape[-1]
        if _SHARED_SHORT_NAME == "q_proj":
            q_head = v_idx // head_dim
            next_row = _SHARED_NEXT_OUT[q_head, seq_info, :]
            next_active_offset = q_head * next_width
        else:
            kv_head = v_idx // head_dim
            q_start = kv_head * repeat
            next_row = _SHARED_NEXT_OUT[kv_head, seq_info, :]
            next_active_offset = q_start * (next_width // repeat)
    else:
        next_row = None if _SHARED_NEXT_OUT is None else _SHARED_NEXT_OUT[seq_info]

    if _SHARED_SHORT_NAME == "o_proj":
        prev_row = None if _SHARED_PREV_IN is None else _SHARED_PREV_IN[u_idx]
    else:
        prev_row = None if _SHARED_PREV_IN is None else _SHARED_PREV_IN[seq_info]

    residual_prev = _edge_prev_distribution_with_residual(seq_info, u_idx)
    if residual_prev is None:
        mu, prev_active = _edge_distribution(prev_row, _SHARED_ALPHA)
        mu, prev_active = _filter_residual_prev_distribution(u_idx, mu, prev_active)
    else:
        mu, prev_active = residual_prev
    nu, next_active = _edge_distribution(next_row, _SHARED_ALPHA)
    if next_active_offset:
        next_active = next_active + next_active_offset

    return u_idx, v_idx, sp_uv, mu, prev_active, nu, next_active


def _parameter_log_detail(edge, seq_idx, sp_uv, mu, prev_active, nu, next_active, cost):
    if not _SHARED_SAVE_PARAMETER_LOGS:
        return {}

    u_idx, v_idx = (int(edge[0]), int(edge[1]))
    metric_score, metric_prev_score, metric_next_score = metric_utils.score_components_for_edge(edge)

    curr_weight_magnitude = _SHARED_SP.get("curr_weight_magnitude") if _SHARED_SP is not None else None
    original_weight_magnitude = None
    if curr_weight_magnitude is not None:
        original_weight_magnitude = float(curr_weight_magnitude[u_idx, v_idx])
    detail = {
        "in_neighbors": [int(idx) for idx in prev_active.tolist()],
        "out_neighbors": [int(idx) for idx in next_active.tolist()],
        "mu": np.asarray(mu, dtype=curvature_np_dtype()).tolist(),
        "nu": np.asarray(nu, dtype=curvature_np_dtype()).tolist(),
    }
    if _SHARED_SHORT_NAME not in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        if _SHARED_PREV_NODE_VALUES is not None and int(seq_idx) < _SHARED_PREV_NODE_VALUES.shape[0]:
            prev_before = _build_node_distribution_row_from_values(
                _SHARED_PREV_NODE_VALUES[int(seq_idx)],
                _SHARED_ALPHA,
                apply_neighbor_reduction=False,
            )
            mu_before, _ = _edge_distribution(prev_before, _SHARED_ALPHA)
            detail["mu_before_reduction"] = np.asarray(mu_before, dtype=curvature_np_dtype()).tolist()
        if _SHARED_NEXT_NODE_VALUES is not None and int(seq_idx) < _SHARED_NEXT_NODE_VALUES.shape[0]:
            next_before = _build_node_distribution_row_from_values(
                _SHARED_NEXT_NODE_VALUES[int(seq_idx)],
                _SHARED_ALPHA,
                apply_neighbor_reduction=False,
            )
            nu_before, _ = _edge_distribution(next_before, _SHARED_ALPHA)
            detail["nu_before_reduction"] = np.asarray(nu_before, dtype=curvature_np_dtype()).tolist()
    if original_weight_magnitude is not None:
        detail["original_weight_magnitude"] = original_weight_magnitude
        detail["weight_magnitude"] = original_weight_magnitude
    residual_source_values = _residual_source_node_values(seq_idx, u_idx, mu, prev_active)
    if residual_source_values is not None:
        detail["residual_source_node_values"] = residual_source_values
    mu_node_values = _mu_node_values(seq_idx, u_idx, prev_active, mu)
    source_node_value = _source_node_value(seq_idx, u_idx)
    importance_values = []
    if mu_node_values is not None:
        importance_values.extend(mu_node_values)
    if residual_source_values is not None:
        logged_nodes = {int(item["node_idx"]) for item in importance_values}
        importance_values.extend(
            item for item in residual_source_values
            if int(item["node_idx"]) not in logged_nodes
        )
    if source_node_value is not None:
        detail["source_node_raw_value"] = source_node_value
        if original_weight_magnitude is not None:
            source_node_weighted_value = float(source_node_value * original_weight_magnitude)
            source_node_probability = float(np.asarray(mu, dtype=curvature_np_dtype())[-1])
            detail["source_node_mu_probability"] = source_node_probability
            detail["source_node_times_weight_magnitude"] = source_node_weighted_value
            detail["source_probability_times_node_weight"] = float(
                source_node_probability * source_node_weighted_value
            )
    if importance_values:
        detail["mu_importance_values"] = _with_importance(importance_values)
        detail["mu_node_weighted_sum_to_u"] = float(sum(
            abs(float(item.get("node_times_weight_magnitude", 0.0)))
            for item in importance_values
            if item.get("node_times_weight_magnitude") is not None
        ))
    out_neighbor_values = _out_neighbor_node_values(seq_idx, next_active, nu, cost)
    if out_neighbor_values is not None:
        detail["out_neighbor_node_values"] = out_neighbor_values
    curr_out_to_next = _SHARED_SP.get("curr_out_to_next") if _SHARED_SP is not None else None
    if curr_out_to_next is not None and len(next_active) > 0:
        v_to_out_neighbors_cost = np.asarray(
            curr_out_to_next[v_idx, np.asarray(next_active, dtype=np.int64)],
            dtype=curvature_np_dtype(),
        )
        inf_mask = np.isinf(v_to_out_neighbors_cost)
        if np.any(inf_mask):
            curr_out_row = np.asarray(curr_out_to_next[v_idx], dtype=curvature_np_dtype())
            detail["v_to_out_neighbors_inf_nodes"] = [
                int(node_idx) for node_idx in np.asarray(next_active, dtype=np.int64)[inf_mask].tolist()
            ]
            detail["v_to_out_neighbors_finite_count"] = int(np.isfinite(v_to_out_neighbors_cost).sum())
            detail["v_to_all_out_finite_count"] = int(np.isfinite(curr_out_row).sum())
            detail["v_to_all_out_count"] = int(curr_out_row.size)
    if metric_score is not None:
        detail["metric_score"] = float(metric_score[int(seq_idx)])
    if metric_prev_score is not None:
        detail["metric_prev_score"] = float(metric_prev_score[int(seq_idx)])
    if metric_next_score is not None:
        detail["metric_next_score"] = float(metric_next_score[int(seq_idx)])

    ot_prev_score = _score_value(_SHARED_OT_PREV_SCORE, v_idx, seq_idx, other_idx=u_idx)
    ot_next_score = _score_value(_SHARED_OT_NEXT_SCORE, u_idx, seq_idx, other_idx=v_idx)
    if ot_prev_score is not None:
        detail["ot_prev_score"] = ot_prev_score
    if ot_next_score is not None:
        detail["ot_next_score"] = ot_next_score
    if ot_prev_score is not None or ot_next_score is not None:
        detail["ot_neighbor_score"] = float(ot_prev_score or 0.0) + float(ot_next_score or 0.0)

    return detail


def _compute_single_edge_seq_global(edge_info, seq_info):
    u_idx, v_idx, sp_uv, mu, prev_active, nu, next_active = _edge_seq_distributions(
        edge_info,
        seq_info,
    )
    source_value = _source_node_value(seq_info, u_idx)
    target_value = _target_node_value(seq_info, v_idx)
    gate_beta = _gate_beta_value(seq_info, v_idx) if _SHARED_SHORT_NAME == "gate_proj" else 1.0
    if gate_beta is None:
        gate_beta = 1.0
    source_node_zero = _node_value_is_zero(source_value)
    target_node_zero = _node_value_is_zero(target_value)
    gate_beta_zero = _node_value_is_zero(gate_beta)
    duv_beta = 1.0
    if _SHARED_SAVE_PARAMETER_LOGS:
        curr_weight_magnitude = _SHARED_SP.get("curr_weight_magnitude") if _SHARED_SP is not None else None
        weight_magnitude = 1.0 if curr_weight_magnitude is None else float(curr_weight_magnitude[u_idx, v_idx])
        duv_beta = _edge_duv_beta(source_value, target_value, weight_magnitude)

    if source_node_zero or target_node_zero or gate_beta_zero:
        curv = 1.0 if len(prev_active) == 0 or len(next_active) == 0 else 2.0
        result = {
            "seq_idx": seq_info,
            "v_idx": v_idx,
            "u_idx": u_idx,
            "curv": curvature_np_dtype()(curv),
            "mu_len": int(len(mu)),
            "nu_len": int(len(nu)),
            "cost_has_inf": False,
            "cost_inf_count": 0,
            "source_node_zero": bool(source_node_zero),
            "target_node_zero": bool(target_node_zero),
            "gate_beta_zero": bool(gate_beta_zero),
        }
        if _SHARED_SHORT_NAME == "gate_proj":
            result["gate_beta"] = float(gate_beta)
        if _SHARED_SAVE_PARAMETER_LOGS:
            result["w_dist"] = 0.0
            result["sp_uv"] = sp_uv
            result["duv_beta"] = duv_beta
        return result

    cost = _edge_cost_matrix_seq_aware(
        u_idx=u_idx,
        v_idx=v_idx,
        prev_active=prev_active,
        next_active=next_active,
        sp_uv=sp_uv,
        seq=seq_info,
    )
    cost_inf_count = int(np.isinf(cost).sum())

    emd_error = None
    try:
        w_dist, transport_log = ot.emd2(mu, nu, cost, log=True)
        emd_error = transport_log.get("warning")
        w_dist = float("inf") if emd_error else float(w_dist)
    except Exception as exc:
        emd_error = str(exc)
        print(f"OT emd2 failed for seq={seq_info}, u={u_idx}, v={v_idx}: {emd_error}")
        w_dist = float("inf")

    d = np.divide(sp_uv, gate_beta)
    curv = float("inf") if (not np.isfinite(w_dist) or d == 0.0) else 1.0 - (w_dist / d)
    if np.isfinite(curv):
        curv = curv / (1-_SHARED_ALPHA)
    result = {
        "seq_idx": seq_info,
        "v_idx": v_idx,
        "u_idx": u_idx,
        "curv": curvature_np_dtype()(curv),
        "mu_len": int(len(mu)),
        "nu_len": int(len(nu)),
        "cost_has_inf": bool(cost_inf_count > 0),
        "cost_inf_count": cost_inf_count,
    }
    if _SHARED_SHORT_NAME == "gate_proj":
        result["gate_beta"] = float(gate_beta)
        result["gate_beta_zero"] = False
    if _SHARED_SAVE_PARAMETER_LOGS:
        result["w_dist"] = w_dist
        result["sp_uv"] = sp_uv
        result["emd_error"] = emd_error
        result["duv_beta"] = duv_beta
        result.update(_parameter_log_detail(edge_info, seq_info, sp_uv, mu, prev_active, nu, next_active, cost))
    return result


def _init_worker(
    curr_dist_meta,
    prev_meta,
    next_meta,
    source_node_meta=None,
    target_node_meta=None,
    gate_beta_meta=None,
    prev_node_meta=None,
    next_node_meta=None,
    prev_seq_metas=None,
    next_seq_metas=None,
    prev_score_meta=None,
    next_score_meta=None,
    prev_score_value=None,
    next_score_value=None,
    ot_prev_score_value=None,
    ot_next_score_value=None,
):
    global _SHARED_CURR_DIST, _SHARED_PREV_IN, _SHARED_NEXT_OUT
    global _SHARED_SOURCE_NODE_VALUES, _SHARED_TARGET_NODE_VALUES, _SHARED_GATE_BETA_VALUES
    global _SHARED_PREV_NODE_VALUES, _SHARED_NEXT_NODE_VALUES
    global _SHARED_OT_PREV_SCORE, _SHARED_OT_NEXT_SCORE
    global _WORKER_CURR_DIST_SHM, _WORKER_PREV_IN_SHM, _WORKER_NEXT_OUT_SHM
    global _WORKER_SOURCE_NODE_SHM, _WORKER_TARGET_NODE_SHM, _WORKER_GATE_BETA_SHM
    global _WORKER_PREV_NODE_SHM, _WORKER_NEXT_NODE_SHM
    global _WORKER_SCORE_SHMS
    global _WORKER_PREV_SEQ_METAS, _WORKER_NEXT_SEQ_METAS
    global _WORKER_PREV_SEQ_SHMS, _WORKER_NEXT_SEQ_SHMS
    global _WORKER_PREV_SEQ_CACHE, _WORKER_NEXT_SEQ_CACHE

    _WORKER_CURR_DIST_SHM, _SHARED_CURR_DIST = _from_shared_numpy(curr_dist_meta)

    _WORKER_PREV_IN_SHM = None
    _SHARED_PREV_IN = None
    if prev_meta is not None:
        _WORKER_PREV_IN_SHM, _SHARED_PREV_IN = _from_shared_numpy(prev_meta)

    _WORKER_NEXT_OUT_SHM = None
    _SHARED_NEXT_OUT = None
    if next_meta is not None:
        _WORKER_NEXT_OUT_SHM, _SHARED_NEXT_OUT = _from_shared_numpy(next_meta)

    _WORKER_SOURCE_NODE_SHM = None
    _SHARED_SOURCE_NODE_VALUES = None
    if source_node_meta is not None:
        _WORKER_SOURCE_NODE_SHM, _SHARED_SOURCE_NODE_VALUES = _from_shared_numpy(source_node_meta)

    _WORKER_TARGET_NODE_SHM = None
    _SHARED_TARGET_NODE_VALUES = None
    if target_node_meta is not None:
        _WORKER_TARGET_NODE_SHM, _SHARED_TARGET_NODE_VALUES = _from_shared_numpy(target_node_meta)

    _WORKER_GATE_BETA_SHM = None
    _SHARED_GATE_BETA_VALUES = None
    if gate_beta_meta is not None:
        _WORKER_GATE_BETA_SHM, _SHARED_GATE_BETA_VALUES = _from_shared_numpy(gate_beta_meta)

    _WORKER_PREV_NODE_SHM = None
    _SHARED_PREV_NODE_VALUES = None
    if prev_node_meta is not None:
        _WORKER_PREV_NODE_SHM, _SHARED_PREV_NODE_VALUES = _from_shared_numpy(prev_node_meta)

    _WORKER_NEXT_NODE_SHM = None
    _SHARED_NEXT_NODE_VALUES = None
    if next_node_meta is not None:
        _WORKER_NEXT_NODE_SHM, _SHARED_NEXT_NODE_VALUES = _from_shared_numpy(next_node_meta)

    _WORKER_SCORE_SHMS = []
    prev_score = prev_score_value
    if prev_score_meta is not None:
        shm, prev_score = _from_shared_numpy(prev_score_meta)
        _WORKER_SCORE_SHMS.append(shm)

    next_score = next_score_value
    if next_score_meta is not None:
        shm, next_score = _from_shared_numpy(next_score_meta)
        _WORKER_SCORE_SHMS.append(shm)
    metric_utils.set_shared_metric_state(
        prev_score,
        next_score,
        seq_len=_SHARED_SEQ_LEN,
        top_k=_SHARED_TOP_K,
        seq_select=_SHARED_SEQ_SELECT,
        out_count=_SHARED_CURR_DIST.shape[1] if _SHARED_CURR_DIST is not None else None,
    )
    if _SHARED_SAVE_PARAMETER_LOGS:
        _SHARED_OT_PREV_SCORE = ot_prev_score_value if ot_prev_score_value is not None else prev_score
        _SHARED_OT_NEXT_SCORE = ot_next_score_value if ot_next_score_value is not None else next_score
    else:
        _SHARED_OT_PREV_SCORE = None
        _SHARED_OT_NEXT_SCORE = None

    _WORKER_PREV_SEQ_METAS = prev_seq_metas
    _WORKER_NEXT_SEQ_METAS = next_seq_metas
    _WORKER_PREV_SEQ_SHMS = {}
    _WORKER_NEXT_SEQ_SHMS = {}
    _WORKER_PREV_SEQ_CACHE = {}
    _WORKER_NEXT_SEQ_CACHE = {}


def _compute_edge_with_seq(task):
    seq_idx, edge = task

    global _SHARED_PREV_IN, _SHARED_NEXT_OUT

    prev_in_distribution = _SHARED_PREV_IN
    next_out_distribution = _SHARED_NEXT_OUT

    if _SHARED_SHORT_NAME == "o_proj":
        prev_in_distribution = _load_worker_seq_distribution(
            seq_idx,
            _WORKER_PREV_SEQ_METAS,
            _WORKER_PREV_SEQ_SHMS,
            _WORKER_PREV_SEQ_CACHE,
        )
    elif _SHARED_SHORT_NAME == "v_proj":
        next_out_distribution = _load_worker_seq_distribution(
            seq_idx,
            _WORKER_NEXT_SEQ_METAS,
            _WORKER_NEXT_SEQ_SHMS,
            _WORKER_NEXT_SEQ_CACHE,
        )

    old_prev = _SHARED_PREV_IN
    old_next = _SHARED_NEXT_OUT
    _SHARED_PREV_IN = prev_in_distribution
    _SHARED_NEXT_OUT = next_out_distribution

    try:
        return _compute_single_edge_seq_global(edge, seq_idx)
    finally:
        _SHARED_PREV_IN = old_prev
        _SHARED_NEXT_OUT = old_next


def _compute_edge_with_top_seq(edge):
    results = []
    for seq_idx in metric_utils.top_seq_for_edge(edge):
        edge_res = _compute_edge_with_seq((seq_idx, edge))
        if edge_res:
            results.append(edge_res)
    if _SHARED_TOP_K == -1 and _SHARED_LPF_WINDOW > 1 and results:
        valid_results = [
            edge_res for edge_res in results
            if not edge_res.get("source_node_zero")
            and not edge_res.get("target_node_zero")
            and not edge_res.get("gate_beta_zero")
        ]
        source_node_zero_count = sum(1 for edge_res in results if edge_res.get("source_node_zero"))
        target_node_zero_count = sum(1 for edge_res in results if edge_res.get("target_node_zero"))
        gate_beta_zero_count = sum(1 for edge_res in results if edge_res.get("gate_beta_zero"))
        gate_beta_values = [
            float(edge_res["gate_beta"])
            for edge_res in results
            if edge_res.get("gate_beta") is not None and np.isfinite(edge_res.get("gate_beta"))
        ]
        if not valid_results:
            return results
        curvs = np.asarray([float(edge_res["curv"]) for edge_res in valid_results], dtype=curvature_np_dtype())
        raw_idx = int(np.argmin(curvs))
        smoothed_curvs = sliding_median_low_pass(curvs, _SHARED_LPF_WINDOW)
        if _SHARED_SAVE_PARAMETER_LOGS:
            detailed_results = []
            valid_idx = 0
            for edge_res in results:
                detailed_res = dict(edge_res)
                if (
                    not edge_res.get("source_node_zero")
                    and not edge_res.get("target_node_zero")
                    and not edge_res.get("gate_beta_zero")
                ):
                    detailed_res["curv"] = curvature_np_dtype()(curvs[valid_idx])
                    detailed_res["lpf_curv"] = curvature_np_dtype()(smoothed_curvs[valid_idx])
                    valid_idx += 1
                detailed_results.append(detailed_res)
            return detailed_results

        lpf_idx = int(np.argmin(smoothed_curvs))
        best_res = dict(valid_results[raw_idx])
        best_res["curv"] = curvature_np_dtype()(curvs[raw_idx])
        best_res["lpf_curv"] = curvature_np_dtype()(smoothed_curvs[lpf_idx])
        best_res["source_node_zero_count"] = source_node_zero_count
        best_res["target_node_zero_count"] = target_node_zero_count
        best_res["gate_beta_count"] = len(results)
        best_res["gate_beta_zero_count"] = gate_beta_zero_count
        if gate_beta_values:
            best_res["gate_beta_sum"] = float(np.sum(gate_beta_values, dtype=curvature_np_dtype()))
            best_res["gate_beta_sum_sq"] = float(np.sum(np.square(gate_beta_values), dtype=curvature_np_dtype()))
            best_res["gate_beta_min"] = float(np.min(gate_beta_values))
            best_res["gate_beta_max"] = float(np.max(gate_beta_values))
        return [best_res]
    return results


def _edge_batches(edges, batch_size):
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(edges), batch_size):
        yield edges[start:start + batch_size]


def _edge_batch_size(effective_top_k):
    env_value = os.environ.get("CURV_EDGE_BATCH_SIZE")
    if env_value:
        return max(int(env_value), 1)
    if effective_top_k <= 1:
        return 512
    if effective_top_k <= 10:
        return 64
    return 32


def _compute_edge_batch_with_top_seq(edge_batch):
    return [_compute_edge_with_top_seq(edge) for edge in edge_batch]


def _build_seq_node_values(node_tensor, seq_len):
    if node_tensor is None:
        return None

    if torch.is_tensor(node_tensor):
        tensor = node_tensor.detach().cpu()
        if torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=curvature_torch_dtype())
    else:
        tensor = torch.as_tensor(node_tensor, dtype=curvature_torch_dtype())
    if tensor.numel() == 0:
        return None

    if tensor.dim() == 3:
        if tensor.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1 for raw node values, got shape {tuple(tensor.shape)}"
            )
        tensor = tensor.squeeze(0)
    elif tensor.dim() == 1:
        tensor = tensor.reshape(1, -1)
    elif tensor.dim() != 2:
        return None

    tensor = tensor[:int(seq_len)].contiguous()
    values = tensor.detach().cpu().numpy().astype(curvature_np_dtype(), copy=False)
    return np.abs(values)


def _build_next_node_values(operations, graph_data, short_name, seq_len):
    node_tensor = graph_data.get("next_out")
    if node_tensor is None:
        return None

    if short_name == "q_proj":
        tensor = node_tensor.detach().cpu() if torch.is_tensor(node_tensor) else torch.as_tensor(node_tensor)
        if torch.is_floating_point(tensor):
            tensor = tensor.to(dtype=curvature_torch_dtype())
        if tensor.dim() != 4 or tensor.shape[0] != 1:
            return None
        tensor = tensor.squeeze(0)  # [q_heads, seq_q, seq_k]
        tensor = tensor[:, :int(seq_len), :].transpose(0, 1).contiguous()
        values = tensor.reshape(tensor.shape[0], -1).numpy().astype(curvature_np_dtype(), copy=False)
        return np.abs(values)

    return _build_seq_node_values(node_tensor, seq_len)


def _source_node_tensor_for_op(operations, graph_data, short_name):
    if short_name in {"q_proj", "k_proj", "v_proj"} and "layer_input" in operations:
        return operations["layer_input"]
    if short_name == "o_proj" and "Att_out" in operations:
        return operations["Att_out"]
    if short_name in {"gate_proj", "up_proj"} and "o_proj" in operations:
        return operations["o_proj"]
    if short_name == "down_proj" and "gate_up_out" in operations:
        return operations["gate_up_out"]
    return graph_data["prev_in"]


def _source_node_name_for_op(operations, graph_data, short_name):
    if short_name in {"q_proj", "k_proj", "v_proj"} and "layer_input" in operations:
        return "layer_input"
    if short_name == "o_proj" and "Att_out" in operations:
        return "Att_out"
    if short_name in {"gate_proj", "up_proj"} and "o_proj" in operations:
        return "o_proj"
    if short_name == "down_proj" and "gate_up_out" in operations:
        return "gate_up_out"
    return graph_data["prev_in_name"]


def _reset_shared_state():
    global _SHARED_CURR_DIST, _SHARED_PREV_IN, _SHARED_NEXT_OUT
    global _SHARED_SP, _SHARED_ALPHA, _A, _Q_to_A, _V_COST, _SPK
    global _SHARED_SOURCE_NODE_VALUES, _SHARED_TARGET_NODE_VALUES, _SHARED_GATE_BETA_VALUES
    global _SHARED_PREV_NODE_VALUES, _SHARED_NEXT_NODE_VALUES
    global _SHARED_RESIDUAL_NODE_META, _SHARED_RESIDUAL_STARTS, _SHARED_RESIDUAL_ENDS
    global _SHARED_OT_PREV_SCORE, _SHARED_OT_NEXT_SCORE
    global _SHARED_SHORT_NAME, _SHARED_SAMPLE_IDX, _SHARED_MODEL_META, _SHARED_SEQ_LEN
    global _SHARED_TOP_K, _SHARED_SEQ_SELECT, _SHARED_LPF_WINDOW, _SHARED_SAVE_PARAMETER_LOGS

    _SHARED_CURR_DIST = None
    _SHARED_PREV_IN = None
    _SHARED_NEXT_OUT = None
    _SHARED_SP = None
    _SHARED_ALPHA = 0.0
    _A = None
    _Q_to_A = None
    _V_COST = None
    _SPK = None
    _SHARED_SOURCE_NODE_VALUES = None
    _SHARED_TARGET_NODE_VALUES = None
    _SHARED_GATE_BETA_VALUES = None
    _SHARED_PREV_NODE_VALUES = None
    _SHARED_NEXT_NODE_VALUES = None
    _SHARED_RESIDUAL_NODE_META = None
    _SHARED_RESIDUAL_STARTS = None
    _SHARED_RESIDUAL_ENDS = None
    _SHARED_OT_PREV_SCORE = None
    _SHARED_OT_NEXT_SCORE = None
    _SHARED_SHORT_NAME = None
    _SHARED_SAMPLE_IDX = None
    _SHARED_MODEL_META = None
    _SHARED_SEQ_LEN = 1
    _SHARED_TOP_K = 5
    _SHARED_SEQ_SELECT = "top"
    _SHARED_LPF_WINDOW = 0
    _SHARED_SAVE_PARAMETER_LOGS = False
    metric_utils.reset_shared_metric_state()


def _get_vproj_aux_shortest_paths(operations, layer_cache, sp_cache, device):
    sp_q, _ = build_shortest_path_cache(
        operations=operations,
        layer_cache=layer_cache,
        short_name="q_proj",
        sp_cache=sp_cache,
        device=device,
        model_meta=_SHARED_MODEL_META,
    )
    sp_k, _ = build_shortest_path_cache(
        operations=operations,
        layer_cache=layer_cache,
        short_name="k_proj",
        sp_cache=sp_cache,
        device=device,
        model_meta=_SHARED_MODEL_META,
    )
    return sp_q, sp_k


def compute_op_curvature(
    operations,
    short_name,
    layer_id,
    sample_idx,
    layer_cache,
    sp_cache=None,
    alpha=0.0,
    device="cpu",
    seq_len = 1,
    num_q_heads=0, num_kv_heads=0, head_dim=0, repeat=0,
    sample_edge_num = -1,
    sample_edge_ratio = 1.,
    dataset_name="unknown_dataset",
    l2_norm=False,
    l2_norm_mode="per_example",
    l2_norm_stats=None,
    shared_top_k=10,
    shared_seq_select="top",
    curvature_lpf_window=0,
    analysis_dir=None,
    parameter_log_root=None,
):
    global _SHARED_CURR_DIST, _SHARED_PREV_IN, _SHARED_NEXT_OUT
    global _SHARED_SP, _SHARED_ALPHA, _A, _Q_to_A, _V_COST, _SPK
    global _SHARED_SOURCE_NODE_VALUES, _SHARED_TARGET_NODE_VALUES, _SHARED_GATE_BETA_VALUES
    global _SHARED_PREV_NODE_VALUES, _SHARED_NEXT_NODE_VALUES
    global _SHARED_OT_PREV_SCORE, _SHARED_OT_NEXT_SCORE
    global _SHARED_RESIDUAL_NODE_META, _SHARED_RESIDUAL_STARTS, _SHARED_RESIDUAL_ENDS
    global _SHARED_SHORT_NAME, _SHARED_SAMPLE_IDX, _SHARED_MODEL_META, _SHARED_SEQ_LEN
    global _SHARED_TOP_K, _SHARED_SEQ_SELECT, _SHARED_LPF_WINDOW, _SHARED_SAVE_PARAMETER_LOGS

    if operations is None or layer_cache is None:
        return None

    def l2_reference(name):
        if name is None or not l2_norm or l2_norm_mode != "all_examples" or l2_norm_stats is None:
            return None
        stat_name = name
        if name == "A" and short_name in {"q_proj", "k_proj"}:
            stat_name = f"{short_name}:A"
        if stat_name not in l2_norm_stats:
            raise ValueError(f"Missing all-example L2 stats for {stat_name}")
        return l2_norm_stats[stat_name]

    _reset_shared_state()
    _SHARED_MODEL_META = {
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "repeat": repeat,
    }

    sp, graph_data = build_shortest_path_cache(
        operations=operations,
        layer_cache=layer_cache,
        short_name=short_name,
        sp_cache=sp_cache,
        device=device,
        model_meta=_SHARED_MODEL_META,
        include_qk_next=short_name in {"q_proj", "k_proj"},
    )
    if sp is None:
        return None

    if short_name == "v_proj":
        sp_q, sp_k = _get_vproj_aux_shortest_paths(
            operations=operations,
            layer_cache=layer_cache,
            sp_cache=sp_cache,
            device=device,
        )
        if sp_q is None or sp_k is None:
            raise ValueError("q_proj and k_proj shortest-path caches are required for v_proj")

        _SPK = {}

        if sp_k["prev_to_next_all"]:
            _SPK["prev_to_next_all"] = next(iter(sp_k["prev_to_next_all"].values()))
        _SPK["curr_in_to_next_all"] = sp_k["curr_in_to_next_all"]["q_proj"]  # [input, seq * q_head]

    curr_dist = sp["curr_dist"]
    in_dim, out_dim = curr_dist.shape

    prev_in_distribution = None
    metric_prev_in_distribution = None
    next_out_distribution = None

    # if is o_proj, prev is A
    base_prev_in_tensor = graph_data["prev_in"]
    residual_names = graph_data.get("residual_names", [])
    prev_node_tensor = base_prev_in_tensor
    if residual_names and short_name not in ["o_proj"]:
        prev_node_tensor = _append_residual_prev_nodes(
            operations,
            base_prev_in_tensor,
            residual_names,
        )
    _SHARED_PREV_NODE_VALUES = (
        _build_seq_node_values(prev_node_tensor, seq_len)
        if (parameter_log_root or residual_names)
        else None
    )

    if (short_name not in ["o_proj"]) and base_prev_in_tensor is not None:
        prev_in_distribution = _build_node_distribution(
            base_prev_in_tensor,
            graph_data["prev_in_name"],
            alpha,
            l2_norm=l2_norm,
            l2_norm_mode=l2_norm_mode,
            l2_reference=l2_reference(graph_data["prev_in_name"]),
        )
        if residual_names:
            metric_prev_in_distribution = prev_in_distribution

    # if is q, k, v, the next is A or attention out
    if (short_name in {"q_proj", "k_proj"} or short_name not in ["v_proj"]) and graph_data["next_out"] is not None:
        if short_name in {"q_proj", "k_proj"}:
            next_out_distribution = _build_qk_out_node_distribution(
                short_name,
                graph_data["next_out"],
                alpha,
                l2_norm=l2_norm,
                l2_norm_mode=l2_norm_mode,
                l2_reference=l2_reference(graph_data["next_out_name"]),
                repeat=repeat,
            )
        else:
            next_out_distribution = _build_node_distribution(
                graph_data["next_out"],
                graph_data["next_out_name"],
                alpha,
                l2_norm=l2_norm,
                l2_norm_mode=l2_norm_mode,
                l2_reference=l2_reference(graph_data["next_out_name"]),
            ) # [batch, seq, hidden size]


    if prev_in_distribution is None and next_out_distribution is None:
        magnitude = torch.as_tensor(
            sp["curr_weight_magnitude"].T,
            dtype=curvature_torch_dtype(),
        ).contiguous()
        print(f"op = {short_name}, sample = {sample_idx}: using weight magnitude fallback")
        return {
            "curvature": magnitude,
            "magnitude_fallback": True,
        }

    _SHARED_CURR_DIST = curr_dist
    _SHARED_SP = sp
    _SHARED_ALPHA = alpha
    _SHARED_SHORT_NAME = short_name
    _SHARED_SAMPLE_IDX = sample_idx
    _SHARED_SEQ_LEN = seq_len

    _SHARED_PREV_IN = None if residual_names else prev_in_distribution
    _SHARED_NEXT_OUT = next_out_distribution
    source_node_name = _source_node_name_for_op(operations, graph_data, short_name)
    _SHARED_SOURCE_NODE_VALUES = _build_distribution_node_values(
        _source_node_tensor_for_op(operations, graph_data, short_name),
        seq_len,
        l2_norm=l2_norm,
        l2_norm_mode=l2_norm_mode,
        l2_reference=l2_reference(source_node_name),
    )
    _SHARED_TARGET_NODE_VALUES = _build_distribution_node_values(
        operations.get(short_name),
        seq_len,
        l2_norm=l2_norm,
        l2_norm_mode=l2_norm_mode,
        l2_reference=l2_reference(short_name),
    )
    _SHARED_GATE_BETA_VALUES = (
        np.ones_like(_SHARED_TARGET_NODE_VALUES, dtype=curvature_np_dtype())
        if short_name == "gate_proj"
        else None
    )
    _SHARED_NEXT_NODE_VALUES = (
        _build_next_node_values(operations, graph_data, short_name, seq_len)
        if parameter_log_root
        else None
    )
    prev_width = 0 if graph_data["prev_in"] is None else int(graph_data["prev_in"].shape[-1])
    residual_start = prev_width if residual_names else None
    _set_shared_residual_node_meta(
        _build_residual_node_meta(
            operations,
            residual_names,
            prev_width,
        )
    )
    del prev_node_tensor

    curvature = torch.full((out_dim, in_dim), float("inf"), dtype=curvature_torch_dtype())
    lpf_curvature = None

    curr_dist_np = np.asarray(curr_dist, dtype=curvature_np_dtype())
    curr_dist_finite_edges = int(np.isfinite(curr_dist_np).sum())
    curr_dist_infinite_edges = int(np.isinf(curr_dist_np).sum())
    cost_stats = _example_cost_stats(curr_dist_np, sp)
    curr_weight_magnitude_np = np.asarray(sp["curr_weight_magnitude"], dtype=curvature_np_dtype())
    nonzero_weight_mask = curr_weight_magnitude_np != 0
    zero_weight_parameter_count = int((~nonzero_weight_mask).sum())
    # Sample flat indices first, avoiding two coordinates per eligible parameter.
    finite_edges = np.flatnonzero(np.isfinite(curr_dist_np) & (curr_dist_np > 0) & nonzero_weight_mask)
    del curr_weight_magnitude_np, nonzero_weight_mask

    analysis_path = analysis_utils.start_curvature_analysis(
        layer_id=layer_id,
        short_name=short_name,
        sample_idx=sample_idx,
        curvature_shape=curvature.shape,
        seq_len=seq_len,
        dataset_name=dataset_name,
        analysis_dir=analysis_dir,
    )
    precomputed_prev_dists = None
    precomputed_next_dists = None


    # build neighbor distribution for v_proj or o_proj with A
    if short_name in {"v_proj", "o_proj"}:
        _A = operations.get("A", None)

        if short_name == "v_proj":
            value_map = _build_vproj_to_att_out_value_map(
                graph_data["next_out"], out_dim, seq_len, head_dim, repeat
            )

            # # node distribution for all seq w/ mask
            # precomputed_next_dists = _precompute_vproj_next_distributions(
            #     value_map, seq_len, repeat, graph_data["next_out_name"], alpha
            # )

            # node distribution for all seq w/o mask
            precomputed_next_dists = _precompute_vproj_next_distributions(
                value_map,
                graph_data["next_out_name"],
                alpha,
                l2_norm=l2_norm,
                l2_norm_mode=l2_norm_mode,
                l2_reference=l2_reference(graph_data["next_out_name"]),
            )

            # x -> Q -> A -> out
            v_cost = operations.get("v_proj", None)
            if v_cost is not None:
                # [q_heads, seq, head_dim]
                _V_COST = _build_v_to_att_out_template(v_cost, _SHARED_MODEL_META)

                # cost matrix from inout to attention out via q_proj
                _Q_to_A = _build_x_to_out_cost(
                    _V_COST,
                    sp_q,
                    _SHARED_MODEL_META,
                    device,
                ) # [input, head_dim * q_head]

                _V_COST = _V_COST.cpu().contiguous().numpy()

        elif short_name == "o_proj":
            value_map = _build_oproj_to_att_in_value_map(
                graph_data["prev_in"], in_dim, seq_len, head_dim, repeat
            )
            precomputed_prev_dists = _precompute_oproj_prev_distributions(
                value_map,
                seq_len,
                graph_data["prev_in_name"],
                alpha,
                l2_norm=l2_norm,
                l2_norm_mode=l2_norm_mode,
                l2_reference=l2_reference(graph_data["prev_in_name"]),
            )

    ctx = get_context("fork")

    base_owned_shms = []
    seq_owned_shms = []
    try:
        curr_shm, curr_meta = _to_shared_numpy(curr_dist_np)
        base_owned_shms.append(curr_shm)

        base_prev_meta = None
        base_next_meta = None
        source_node_meta = None
        target_node_meta = None
        gate_beta_meta = None
        prev_node_meta = None
        next_node_meta = None

        if _SHARED_PREV_IN is not None:
            shm, base_prev_meta = _to_shared_numpy(np.asarray(_SHARED_PREV_IN, dtype=curvature_np_dtype()))
            base_owned_shms.append(shm)

        if _SHARED_NEXT_OUT is not None:
            shm, base_next_meta = _to_shared_numpy(np.asarray(_SHARED_NEXT_OUT, dtype=curvature_np_dtype()))
            base_owned_shms.append(shm)

        if _SHARED_SOURCE_NODE_VALUES is not None:
            shm, source_node_meta = _to_shared_numpy(
                np.asarray(_SHARED_SOURCE_NODE_VALUES, dtype=curvature_np_dtype())
            )
            base_owned_shms.append(shm)

        if _SHARED_TARGET_NODE_VALUES is not None:
            shm, target_node_meta = _to_shared_numpy(
                np.asarray(_SHARED_TARGET_NODE_VALUES, dtype=curvature_np_dtype())
            )
            base_owned_shms.append(shm)

        if _SHARED_GATE_BETA_VALUES is not None:
            shm, gate_beta_meta = _to_shared_numpy(
                np.asarray(_SHARED_GATE_BETA_VALUES, dtype=curvature_np_dtype())
            )
            base_owned_shms.append(shm)

        if _SHARED_PREV_NODE_VALUES is not None:
            shm, prev_node_meta = _to_shared_numpy(
                np.asarray(_SHARED_PREV_NODE_VALUES, dtype=curvature_np_dtype())
            )
            base_owned_shms.append(shm)

        if _SHARED_NEXT_NODE_VALUES is not None:
            shm, next_node_meta = _to_shared_numpy(
                np.asarray(_SHARED_NEXT_NODE_VALUES, dtype=curvature_np_dtype())
            )
            base_owned_shms.append(shm)

        seq_prev_metas, prev_seq_shms = _to_shared_seq_metas(precomputed_prev_dists)
        seq_owned_shms.extend(prev_seq_shms)
        seq_next_metas, next_seq_shms = _to_shared_seq_metas(precomputed_next_dists)
        seq_owned_shms.extend(next_seq_shms)

        # Select by ratio first, then optionally cap with sample_edge_num.
        if len(finite_edges) > 0:
            seed = 13
            rng = np.random.default_rng(seed)

            if sample_edge_ratio < 1.0:
                ratio_edge_num = max(1, int(len(finite_edges) * sample_edge_ratio))
                ratio_edge_num = min(ratio_edge_num, len(finite_edges))
                selected_idx = rng.choice(len(finite_edges), size=ratio_edge_num, replace=False)
                finite_edges = finite_edges[np.sort(selected_idx)]

            if sample_edge_num > 0 and sample_edge_num < len(finite_edges):
                selected_idx = rng.choice(len(finite_edges), size=sample_edge_num, replace=False)
                finite_edges = finite_edges[np.sort(selected_idx)]
        finite_edges = np.column_stack(np.unravel_index(finite_edges, curr_dist_np.shape))

        print(
            f'op = {short_name}, sample = {sample_idx}, seq = {seq_len}, '
            f'total edges = {len(finite_edges)} per seq, cur dist shape = {curr_dist.shape}'
        )

        mu_len_total = 0.0
        nu_len_total = 0.0
        mu_nu_count = 0
        cost_has_inf = False
        cost_inf_count = 0
        q_parameter_cost_stats = _finite_array_stats(curr_dist_np) if short_name == "q_proj" else None
        total_edge_seq_tasks = 0
        gate_beta_count = 0
        gate_beta_zero_count = 0
        gate_beta_sum = 0.0
        gate_beta_sum_sq = 0.0
        gate_beta_min = float("inf")
        gate_beta_max = float("-inf")
        source_node_zero_count = 0
        target_node_zero_count = 0
        t1 = time.time()

        if l2_norm:
            seq_len = 1

        _SHARED_SEQ_LEN = seq_len
        _SHARED_TOP_K = int(shared_top_k)
        _SHARED_SEQ_SELECT = shared_seq_select
        _SHARED_LPF_WINDOW = int(curvature_lpf_window)
        save_parameter_logs = bool(parameter_log_root)
        _SHARED_SAVE_PARAMETER_LOGS = save_parameter_logs
        if _SHARED_TOP_K == -1 and _SHARED_LPF_WINDOW > 1:
            lpf_curvature = torch.full((out_dim, in_dim), float("inf"), dtype=curvature_torch_dtype())

        ot_prev_score = None
        ot_next_score = None
        if not l2_norm and seq_len > 1:
            ot_prev_score, ot_next_score = metric_utils.build_ot_neighbor_score_matrices(
                seq_len=seq_len,
                sp=sp,
                alpha=alpha,
                prev_in_distribution=(
                    metric_prev_in_distribution
                    if metric_prev_in_distribution is not None
                    else prev_in_distribution
                ),
                next_out_distribution=next_out_distribution,
                short_name=short_name,
                residual_start=residual_start,
            )
        if residual_names:
            prev_in_distribution = None
            metric_prev_in_distribution = None
        prev_score = ot_prev_score
        next_score = ot_next_score

        prev_score_meta = None
        next_score_meta = None
        if prev_score is not None and not np.isscalar(prev_score):
            shm, prev_score_meta = _to_shared_numpy(np.asarray(prev_score, dtype=curvature_np_dtype()))
            base_owned_shms.append(shm)
        if next_score is not None and not np.isscalar(next_score):
            shm, next_score_meta = _to_shared_numpy(np.asarray(next_score, dtype=curvature_np_dtype()))
            base_owned_shms.append(shm)
        metric_utils.set_shared_metric_state(
            prev_score,
            next_score,
            seq_len=_SHARED_SEQ_LEN,
            top_k=_SHARED_TOP_K,
            seq_select=_SHARED_SEQ_SELECT,
            out_count=out_dim,
        )
        if save_parameter_logs:
            _SHARED_OT_PREV_SCORE = ot_prev_score
            _SHARED_OT_NEXT_SCORE = ot_next_score
        else:
            _SHARED_OT_PREV_SCORE = None
            _SHARED_OT_NEXT_SCORE = None

        worker_ot_prev_score = (
            ot_prev_score
            if save_parameter_logs and (ot_prev_score is None or np.isscalar(ot_prev_score))
            else None
        )
        worker_ot_next_score = (
            ot_next_score
            if save_parameter_logs and (ot_next_score is None or np.isscalar(ot_next_score))
            else None
        )
        worker_prev_score = prev_score if np.isscalar(prev_score) else None
        worker_next_score = next_score if np.isscalar(next_score) else None
        selected_seq_count = metric_utils.selected_seq_count()
        total_edge_seq_tasks = len(finite_edges) * max(selected_seq_count, 0)
        edge_batch_size = _edge_batch_size(selected_seq_count)
        print(
            f"Will evaluate {selected_seq_count} seq positions with {_SHARED_SEQ_SELECT} selection "
            f"for about {total_edge_seq_tasks} edge/seq tasks from {len(finite_edges)} edges."
        )
        print(f"Using edge batch size {edge_batch_size} per worker task.")
        if _SHARED_TOP_K == -1 and _SHARED_LPF_WINDOW > 1:
            print(f"Using sliding median low-pass filter with window={_SHARED_LPF_WINDOW}.")
        print('Start creating Pool....')

        with ctx.Pool(processes=proc, initializer=_init_worker,
            initargs=(
                curr_meta,
                base_prev_meta,
                base_next_meta,
                source_node_meta,
                target_node_meta,
                gate_beta_meta,
                prev_node_meta,
                next_node_meta,
                seq_prev_metas,
                seq_next_metas,
                prev_score_meta,
                next_score_meta,
                worker_prev_score,
                worker_next_score,
                worker_ot_prev_score,
                worker_ot_next_score,
            ),) as pool:
            for edge_batch_results in pool.imap_unordered(
                _compute_edge_batch_with_top_seq,
                _edge_batches(finite_edges, edge_batch_size),
                chunksize=1,
            ):
                for edge_results in edge_batch_results:
                    if not edge_results:
                        continue
                    parameter_log_path = None
                    for edge_res in edge_results:
                        if not edge_res:
                            continue
                        if save_parameter_logs:
                            edge_res["sample_idx"] = sample_idx
                            parameter_log_path = analysis_utils.prepare_parameter_detail_log(
                                parameter_log_root,
                                layer_id,
                                short_name,
                                sample_idx,
                                edge_res["v_idx"],
                                edge_res["u_idx"],
                            )
                            analysis_utils.append_parameter_detail_log(
                                parameter_log_root,
                                layer_id,
                                short_name,
                                edge_res,
                        )
                        source_node_zero = bool(edge_res.get("source_node_zero", False))
                        target_node_zero = bool(edge_res.get("target_node_zero", False))
                        source_node_zero_count += int(edge_res.get("source_node_zero_count", source_node_zero))
                        target_node_zero_count += int(edge_res.get("target_node_zero_count", target_node_zero))
                        if "gate_beta_zero" in edge_res:
                            gate_beta_count += int(edge_res.get("gate_beta_count", 1))
                            gate_beta_zero_count += int(edge_res.get(
                                "gate_beta_zero_count",
                                int(bool(edge_res.get("gate_beta_zero"))),
                            ))
                            if "gate_beta_sum" in edge_res:
                                gate_beta_sum += float(edge_res["gate_beta_sum"])
                                gate_beta_sum_sq += float(edge_res["gate_beta_sum_sq"])
                                gate_beta_min = min(gate_beta_min, float(edge_res["gate_beta_min"]))
                                gate_beta_max = max(gate_beta_max, float(edge_res["gate_beta_max"]))
                            else:
                                beta_value = edge_res.get("gate_beta")
                                if beta_value is not None and np.isfinite(beta_value):
                                    beta_value = float(beta_value)
                                    gate_beta_sum += beta_value
                                    gate_beta_sum_sq += beta_value * beta_value
                                    gate_beta_min = min(gate_beta_min, beta_value)
                                    gate_beta_max = max(gate_beta_max, beta_value)
                        v_idx = edge_res["v_idx"]
                        u_idx = edge_res["u_idx"]
                        curv = float(edge_res["curv"])
                        if curv < float(curvature[v_idx, u_idx]):
                            curvature[v_idx, u_idx] = curv
                        if lpf_curvature is not None and "lpf_curv" in edge_res:
                            lpf_curv = float(edge_res["lpf_curv"])
                            if lpf_curv < float(lpf_curvature[v_idx, u_idx]):
                                lpf_curvature[v_idx, u_idx] = lpf_curv
                        mu_len_total += float(edge_res["mu_len"])
                        nu_len_total += float(edge_res["nu_len"])
                        mu_nu_count += 1
                        cost_has_inf = cost_has_inf or bool(edge_res["cost_has_inf"])
                        cost_inf_count += int(edge_res["cost_inf_count"])
                    if parameter_log_path is not None:
                        analysis_utils.finalize_parameter_detail_log(parameter_log_path)

    finally:
        for shm in seq_owned_shms:
            shm.close()
            shm.unlink()

        for shm in base_owned_shms:
            shm.close()
            shm.unlink()
        _reset_shared_state()

    runtime_sec = time.time() - t1
    print(f"sample_idx = {sample_idx}, time = {runtime_sec} s, ")

    if short_name == "q_proj":
        analysis_utils.append_qproj_cost_statistics(
            analysis_path,
            sample_idx,
            q_parameter_cost_stats,
            None,
            None,
        )

    avg_mu_len = mu_len_total / mu_nu_count if mu_nu_count > 0 else float("nan")
    avg_nu_len = nu_len_total / mu_nu_count if mu_nu_count > 0 else float("nan")

    analysis_utils.append_final_min_curvature_summary(
        analysis_path=analysis_path,
        sample_idx=sample_idx,
        curvature=curvature,
        avg_mu_len=avg_mu_len,
        avg_nu_len=avg_nu_len,
        curr_dist_finite_edges=curr_dist_finite_edges,
        curr_dist_infinite_edges=curr_dist_infinite_edges,
        sampled_edge_count=len(finite_edges),
        cost_has_inf=cost_has_inf,
        cost_inf_count=cost_inf_count,
        runtime_sec=runtime_sec,
        beta_stats={
            "count": gate_beta_count,
            "expected_count": total_edge_seq_tasks,
            "source_node_zero_count": source_node_zero_count,
            "target_node_zero_count": target_node_zero_count,
            "zero_weight_parameter_count": zero_weight_parameter_count,
            "sum": gate_beta_sum,
            "sum_sq": gate_beta_sum_sq,
            "min": gate_beta_min,
            "max": gate_beta_max,
        },
        cost_stats=cost_stats,
    )

    if lpf_curvature is not None:
        return {
            "curvature": curvature,
            "lpf_curvature": lpf_curvature,
        }
    return curvature
