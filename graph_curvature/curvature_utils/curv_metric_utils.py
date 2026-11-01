import numpy as np

from graph_curvature.curvature_utils.curv_dtype_utils import curvature_np_dtype
from graph_curvature.curvature_utils.curv_distribution_utils import _min_reduce_blocks

_SHARED_PREV_SCORE = None
_SHARED_NEXT_SCORE = None
_SHARED_SEQ_LEN = 1
_SHARED_TOP_K = 5
_SHARED_SEQ_SELECT = "top"
_SHARED_OUT_COUNT = None
_OT_NEIGHBOR_TOP_K = 10


def set_shared_metric_state(prev_score, next_score, seq_len, top_k, seq_select="top", out_count=None):
    global _SHARED_PREV_SCORE, _SHARED_NEXT_SCORE
    global _SHARED_SEQ_LEN, _SHARED_TOP_K, _SHARED_SEQ_SELECT, _SHARED_OUT_COUNT

    _SHARED_PREV_SCORE = prev_score
    _SHARED_NEXT_SCORE = next_score
    _SHARED_SEQ_LEN = int(seq_len)
    _SHARED_TOP_K = int(top_k)
    _SHARED_SEQ_SELECT = seq_select
    _SHARED_OUT_COUNT = None if out_count is None else int(out_count)


def reset_shared_metric_state():
    set_shared_metric_state(None, None, 1, 5, "top")


def _as_seq_distribution_matrix(distribution, seq_len, short_name=None):
    if distribution is None:
        return None
    if type(distribution) is list:
        return None

    arr = np.asarray(distribution, dtype=curvature_np_dtype())
    if arr.ndim == 3 and short_name in {"q_proj", "k_proj"}:
        arr = arr.transpose(1, 0, 2).reshape(arr.shape[1], -1)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3:
        arr = arr.transpose(1, 0, 2).reshape(arr.shape[1], -1)
    if arr.ndim != 2:
        return None
    return arr[:seq_len]


def _fill_negative_distribution_entries(distribution, alpha):
    if distribution is None:
        return None

    distribution = np.asarray(distribution, dtype=curvature_np_dtype())
    neg_mask = distribution == -1.0
    if not np.any(neg_mask):
        return distribution

    distribution = distribution.copy()
    non_zero_count = np.count_nonzero(distribution, axis=1)
    fill_values = np.zeros((distribution.shape[0], 1), dtype=curvature_np_dtype())
    valid_rows = non_zero_count > 0
    fill_values[valid_rows, 0] = (1.0 - alpha) / non_zero_count[valid_rows]
    distribution[neg_mask] = np.broadcast_to(fill_values, distribution.shape)[neg_mask]
    return distribution


def _distribution_cost_score(distribution, cost, top_k=_OT_NEIGHBOR_TOP_K, transpose_cost=False):
    if distribution is None and cost is None:
        return 0.0
    if distribution is None or cost is None:
        print("Error: both distribution and cost must be present")
        return None

    distribution = np.asarray(distribution, dtype=curvature_np_dtype())
    cost = np.asarray(cost, dtype=curvature_np_dtype())

    if transpose_cost:
        cost = cost.T

    if distribution.shape[1] != cost.shape[0]:
        return None

    distribution = np.where(np.isfinite(distribution), distribution, 0.0)
    cost = np.where(np.isfinite(cost), cost, 0.0)

    seq_count, node_count = distribution.shape
    out_count = cost.shape[1]

    top_k = min(top_k, node_count)
    if top_k <= 0:
        return np.zeros((seq_count, out_count), dtype=curvature_np_dtype())

    top_idx = np.argpartition(-distribution, top_k - 1, axis=1)[:, :top_k]
    top_weights = np.take_along_axis(distribution, top_idx, axis=1)
    score = np.empty((seq_count, out_count), dtype=curvature_np_dtype())
    for start in range(0, seq_count, 64):
        end = min(start + 64, seq_count)
        gathered_cost = cost[top_idx[start:end]]
        score[start:end] = np.sum(top_weights[start:end, :, None] * gathered_cost, axis=1)
    return score.astype(curvature_np_dtype(), copy=False)


def _qk_next_score_matrix(distribution, cost, alpha, top_k=_OT_NEIGHBOR_TOP_K, in_chunk_size=256):
    if distribution is None or cost is None:
        return None

    dtype = curvature_np_dtype()
    distribution = np.asarray(distribution, dtype=dtype)
    cost = np.asarray(cost, dtype=dtype)
    if distribution.ndim != 3 or cost.ndim != 2:
        return None

    head_count, seq_count, block_width = distribution.shape
    if cost.shape[1] != head_count * block_width:
        return None

    dist_2d = _fill_negative_distribution_entries(
        distribution.reshape(-1, block_width),
        alpha,
    )
    dist = np.where(np.isfinite(dist_2d), dist_2d, 0.0).reshape(
        head_count,
        seq_count,
        block_width,
    )
    cost = np.where(np.isfinite(cost), cost, 0.0)

    keep_k = min(int(top_k), block_width)
    if keep_k <= 0:
        return np.zeros((seq_count, cost.shape[0], head_count), dtype=dtype)

    masked_dist = np.where(dist > 0, dist, -np.inf)
    top_idx = np.argpartition(-masked_dist, keep_k - 1, axis=2)[:, :, :keep_k]
    top_weights = np.take_along_axis(dist, top_idx, axis=2)

    block_cost = cost.reshape(cost.shape[0], head_count, block_width).transpose(1, 0, 2)
    score = np.empty((seq_count, cost.shape[0], head_count), dtype=dtype)
    for start in range(0, cost.shape[0], int(in_chunk_size)):
        end = min(start + int(in_chunk_size), cost.shape[0])
        gathered_cost = np.take_along_axis(
            block_cost[:, None, start:end, :],
            top_idx[:, :, None, :],
            axis=3,
        )
        chunk_score = np.sum(top_weights[:, :, None, :] * gathered_cost, axis=3)
        score[:, start:end, :] = chunk_score.transpose(1, 2, 0)
    return score


def _sanitize_score_values(score, seq_len=None):
    if score is None:
        return None
    if np.isscalar(score):
        value = float(score)
        if not np.isfinite(value):
            value = 0.0
        if seq_len is None:
            return value
        return np.full((int(seq_len),), value, dtype=curvature_np_dtype())

    score = np.asarray(score, dtype=curvature_np_dtype())
    return np.nan_to_num(score, copy=True, nan=0.0, posinf=0.0, neginf=0.0)


def build_ot_neighbor_score_matrices(
    seq_len,
    sp,
    alpha,
    prev_in_distribution,
    next_out_distribution,
    short_name=None,
    residual_start=None,
):
    next_cost = _min_reduce_blocks(sp.get("curr_in_to_next_all", {}).values())
    if short_name in {"q_proj", "k_proj"}:
        next_score = _qk_next_score_matrix(next_out_distribution, next_cost, alpha)
        next_dist = None
    else:
        next_dist = _as_seq_distribution_matrix(next_out_distribution, seq_len, short_name=short_name)
        next_dist = _fill_negative_distribution_entries(next_dist, alpha)
        next_score = _distribution_cost_score(next_dist, next_cost, transpose_cost=True)
    next_score = _sanitize_score_values(next_score)
    del next_dist, next_cost

    prev_dist = _as_seq_distribution_matrix(prev_in_distribution, seq_len)
    prev_dist = _fill_negative_distribution_entries(prev_dist, alpha)
    prev_cost = _min_reduce_blocks(sp.get("prev_to_curr_out_all", {}).values())
    if residual_start is not None and prev_dist is not None and prev_cost is not None:
        prev_cost = prev_cost[:prev_dist.shape[1], :]
    prev_score = _distribution_cost_score(prev_dist, prev_cost)
    prev_score = _sanitize_score_values(prev_score)
    del prev_dist, prev_cost

    return prev_score, next_score


def _score_column(score, idx, other_idx=None):
    if score is None:
        return None
    if np.isscalar(score):
        return _sanitize_score_values(score, seq_len=_SHARED_SEQ_LEN)
    if getattr(score, "ndim", 0) < 2:
        return None
    if score.ndim == 3:
        if other_idx is None or _SHARED_OUT_COUNT is None or score.shape[2] == 0:
            return None
        head_dim = int(_SHARED_OUT_COUNT) // int(score.shape[2])
        if head_dim <= 0:
            return None
        head_idx = int(other_idx) // head_dim
        if head_idx >= score.shape[2] or idx >= score.shape[1]:
            return None
        return _sanitize_score_values(score[:, idx, head_idx])
    if score.shape[1] == 0:
        return None
    if idx < score.shape[1]:
        return _sanitize_score_values(score[:, idx])
    return None


def top_seq_for_edge(edge):
    if _SHARED_SEQ_SELECT == "last":
        return [int(_SHARED_SEQ_LEN - 1)]

    total_count = _SHARED_SEQ_LEN if _SHARED_TOP_K == -1 else min(_SHARED_TOP_K, _SHARED_SEQ_LEN)
    if total_count <= 0:
        return []

    if _SHARED_TOP_K == -1:
        return [int(seq_idx) for seq_idx in range(_SHARED_SEQ_LEN)]

    metric, _, _ = score_components_for_edge(edge)
    np.nan_to_num(metric, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if _SHARED_SEQ_SELECT == "median":
        median_value = float(np.median(metric))
        ordered = np.argsort(np.abs(metric - median_value), kind="stable")
    else:
        ordered = np.argsort(-metric, kind="stable")
    return [int(seq_idx) for seq_idx in ordered[:total_count]]


def selected_seq_count():
    if _SHARED_SEQ_SELECT == "last":
        return 1
    total_count = _SHARED_SEQ_LEN if _SHARED_TOP_K == -1 else min(_SHARED_TOP_K, _SHARED_SEQ_LEN)
    return max(int(total_count), 0)


def score_components_for_edge(edge):
    u_idx, v_idx = (int(edge[0]), int(edge[1]))
    metric = np.zeros((_SHARED_SEQ_LEN,), dtype=curvature_np_dtype())

    prev_col = _score_column(_SHARED_PREV_SCORE, v_idx)
    next_col = _score_column(_SHARED_NEXT_SCORE, u_idx, other_idx=v_idx)
    if prev_col is not None:
        metric += prev_col
    if next_col is not None:
        metric += next_col

    metric = _sanitize_score_values(metric)

    return metric, prev_col, next_col
