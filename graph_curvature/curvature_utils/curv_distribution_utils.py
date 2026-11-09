import numpy as np
import torch

from graph_curvature.curvature_utils.curv_dtype_utils import curvature_np_dtype, curvature_torch_dtype

_NEIGHBOR_KEEP_QUANTILE = 0.7


def minmax_per_batch_nonzero_nozero(x, eps=1e-4, dim=-1):
    mask = torch.isfinite(x) & (x != 0)

    xmin = x.masked_fill(~mask, float("inf")).amin(dim=dim, keepdim=True)
    xmax = x.masked_fill(~mask, float("-inf")).amax(dim=dim, keepdim=True)

    has_nonzero = mask.any(dim=dim, keepdim=True)
    xmin = torch.where(has_nonzero, xmin, torch.zeros_like(xmin))
    xmax = torch.where(has_nonzero, xmax, torch.ones_like(xmax))

    denom = (xmax - xmin).clamp_min(eps)

    # normalize nonzero values
    norm = torch.zeros_like(x)
    norm = torch.where(mask, (x - xmin) / denom, norm)

    # push nonzero zeros upward a little so smallest nonzero won't be 0
    norm = torch.where(mask & (norm == 0), torch.full_like(norm, eps), norm)
    return norm


def _normalize_node_value_per_sequence(node_val, dim=-1):
    node_val = node_val.abs()
    node_norm = minmax_per_batch_nonzero_nozero(node_val, dim=dim)
    return 1.0 / node_norm


def _edge_distribution(dist_row_or_col, alpha):
    if dist_row_or_col is None:
        return np.array([1.0]), np.empty((0,), dtype=np.int64)

    probs = np.asarray(dist_row_or_col).reshape(-1)

    non_zero = np.nonzero(probs)[0]
    if non_zero.size == 0:
        return np.array([1.0]), np.empty((0,), dtype=np.int64)

    mass = probs[non_zero].astype(np.float64)
    if np.any(mass == -1.0):
        mass[mass == -1.0] = (1.0 - alpha) / non_zero.size
    # Preserve exact center mass; float32 summation must not perturb its balance.
    mass *= (1.0 - alpha) / mass.sum()

    return np.r_[mass, alpha], non_zero


def _renormalize_dist_from_scaled_torch(scaled, keep_mask, alpha, eps):
    weights = torch.exp(-(scaled ** 2)) * keep_mask
    sum_weights = weights.sum(dim=-1, keepdim=True)
    dist = torch.where(
        sum_weights > eps,
        ((1.0 - alpha) * weights) / sum_weights,
        torch.zeros_like(weights),
    )
    empty_mask = (sum_weights <= eps).expand_as(keep_mask)
    return torch.where(empty_mask & keep_mask, torch.full_like(dist, -1.0), dist) * keep_mask


def _quantile_keep_mask_torch(dist, keep_quantile=_NEIGHBOR_KEEP_QUANTILE):
    positive = torch.isfinite(dist) & (dist > 0)
    if keep_quantile <= 0.0 or keep_quantile >= 1.0:
        return positive

    rows = dist.reshape(-1, dist.shape[-1])
    positive_rows = positive.reshape_as(rows)
    keep_rows = torch.zeros_like(positive_rows)
    row_counts = positive_rows.sum(dim=-1)

    for row_idx in row_counts.nonzero(as_tuple=False).flatten().tolist():
        row = rows[row_idx]
        row_positive = row[positive_rows[row_idx]]
        threshold = torch.quantile(row_positive, keep_quantile)
        row_keep = positive_rows[row_idx] & (row > threshold)
        if not bool(row_keep.any()):
            row_keep = positive_rows[row_idx] & (row == row_positive.max())
        keep_rows[row_idx] = row_keep

    return keep_rows.reshape_as(dist)


def _reduce_distribution_to_top_quantile_torch(scaled, dist, alpha, eps):
    positive = torch.isfinite(dist) & (dist > 0)
    if not bool(positive.any()):
        return dist

    keep_mask = _quantile_keep_mask_torch(dist)
    reduced = _renormalize_dist_from_scaled_torch(scaled, keep_mask, alpha, eps)
    row_has_positive = positive.reshape(-1, positive.shape[-1]).any(dim=-1).reshape(
        *positive.shape[:-1],
        1,
    )
    return torch.where(row_has_positive, reduced, dist)


def _distribution_from_scaled_torch(scaled, valid_mask, alpha, eps):
    weights = torch.exp(-(scaled ** 2)) * valid_mask
    sum_weights = weights.sum(dim=-1, keepdim=True)
    dist = torch.where(
        sum_weights > eps,
        ((1.0 - alpha) * weights) / sum_weights,
        torch.zeros_like(weights),
    )
    empty_mask = (sum_weights <= eps).expand_as(valid_mask)
    dist = torch.where(empty_mask & valid_mask, torch.full_like(dist, -1.0), dist)
    dist = dist * valid_mask
    return _reduce_distribution_to_top_quantile_torch(scaled, dist, alpha, eps)


def _renormalize_dist_from_scaled_np(scaled, keep, alpha, eps):
    weights = np.exp(-(scaled ** 2)) * keep
    weight_sum = float(weights.sum())
    if weight_sum > eps:
        return (((1.0 - alpha) * weights) / weight_sum).astype(curvature_np_dtype(), copy=False)

    dist = np.zeros_like(weights, dtype=curvature_np_dtype())
    dist[keep] = -1.0
    return dist


def _distribution_from_scaled_np(scaled, valid, alpha, eps, apply_neighbor_reduction=True):
    weights = np.exp(-(scaled ** 2)) * valid
    weight_sum = float(weights.sum())
    if weight_sum > eps:
        dist = ((1.0 - alpha) * weights) / weight_sum
    else:
        dist = np.zeros_like(weights, dtype=curvature_np_dtype())
        dist[valid] = -1.0

    if apply_neighbor_reduction:
        dist = _reduce_distribution_to_top_quantile_np(scaled, dist, alpha, eps)

    return dist.astype(curvature_np_dtype(), copy=False)


def _reduce_distribution_to_top_quantile_np(scaled, dist, alpha, eps, keep_quantile=_NEIGHBOR_KEEP_QUANTILE):
    positive = np.isfinite(dist) & (dist > 0)
    if not np.any(positive):
        return dist.astype(curvature_np_dtype(), copy=False)
    if keep_quantile <= 0.0 or keep_quantile >= 1.0:
        keep = positive
    else:
        positive_values = dist[positive]
        threshold = np.quantile(positive_values, keep_quantile)
        keep = positive & (dist > threshold)
        if not np.any(keep):
            keep = positive & (dist == positive_values.max())
    return _renormalize_dist_from_scaled_np(scaled, keep, alpha, eps)


def _build_node_distribution(
    node_tensor,
    node_name,
    alpha,
    eps=1e-7,
):
    if node_tensor is None or node_tensor.numel() == 0:
        return None

    node_tensor = node_tensor.to(dtype=curvature_torch_dtype())

    if node_tensor.dim() == 3:
        if node_tensor.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1 for node tensor, got shape {tuple(node_tensor.shape)}"
            )
        node_tensor = node_tensor.squeeze(0)

    node_tensor = _normalize_node_value_per_sequence(node_tensor)

    valid_mask = torch.isfinite(node_tensor) & (node_tensor != 0)
    dist = _distribution_from_scaled_torch(node_tensor, valid_mask, alpha, eps)

    return dist.detach().cpu().numpy().astype(curvature_np_dtype(), copy=False)


def _build_node_distribution_row_from_values(values, alpha, eps=1e-7, apply_neighbor_reduction=True):
    values = np.asarray(values, dtype=curvature_np_dtype()).reshape(-1)
    node_values = np.abs(values)
    valid_raw = np.isfinite(node_values) & (node_values != 0)
    if not np.any(valid_raw):
        return np.zeros_like(node_values, dtype=curvature_np_dtype())

    xmin = np.min(node_values[valid_raw])
    xmax = np.max(node_values[valid_raw])
    denom = max(float(xmax - xmin), 1e-4)
    norm = np.zeros_like(node_values, dtype=curvature_np_dtype())
    norm[valid_raw] = (node_values[valid_raw] - xmin) / denom
    norm[valid_raw & (norm == 0)] = 1e-4

    scaled = np.zeros_like(node_values, dtype=curvature_np_dtype())
    scaled[valid_raw] = 1.0 / norm[valid_raw]
    valid = np.isfinite(scaled) & (scaled != 0)
    return _distribution_from_scaled_np(
        scaled,
        valid,
        alpha,
        eps,
        apply_neighbor_reduction=apply_neighbor_reduction,
    )
