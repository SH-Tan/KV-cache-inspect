import os

import numpy as np
import torch

from graph_curvature.curvature_utils.curv_dtype_utils import curvature_np_dtype, curvature_torch_dtype

_NEIGHBOR_KEEP_QUANTILE = 0.9


def minmax_per_batch_nonzero_nozero(x, eps=1e-4, dim=-1):
    mask = x != 0

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


def _resolve_node_name(operations, names):
    for name in names:
        if not name:
            continue
        if name in operations:
            return name
    return None


def _edge_distribution(dist_row_or_col, alpha):
    if dist_row_or_col is None:
        return np.array([1.0], dtype=curvature_np_dtype()), np.empty((0,), dtype=np.int64)

    probs = np.asarray(dist_row_or_col, dtype=curvature_np_dtype()).reshape(-1).copy()

    non_zero = np.nonzero(probs)[0]
    if non_zero.size == 0:
        return np.array([1.0], dtype=curvature_np_dtype()), np.empty((0,), dtype=np.int64)

    if np.any(probs == -1.0):
        tmp = (1.0 - alpha) / non_zero.size
        probs[probs == -1.0] = tmp

    return np.hstack((probs[non_zero], np.array([alpha], dtype=curvature_np_dtype()))), non_zero


def _min_reduce_blocks(blocks):
    """Elementwise minimum without stacking dense matrices or modifying inputs."""
    blocks = list(blocks)
    if not blocks:
        return None
    if len(blocks) == 1:
        return np.asarray(blocks[0], dtype=curvature_np_dtype())
    result = np.array(blocks[0], dtype=curvature_np_dtype(), copy=True)
    for block in blocks[1:]:
        np.minimum(result, np.asarray(block, dtype=curvature_np_dtype()), out=result)
    return result


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


def _all_examples_l2_tensor(node_tensor, l2_reference, node_name):
    ref_tensor = torch.as_tensor(l2_reference, dtype=curvature_torch_dtype()).reshape(1, -1)
    if ref_tensor.shape[-1] != node_tensor.shape[-1]:
        raise ValueError(
            f"Invalid {node_name} all-example L2 reference width "
            f"{ref_tensor.shape[-1]}, expected {node_tensor.shape[-1]}"
        )
    return ref_tensor


def _build_node_distribution(
    node_tensor,
    node_name,
    alpha,
    eps=1e-7,
    l2_norm=False,
    l2_norm_mode="per_example",
    l2_reference=None,
):
    if node_tensor is None or node_tensor.numel() == 0:
        return None

    if l2_norm and l2_norm_mode == "all_examples" and l2_reference is not None:
        node_tensor = _all_examples_l2_tensor(node_tensor, l2_reference, node_name)
    else:
        node_tensor = node_tensor.to(dtype=curvature_torch_dtype())

    if node_tensor.dim() == 3:
        if node_tensor.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1 for node tensor, got shape {tuple(node_tensor.shape)}"
            )
        node_tensor = node_tensor.squeeze(0)

    if l2_norm and node_tensor.dim() == 2:
        node_tensor = torch.norm(node_tensor, p=2, dim=0, keepdim=True)

    node_tensor = _normalize_node_value_per_sequence(node_tensor)

    valid_mask = torch.isfinite(node_tensor) & (node_tensor != 0)
    dist = _distribution_from_scaled_torch(node_tensor, valid_mask, alpha, eps)

    return dist.detach().cpu().numpy().astype(curvature_np_dtype(), copy=False)


def _build_distribution_node_values(
    node_tensor,
    seq_len,
    l2_norm=False,
    l2_norm_mode="per_example",
    l2_reference=None,
):
    if node_tensor is None:
        return None

    if l2_norm and l2_norm_mode == "all_examples" and l2_reference is not None:
        tensor = _all_examples_l2_tensor(node_tensor, l2_reference, "node")
    elif torch.is_tensor(node_tensor):
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
                f"Expected batch size 1 for node values, got shape {tuple(tensor.shape)}"
            )
        tensor = tensor.squeeze(0)
    elif tensor.dim() == 1:
        tensor = tensor.reshape(1, -1)
    elif tensor.dim() != 2:
        return None

    if l2_norm:
        tensor = torch.norm(tensor, p=2, dim=0, keepdim=True)
    else:
        tensor = tensor[:int(seq_len)].contiguous()

    values = tensor.detach().cpu().numpy().astype(curvature_np_dtype(), copy=False)
    return np.abs(values)


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


def _build_block_row_node_distribution(
    node_tensor,
    alpha,
    eps=1e-7,
    l2_norm=False,
    l2_reference=None,
):
    if node_tensor is None or node_tensor.numel() == 0:
        return None

    node_tensor = node_tensor.to(dtype=curvature_torch_dtype())

    if l2_norm and l2_reference is not None:
        ref_tensor = torch.as_tensor(
            l2_reference,
            dtype=curvature_torch_dtype(),
            device=node_tensor.device,
        )

        expected_shape = (node_tensor.shape[0], node_tensor.shape[2])

        if ref_tensor.shape == expected_shape:
            node_tensor = ref_tensor.unsqueeze(1)
        elif ref_tensor.numel() == int(np.prod(expected_shape)):
            node_tensor = ref_tensor.reshape(*expected_shape).unsqueeze(1)
        else:
            raise ValueError(
                f"Invalid l2_reference shape {tuple(ref_tensor.shape)}, "
                f"expected {expected_shape} or numel={int(np.prod(expected_shape))}"
            )

    if l2_norm:
        node_tensor = torch.norm(node_tensor, p=2, dim=1, keepdim=True)

    # Normalize each row over the feature axis, then distribution-normalize
    # each row over the same axis.
    # q_proj: [q_heads, seq_q, seq_k] -> per (head, seq_q) over seq_k.
    # k_proj: [k_heads, seq_k, seq_q * repeat] -> per (head, seq_k) over features.
    # L2: [heads, 1, feature_axis] -> per head over feature_axis.
    node_tensor = _normalize_node_value_per_sequence(node_tensor, dim=-1)

    valid_mask = torch.isfinite(node_tensor) & (node_tensor != 0)
    dist = _distribution_from_scaled_torch(node_tensor, valid_mask, alpha, eps)

    return dist.detach().cpu().numpy().astype(
        curvature_np_dtype(),
        copy=False,
    )


def _check_qk_l2_reference_shape(l_name, l2_reference, expected_shape):
    if l2_reference is None:
        return

    ref_shape = tuple(torch.as_tensor(l2_reference).shape)
    if ref_shape != tuple(expected_shape):
        raise ValueError(
            f"Invalid {l_name} A L2 reference shape {ref_shape}, "
            f"expected {tuple(expected_shape)}"
        )


def _build_qk_out_node_distribution(
    l_name,
    node,
    alpha,
    l2_norm=False,
    l2_norm_mode="per_example",
    l2_reference=None,
    repeat=1,
):
    if node is None or node.numel() == 0:
        return None

    if node.dim() != 4:
        raise ValueError(
            f"Expected A node shape [B, q_heads, seq, seq], got {tuple(node.shape)}"
        )

    if l2_norm and l2_norm_mode == "all_examples" and l2_reference is not None:
        A = node[0].contiguous()
    elif node.shape[0] != 1:
        raise ValueError(
            f"Expected batch size 1 for A node, got shape {tuple(node.shape)}"
        )
    else:
        # [1, q_heads, seq_q, seq_k] -> [q_heads, seq_q, seq_k]
        A = node.squeeze(0).contiguous()
    if A.shape[0] == A.shape[2] and A.shape[1] < A.shape[0]:
        A = A.permute(1, 0, 2).contiguous()

    q_heads, seq_q, seq_k = A.shape
    repeat = max(int(repeat), 1)

    if l_name == "q_proj":
        # Q node:
        # For each q head h and query token i:
        # node = A[h, i, :]
        #
        # Shape: [q_heads, seq_q, seq_k].
        # Distribution rows are per (q_head, seq_q) over seq_k.
        out_node = A
        if l2_norm and l2_reference is not None:
            _check_qk_l2_reference_shape(l_name, l2_reference, (q_heads, seq_k))

    elif l_name == "k_proj":
        # K node:
        # q_heads = k_heads * repeat
        #
        # Original:
        #   A: [q_heads, seq_q, seq_k]
        #
        # Reshape repeated Q heads into K-head groups:
        #   [k_heads, repeat, seq_q, seq_k]
        #
        # Then for each K head and key token j, collect all repeated
        # Q-head/query-token attention values:
        #   [k_heads, seq_k, seq_q * repeat]
        # Feature axis order is [repeat, seq_q], matching _get_qk_next_cost.
        #
        # Distribution rows are per (k_head, seq_k) over seq_q * repeat.
        if q_heads % repeat != 0:
            raise ValueError(
                f"q_heads={q_heads} is not divisible by repeat={repeat}"
            )

        k_heads = q_heads // repeat

        out_node = (
            A.reshape(k_heads, repeat, seq_q, seq_k)
             .permute(0, 3, 1, 2)          # [k_heads, seq_k, repeat, seq_q]
             .reshape(k_heads, seq_k, seq_q * repeat)
             .contiguous()
        )
        if l2_norm and l2_reference is not None:
            _check_qk_l2_reference_shape(l_name, l2_reference, (k_heads, seq_q * repeat))

    else:
        raise ValueError(f"Unsupported l_name={l_name}")

    return _build_block_row_node_distribution(
        out_node,
        alpha=alpha,
        l2_norm=l2_norm,
        l2_reference=l2_reference,
    )


def draw_nonzero_distribution_curve(dist,
    node_name=None,
    save_path=None,
    ignore_negative=True,
    dpi=300,):
    """
    Draw smooth curve distribution of non-zero values in final dist.
    No histogram bins.
    """
    if hasattr(dist, "detach"):
        values = dist.detach().cpu().numpy()
    else:
        values = np.asarray(dist)

    values = values.reshape(-1)

    if ignore_negative:
        nonzero_values = values[
            (values != 0) & np.isfinite(values) & (values > 0)
        ]
    else:
        nonzero_values = values[
            (values != 0) & np.isfinite(values)
        ]

    nonzero_count = nonzero_values.size
    print(f"[{node_name}] non-zero value number: {nonzero_count}")

    if nonzero_count == 0:
        print("No non-zero values to draw.")
        return nonzero_values

    if nonzero_count == 1:
        print(f"Only one non-zero value: {nonzero_values[0]}")
        return nonzero_values

    sorted_values = np.sort(nonzero_values)
    value_rank = np.arange(nonzero_count)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(value_rank, sorted_values, linewidth=1.0)
    ax.set_xlabel("Sorted non-zero value index")
    ax.set_ylabel("Non-zero distribution value")
    title = f"Sorted non-zero values, n={nonzero_count}"
    if node_name is not None:
        title += f" - {node_name}"
    ax.set_title(title)
    stats_text = (
        f"min={sorted_values[0]:.4g}\n"
        f"p50={np.percentile(sorted_values, 50):.4g}\n"
        f"p95={np.percentile(sorted_values, 95):.4g}\n"
        f"max={sorted_values[-1]:.4g}"
    )
    ax.text(
        0.98,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    plt.close(fig)
    return nonzero_values
