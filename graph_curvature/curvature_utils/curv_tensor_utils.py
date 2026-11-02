import numpy as np
import torch
from graph_curvature.curvature_utils.curv_dtype_utils import curvature_np_dtype, curvature_torch_dtype
from graph_curvature.curvature_utils.layerwrapper_curv import (
    _reshape_for_heads,
    _repeat_kv,
)

from graph_curvature.curvature_utils.curv_model_utils import _operation_distance_matrix_torch


def _residual_distance_matrix_torch(operations, name, device):
    residual = operations.get(name)
    if residual is None:
        return None
    width = int(residual.shape[-1])
    dist = torch.full(
        (width, width),
        float("inf"),
        dtype=curvature_torch_dtype(),
        device=device,
    )
    idx = torch.arange(width, device=device)
    dist[idx, idx] = 1.0
    return dist


def adaptive_chunksize(max_chunk=512, device=None):
    if device is not None and not str(device).startswith("cuda"):
        return min(max_chunk, 64), min(max_chunk, 64)
    if not torch.cuda.is_available():
        return max_chunk, max_chunk

    free_mem, _ = torch.cuda.mem_get_info(device)
    gb_free = free_mem / (1024**3)
    if gb_free < 10:
        return 64, 64
    elif gb_free < 20:
        return 128, 128
    elif gb_free < 30:
        return 256, 256
    else:
        return max_chunk, max_chunk


def _min_plus_torch(a, b, chunk_k=256, chunk_p=256):
    m, n = a.shape
    n2, p = b.shape
    assert n == n2, f"Dimension mismatch: {a.shape} vs {b.shape}"

    result = torch.full((m, p), float("inf"), dtype=a.dtype, device=a.device)

    for start_k in range(0, n, chunk_k):
        end_k = min(start_k + chunk_k, n)
        a_chunk = a[:, start_k:end_k]
        b_chunk = b[start_k:end_k, :]

        for start_p in range(0, p, chunk_p):
            end_p = min(start_p + chunk_p, p)
            b_sub = b_chunk[:, start_p:end_p]
            partial = (a_chunk.unsqueeze(2) + b_sub.unsqueeze(0)).min(dim=1).values
            result[:, start_p:end_p] = torch.minimum(result[:, start_p:end_p], partial)

    return result


def _get_matrix_torch(layer_cache, name, device):
    matrix = layer_cache.get(f"{name}__dist")
    if matrix is None:
        return None
    if torch.is_tensor(matrix):
        target_device = torch.device(device)
        if matrix.device == target_device:
            return matrix.to(dtype=curvature_torch_dtype())
        return matrix.to(device=target_device, dtype=curvature_torch_dtype(), non_blocking=True)
    arr = np.asarray(matrix, dtype=curvature_np_dtype())
    if arr.ndim != 2:
        return None
    return torch.as_tensor(arr, dtype=curvature_torch_dtype(), device=device)


def _all_cost_matrices(layer_cache, names, device):
    matrices = {}
    for name in names:
        if not name:
            continue
        matrix = _get_matrix_torch(layer_cache, name, device)
        if matrix is not None:
            matrices[name] = matrix
    return matrices


def build_layer_cache(model, operations, layer_id, cache=None, device="cuda", required_names=None):
    """
    Build or update the layer cache with distance matrices.
    Keep this once per layer/model state, then reuse across samples.
    """
    if cache is None:
        cache = {}
    if required_names is not None:
        required_names = set(required_names)

    for name in operations.keys():
        if required_names is not None and name not in required_names:
            continue
        if name in {"layer_input", "A", "Att_out", "gate_up_out", "x_norm2", "gate_beta"}:
            continue

        if name.startswith("prev_"):
            real_name = name.replace("prev_", "")
            if real_name in {"layer_input", "A", "Att_out", "gate_up_out", "x_norm2", "gate_beta"}:
                continue

        if name.endswith("_residual"):
            dist_matrix = _residual_distance_matrix_torch(operations, name, device)
            if dist_matrix is None:
                continue
        else:
            dist_matrix = _operation_distance_matrix_torch(model, name, layer_id, device)
        cache[f"{name}__dist"] = dist_matrix.cpu().contiguous()
        del dist_matrix
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return cache


def _get_qk_next_cost(cost, name, meta, device = "cpu"):
    head_dim = meta["head_dim"]
    num_q_heads = meta["num_q_heads"]
    num_kv_heads = meta["num_kv_heads"]
    repeat = meta["repeat"]

    cost = cost.to(curvature_torch_dtype())
    b, seq, d = cost.shape
    dtype = cost.dtype

    if name == "q_proj":
        # q node -> A uses k node values
        assert d == num_kv_heads * head_dim, (
            f"k_proj dim mismatch: got {d}, expected {num_kv_heads * head_dim}"
        )

        kh = _reshape_for_heads(cost, num_kv_heads, head_dim)   # [B, kv_heads, S, D]
        kh_rep = _repeat_kv(kh, repeat)                         # [B, q_heads, S, D]

        c = 1.0 / (kh_rep.abs())                          # [B, q_heads, S, D]
        c = c.mean(dim=0)                                       # [q_heads, S, D]
        c = c.permute(0, 2, 1).contiguous()                    # [q_heads, D, S]

        # Build sparse block matrix: [q_heads*D, q_heads*S]
        out = torch.full(
            (num_q_heads * head_dim, num_q_heads * seq),
            float("inf"),
            device=device,
            dtype=dtype,
        )

        for h in range(num_q_heads):
            r0 = h * head_dim
            r1 = (h + 1) * head_dim
            c0 = h * seq
            c1 = (h + 1) * seq
            out[r0:r1, c0:c1] = c[h]

        return out

    elif name == "k_proj":
        # k node -> A uses q node values
        assert d == num_q_heads * head_dim, (
            f"q_proj dim mismatch: got {d}, expected {num_q_heads * head_dim}"
        )

        qh = _reshape_for_heads(cost, num_q_heads, head_dim)    # [B, q_heads, S, D]

        q_cost = 1.0 / (qh.abs())                         # [B, q_heads, S, D]
        q_cost = q_cost.mean(dim=0)                             # [q_heads, S, D]

        # Group repeated q-heads back under each kv-head
        q_cost = q_cost.view(num_kv_heads, repeat, seq, head_dim)
        q_cost = q_cost.permute(0, 3, 1, 2).contiguous()       # [kv_heads, D, repeat, S]
        q_cost = q_cost.view(num_kv_heads, head_dim, repeat * seq)

        # Output shape: [kv_heads*D, q_heads*S]
        out = torch.full(
            (num_kv_heads * head_dim, num_q_heads * seq),
            float("inf"),
            device=device,
            dtype=dtype,
        )

        for kvh in range(num_kv_heads):
            r0 = kvh * head_dim
            r1 = (kvh + 1) * head_dim

            # This kv head connects to its repeated q-head group
            qh0 = kvh * repeat
            qh1 = (kvh + 1) * repeat
            c0 = qh0 * seq
            c1 = qh1 * seq

            out[r0:r1, c0:c1] = q_cost[kvh]

        return out

    else:
        raise ValueError(f"Unsupported name={name}")


def _build_v_to_att_out_template(cost, meta, reduce_batch=True):
    head_dim = meta["head_dim"]
    num_kv_heads = meta["num_kv_heads"]
    repeat = meta["repeat"]

    cost = cost.to(curvature_torch_dtype())

    b, seq, d = cost.shape
    assert d == num_kv_heads * head_dim

    vh = _reshape_for_heads(cost, num_kv_heads, head_dim)   # [B, kv_heads, S, D]
    vh_rep = _repeat_kv(vh, repeat)                         # [B, q_heads, S, D]
    vh_abs = vh_rep.abs()

    template = 1.0 / vh_abs                  # [B, q_heads, S, D]

    if reduce_batch:
        template = template.mean(dim=0)                     # [q_heads, S, D]

    return template


def _build_x_to_out_cost(v, sp_q, meta, device):
    head_dim = meta["head_dim"]
    num_q_heads = meta["num_q_heads"]

    assert v.ndim == 3, f"Expected v shape [q_heads, seq, head_dim], got {v.shape}"
    qh, seq, d = v.shape
    assert qh == num_q_heads
    assert d == head_dim

    dtype = v.dtype

    if not torch.is_tensor(v):
        v = torch.as_tensor(v, device=device)
    else:
        v = v.to(device)

    # A -> out : [q_heads*seq, q_heads*head_dim]
    out = torch.full(
        (num_q_heads * seq, num_q_heads * head_dim),
        float("inf"),
        device=device,
        dtype=dtype,
    )

    for h in range(num_q_heads):
        r0 = h * seq
        r1 = (h + 1) * seq
        c0 = h * head_dim
        c1 = (h + 1) * head_dim

        # v[h]: [seq, head_dim]
        out[r0:r1, c0:c1] = v[h]

    cost = {}

    # input to A
    if sp_q["prev_to_next_all"]:
        a = next(iter(sp_q["prev_to_next_all"].values()))

        if not torch.is_tensor(a):
            a = torch.as_tensor(a, device=device)
        else:
            a = a.to(device)

        chunk_k, chunk_p = adaptive_chunksize(device=device)
        res = _min_plus_torch(a, out, chunk_k=chunk_k, chunk_p=chunk_p)

        cost["prev_to_next_all"] = res.cpu().contiguous().numpy()
        del a, res


    a = sp_q["curr_in_to_next_all"]["k_proj"]

    if not torch.is_tensor(a):
        a = torch.as_tensor(a, device=device)
    else:
        a = a.to(device)

    chunk_k, chunk_p = adaptive_chunksize(device=device)
    res = _min_plus_torch(a, out, chunk_k=chunk_k, chunk_p=chunk_p)

    cost["curr_in_to_next_all"] = res.cpu().contiguous().numpy()
    del a, out, res, v
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cost


