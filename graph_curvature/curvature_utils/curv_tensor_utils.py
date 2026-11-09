"""Bounded-memory tensor helpers used by A-curvature path calculations."""

import numpy as np
import torch

from graph_curvature.curvature_utils.curv_dtype_utils import curvature_np_dtype, curvature_torch_dtype


def adaptive_chunksize(max_chunk=512, device=None):
    if device is not None and not str(device).startswith("cuda"):
        return min(max_chunk, 64), min(max_chunk, 64)
    if not torch.cuda.is_available():
        return max_chunk, max_chunk

    free_mem, _ = torch.cuda.mem_get_info(device)
    gb_free = free_mem / (1024**3)
    if gb_free < 10:
        chunk = 64
    elif gb_free < 20:
        chunk = 128
    elif gb_free < 30:
        chunk = 256
    else:
        chunk = max_chunk
    chunk = min(chunk, max_chunk)
    return chunk, chunk


def _min_plus_torch(a, b, chunk_k=256, chunk_p=256):
    """Min-plus dynamic programming with bounded row/intermediate/output blocks."""
    m, n = a.shape
    n2, p = b.shape
    assert n == n2, f"Dimension mismatch: {a.shape} vs {b.shape}"

    result = torch.full((m, p), float("inf"), dtype=a.dtype, device=a.device)

    # Bound the broadcast temporary even with many active input neighbors.
    temp_bytes = (64 if a.device.type == "cuda" else 8) * 1024**2
    block_elements = max(1, min(n, chunk_k)) * max(1, min(p, chunk_p))
    chunk_m = max(1, temp_bytes // (a.element_size() * block_elements))
    for start_m in range(0, m, chunk_m):
        end_m = min(start_m + chunk_m, m)
        for start_k in range(0, n, chunk_k):
            end_k = min(start_k + chunk_k, n)
            a_chunk = a[start_m:end_m, start_k:end_k]
            for start_p in range(0, p, chunk_p):
                end_p = min(start_p + chunk_p, p)
                b_sub = b[start_k:end_k, start_p:end_p]
                partial = (a_chunk.unsqueeze(2) + b_sub.unsqueeze(0)).amin(dim=1)
                target = result[start_m:end_m, start_p:end_p]
                torch.minimum(target, partial, out=target)

    return result


def _get_matrix_torch(layer_cache, name, device):
    matrix = layer_cache.get(f"{name}__dist")
    if matrix is None:
        return None
    if torch.is_tensor(matrix):
        target_device = torch.device(device)
        if matrix.device == target_device:
            return matrix.to(dtype=curvature_torch_dtype())
        return matrix.to(device=target_device, dtype=curvature_torch_dtype(),
                         non_blocking=target_device.type == "cuda")
    arr = np.asarray(matrix, dtype=curvature_np_dtype())
    if arr.ndim != 2:
        return None
    return torch.as_tensor(arr, dtype=curvature_torch_dtype(), device=device)
