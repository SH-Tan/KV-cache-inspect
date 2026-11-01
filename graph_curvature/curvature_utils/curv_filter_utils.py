import numpy as np

from graph_curvature.curvature_utils.curv_dtype_utils import curvature_np_dtype


def sliding_median_low_pass(values, window_size=5):
    values = np.asarray(values, dtype=curvature_np_dtype())
    window_size = int(window_size)
    if window_size <= 1 or values.size == 0:
        return values.copy()

    left = window_size // 2
    right = window_size - left
    smoothed = np.empty_like(values, dtype=curvature_np_dtype())
    for idx in range(values.size):
        start = max(0, idx - left)
        end = min(values.size, idx + right)
        window = values[start:end]
        finite_window = window[np.isfinite(window)]
        smoothed[idx] = np.median(finite_window) if finite_window.size > 0 else float("inf")
    return smoothed
