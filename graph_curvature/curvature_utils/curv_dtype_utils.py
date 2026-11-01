import numpy as np
import torch


_CURVATURE_DTYPE_NAME = "float32"


def set_curvature_dtype(dtype_name):
    global _CURVATURE_DTYPE_NAME
    if dtype_name not in {"float32", "float64"}:
        raise ValueError(f"Unsupported curvature dtype: {dtype_name}")
    _CURVATURE_DTYPE_NAME = dtype_name


def curvature_np_dtype():
    return np.float32 if _CURVATURE_DTYPE_NAME == "float32" else np.float64


def curvature_torch_dtype():
    return torch.float32 if _CURVATURE_DTYPE_NAME == "float32" else torch.float64
