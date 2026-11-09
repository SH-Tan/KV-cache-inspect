"""Reuse saved curvature scores with kvpress's prefill compression hooks."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import kvpress
from kvpress import PyramidKVPress, ScorerPress, SnapKVPress
from kvpress.utils import compute_n_kept, get_query_states
from transformers.models.llama.modeling_llama import repeat_kv


def model_window_attention(module, hidden_states, keys, window_size, position_embeddings):
    """Reconstruct SnapKV's window attention with the model's actual QK scale."""
    cos, sin = position_embeddings
    query = get_query_states(module, hidden_states[:, -window_size:],
                             (cos[:, -window_size:], sin[:, -window_size:]))
    key = repeat_kv(keys, query.shape[1] // keys.shape[1])
    weights = query @ key.transpose(-2, -1)
    weights.mul_(getattr(module, "scaling", module.head_dim ** -0.5))
    length = keys.shape[2]
    future = torch.arange(length, device=keys.device)[None, :] > torch.arange(
        length - window_size, length, device=keys.device,
    )[:, None]
    weights.masked_fill_(future, -torch.inf)
    return weights.softmax(dim=-1, dtype=torch.float32).to(query.dtype)[..., :-window_size]


@dataclass
class ModelScaledSnapKVPress(SnapKVPress):
    compute_window_attention = staticmethod(model_window_attention)


@dataclass
class SavedCurvaturePress(ScorerPress):
    curvature_dir: Path = Path(".")
    reduction: str = "min"
    prompt_id: int = 0

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        path = Path(self.curvature_dir) / f"layer_{module.layer_idx:03d}" / f"prompt_{self.prompt_id:05d}.pt"
        record = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if record["token_count"] != keys.shape[2] or record["query_index"] != keys.shape[2] - 1:
            raise ValueError(f"Saved curvature does not match the inference prompt length: {path}")
        curvature = record["curvature_scores"]["A"]
        finite = torch.isfinite(curvature)
        if self.reduction == "min":
            reduced = curvature.masked_fill(~finite, torch.inf).amin(dim=0)
        elif self.reduction == "mean":
            reduced = curvature.masked_fill(~finite, 0).sum(dim=0, dtype=torch.float64) / finite.sum(dim=0)
        else:
            raise ValueError(f"Unknown curvature reduction: {self.reduction}")
        # Remove highest raw curvature first by retaining the lowest curvature.
        # kvpress keeps the highest importance scores; undefined tokens are last.
        importance = (-reduced).masked_fill(~torch.isfinite(reduced), -torch.inf)
        return importance.to(keys.device).view(1, 1, -1).expand(keys.shape[:3])


@dataclass
class BudgetMatchedPyramidKVPress(PyramidKVPress):
    """Keep PyramidKV's layer allocation, correcting rounding to an exact total."""

    compute_window_attention = staticmethod(model_window_attention)

    def get_layer_budget(self, module, q_len):
        if getattr(self, "_budget_length", None) != q_len:
            layers = module.config.num_hidden_layers
            budgets = [
                super(BudgetMatchedPyramidKVPress, self).get_layer_budget(
                    SimpleNamespace(config=module.config, layer_idx=index), q_len,
                )
                for index in range(layers)
            ]
            remaining = layers * compute_n_kept(q_len, self.compression_ratio) - sum(budgets)
            while remaining:
                step = 1 if remaining > 0 else -1
                indices = range(layers) if step > 0 else range(layers - 1, -1, -1)
                for index in indices:
                    budget = budgets[index]
                    if 1 <= budget + step <= q_len:
                        budgets[index] += step
                        remaining -= step
                        if not remaining:
                            break
            self._budget_length, self._budgets = q_len, budgets
        return self._budgets[module.layer_idx]


def make_press(method, ratio, curvature_dir=".", **kwargs):
    """Create the same prefill-only methods in both comparison entry points."""
    if method in ("curvature_min", "curvature_mean"):
        return SavedCurvaturePress(compression_ratio=ratio, curvature_dir=Path(curvature_dir),
                                   reduction=method.removeprefix("curvature_"), **kwargs)
    if method == "PyramidKVPress":
        cls = BudgetMatchedPyramidKVPress
    elif method == "SnapKVPress":
        cls = ModelScaledSnapKVPress
    else:
        cls = getattr(kvpress, method)
    if cls.forward_hook is not kvpress.BasePress.forward_hook:
        raise ValueError(f"{method} does not use prefill-only compression")
    return cls(compression_ratio=ratio, **kwargs)
