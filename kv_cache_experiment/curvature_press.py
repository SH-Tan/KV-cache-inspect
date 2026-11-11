"""Reuse saved curvature scores with kvpress's prefill compression hooks."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F
import kvpress
from kvpress import PyramidKVPress, ScorerPress, SnapKVPress
from kvpress.utils import compute_n_kept, get_query_states
from transformers.models.llama.modeling_llama import repeat_kv


CURVATURE_SCORE_METHODS = ("SnapKVPress", "PyramidKVPress", "ExpectedAttentionPress")


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
    curvature_provider: object = None

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.curvature_provider is not None:
            curvature = self.curvature_provider(module, hidden_states, keys, values, attentions, kwargs)
        else:
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


def curvature_method_score(press, scorer, module, hidden_states, keys, values, attentions, kwargs):
    """Replace attention importance, retaining the method's pooling/window or sink policy.

    Curvature is available for the last prompt query only, not a multi-query window.
    """
    scores = scorer.score(module, hidden_states, keys, values, attentions, kwargs)
    if isinstance(press, SnapKVPress):
        assert hidden_states.shape[1] > press.window_size, "Prompt must exceed the SnapKV window"
        scores = scores[..., :-press.window_size]
        finite = torch.isfinite(scores)
        pooling = dict(kernel_size=press.kernel_size, padding=press.kernel_size // 2, stride=1)
        pooled = F.avg_pool1d(scores.masked_fill(~finite, 0), **pooling)
        counts = F.avg_pool1d(finite.to(scores.dtype), **pooling)
        scores = (pooled / counts.clamp_min(torch.finfo(scores.dtype).tiny)).masked_fill(~finite, -torch.inf)
        return F.pad(scores, (0, press.window_size), value=torch.inf)
    # ExpectedAttention's predicted attention is replaced by curvature importance.
    # Softmax supplies nonnegative weights before its optional value-norm scaling.
    assert keys.shape[2] > press.n_sink, "Prompt must exceed the sink-token count"
    scores = scores[..., press.n_sink:].softmax(dim=-1)
    scores = scores.masked_fill(~torch.isfinite(scores), 0)
    if press.use_vnorm:
        scores = (scores + press.epsilon) * values[..., press.n_sink:, :].norm(dim=-1)
    return F.pad(scores, (press.n_sink, 0), value=torch.inf)


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


def make_press(method, ratio, curvature_dir=".", *, score_source="native", curvature_reduction="mean",
               curvature_provider=None, **kwargs):
    """Create the same prefill-only methods in both comparison entry points."""
    if method in ("curvature_min", "curvature_mean"):
        return SavedCurvaturePress(compression_ratio=ratio, curvature_dir=Path(curvature_dir),
                                   reduction=method.removeprefix("curvature_"),
                                   curvature_provider=curvature_provider, **kwargs)
    if method == "PyramidKVPress":
        cls = BudgetMatchedPyramidKVPress
    elif method == "SnapKVPress":
        cls = ModelScaledSnapKVPress
    else:
        cls = getattr(kvpress, method)
    if cls.forward_hook is not kvpress.BasePress.forward_hook:
        raise ValueError(f"{method} does not use prefill-only compression")
    press = cls(compression_ratio=ratio, **kwargs)
    if score_source == "curvature" and method in CURVATURE_SCORE_METHODS:
        press.curvature_scorer = SavedCurvaturePress(
            curvature_dir=Path(curvature_dir), reduction=curvature_reduction,
            curvature_provider=curvature_provider,
        )
        press.score = partial(curvature_method_score, press, press.curvature_scorer)
    return press
