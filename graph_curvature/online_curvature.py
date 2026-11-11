"""Calculate last-query curvature from the live prefill, before KV compression."""

from contextlib import contextmanager
from pathlib import Path

import torch
from kvpress.utils import get_query_states

from graph_curvature.curvature_utils.cal_curvature_A import compute_a_curvature
from graph_curvature.curvature_utils.curv_analysis_utils import _summarize_curvatures, save_a_curvature
from graph_curvature.curvature_utils.curv_dtype_utils import curvature_torch_dtype
from graph_curvature.model_utils import build_a_layer_cache


class OnlineCurvature:
    def __init__(self, model, settings, output_dir):
        self.model, self.settings = model, settings
        self.output_dir = Path(output_dir)
        self.prompt_id = 0
        self.projection = None
        if any(getattr(layer.self_attn, "sliding_window", None) is not None
               for layer in model.model.layers):
            raise NotImplementedError("Online curvature requires full attention, not sliding-window attention")

    @contextmanager
    def capture(self, prompt_length):
        """Retain only the current layer's last projected query; ignore decoding."""
        def record_projection(module, inputs, output):
            if inputs[0].shape[1] == prompt_length and self.projection is None:
                dtype = curvature_torch_dtype()
                self.projection = (
                    inputs[0][:, -1:].detach().to(device="cpu", dtype=dtype, copy=True),
                    output[:, -1:].detach().to(device="cpu", dtype=dtype, copy=True),
                )

        handles = [layer.self_attn.o_proj.register_forward_hook(record_projection)
                   for layer in self.model.model.layers]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            self.projection = None

    @torch.no_grad()
    def __call__(self, module, hidden_states, keys, values, attentions, kwargs):
        """Use the native, uncompressed layer cache and actual O-projection output."""
        cos, sin = kwargs["position_embeddings"]
        query = get_query_states(module, hidden_states[:, -1:], (cos[:, -1:], sin[:, -1:]))
        batch, kv_heads, length, dim = keys.shape
        groups = query.shape[1] // kv_heads
        # Broadcast GQA groups instead of copying the full key cache across heads.
        logits = query.reshape(batch, kv_heads, groups, 1, dim) @ keys.unsqueeze(2).transpose(-2, -1)
        logits = logits.reshape(batch, kv_heads * groups, 1, length)
        logits.mul_(getattr(module, "scaling", dim ** -0.5))
        attention = logits.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
        dtype = curvature_torch_dtype()
        attn_out, projected = self.projection
        self.projection = None
        operations = {
            "layer_input": hidden_states.detach().to(device="cpu", dtype=dtype),
            "v_proj": values.detach().to(device="cpu", dtype=dtype).transpose(1, 2).reshape(batch, length, -1),
            "A": attention.cpu(), "Att_out": attn_out, "o_proj": projected,
        }
        del logits, query, attention
        layer_cache = build_a_layer_cache(self.model, module.layer_idx, self.settings["compute_device"])
        curvature, metrics = compute_a_curvature(
            operations, layer_cache, alpha=self.settings["alpha"],
            device=self.settings["compute_device"], workers=self.settings["workers"],
        )
        if self.settings["save_curvature"]:
            metrics["curvature"] = _summarize_curvatures(curvature)
            path = self.output_dir / f"layer_{module.layer_idx:03d}" / f"prompt_{self.prompt_id:05d}.pt"
            save_a_curvature(path, module.layer_idx, self.prompt_id, operations, curvature, metrics)
        print(f"Online curvature: prompt={self.prompt_id} layer={module.layer_idx}", flush=True)
        return curvature
