import torch

from graph_curvature.curvature_utils.curv_dtype_utils import curvature_torch_dtype

def _weight_from_model(model, short_name, layer_id, device=None):
    if short_name.startswith("prev_"):
        real_name = short_name.replace("prev_", "")
        if layer_id == 0:
            return None
        return _weight_from_model(model, real_name, layer_id - 1, device=device)

    if short_name == "lm_head":
        w = model.lm_head.weight.detach()
    else:
        layer = model.model.layers[layer_id]
        if short_name == "q_proj":
            w = layer.self_attn.q_proj.weight.detach()
        elif short_name == "k_proj":
            w = layer.self_attn.k_proj.weight.detach()
        elif short_name == "v_proj":
            w = layer.self_attn.v_proj.weight.detach()
        elif short_name == "o_proj":
            w = layer.self_attn.o_proj.weight.detach()
        elif short_name == "gate_proj":
            w = layer.mlp.gate_proj.weight.detach()
        elif short_name == "up_proj":
            w = layer.mlp.up_proj.weight.detach()
        elif short_name == "down_proj":
            w = layer.mlp.down_proj.weight.detach()
        else:
            raise KeyError(short_name)

    if device is None:
        return w

    target_device = torch.device(device)
    if w.device.type == "cuda" and target_device.type == "cuda" and w.device != target_device:
        # Direct cross-GPU bf16 copies can silently zero weights on this setup.
        return w.cpu().to(device=target_device, non_blocking=True)
    return w.to(device=target_device, non_blocking=True)


def _operation_distance_matrix_torch(model, operations, short_name, layer_id, device):
    weight = _weight_from_model(model, short_name, layer_id, device=device)
    abs_w = weight.abs().to(dtype=curvature_torch_dtype())
    weight_norm = torch.where(abs_w > 0, 1.0 / abs_w, torch.full_like(abs_w, float("inf")))
    dist = weight_norm.transpose(0, 1).contiguous()
    expected_shape = (weight.shape[1], weight.shape[0])
    del weight, abs_w, weight_norm
    if dist.shape != expected_shape:
        raise ValueError(
            f"Unexpected cost matrix shape for {short_name}: "
            f"got {tuple(dist.shape)}, expected {expected_shape}"
        )
    return dist
