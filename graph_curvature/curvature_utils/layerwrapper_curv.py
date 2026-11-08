import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """Return attention output and last-row attention probabilities.

    Supports full causal prefill (L=S) and single-token decoding (L=1).
    A_last shape: [batch, query_heads, 1, key_length].
    Assumes the decoding query is the latest token.
    """
    if dropout_p != 0.0:
        raise ValueError("This collector is intended for inference (dropout_p=0).")

    if is_causal and attn_mask is not None:
        raise ValueError("Specify either is_causal or attn_mask.")

    L, S = query.size(-2), key.size(-2)

    if is_causal and L not in (1, S):
        raise NotImplementedError(
            "Chunked cached prefill requires an explicit aligned causal mask."
        )

    if enable_gqa:
        groups = query.size(-3) // key.size(-3)
        key = key.repeat_interleave(groups, dim=-3)
        value = value.repeat_interleave(groups, dim=-3)

    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    # Only compute the final query's attention probabilities.
    q_last = query[..., -1:, :]

    scores = torch.matmul(
        q_last, key.transpose(-2, -1)
    ) * scale_factor

    if attn_mask is not None:
        mask_last = attn_mask[..., -1:, :]

        if mask_last.dtype == torch.bool:
            scores = scores.masked_fill(~mask_last, float("-inf"))
        else:
            scores = scores + mask_last.to(scores.dtype)

    # For the final token of causal self-attention,
    # all keys are past or current, so no additional causal mask is needed.
    A_last = torch.softmax(
        scores, dim=-1, dtype=torch.float32
    ).to(query.dtype)

    if L == 1:
        # Decoding: one query, no need for a full attention operation.
        attn_output = A_last @ value
    else:
        # Prefill: compute outputs for all tokens without materializing full A.
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale_factor,
        )

    return attn_output, A_last


def _reshape_for_heads(x, num_heads, head_dim):
    batch, seq_len, _ = x.shape
    return x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()


def _merge_heads(x):
    batch, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)


def _repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _resolve_attention_dims(layer, model):
    attn = layer.self_attn
    config = getattr(attn, "config", None) or getattr(model, "config", None)

    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None and config is not None:
        head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = attn.q_proj.out_features // getattr(config, "num_attention_heads", 1)

    num_heads = getattr(attn, "num_heads", None)
    if num_heads is None and config is not None:
        num_heads = getattr(config, "num_attention_heads", None)
    if num_heads is None:
        num_heads = attn.q_proj.out_features // head_dim

    num_kv_heads = getattr(attn, "num_key_value_heads", None)
    if num_kv_heads is None and config is not None:
        num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = attn.k_proj.out_features // head_dim

    num_kv_groups = getattr(attn, "num_key_value_groups", None)
    if num_kv_groups is None:
        num_kv_groups = max(num_heads // num_kv_heads, 1)

    return num_heads, num_kv_heads, num_kv_groups, head_dim


def _store_operation(op_bank, name, node, dtype=None):
    node = node.detach().cpu()
    if dtype is not None and torch.is_floating_point(node):
        node = node.to(dtype=dtype)
    op_bank[name] = node


@torch.no_grad()
def collect_layer_data(layer, x, attention_mask, position_ids, model, operation_dtype=None):
    """Run one decoder layer; retain only the CPU values needed for A curvature.

    layer_input/V keep all key tokens. A keeps the final query row in its original dtype; Att_out
    and o_proj keep only that query's feature vectors. Full attention and MLP
    computation still produce the hidden states required by the next layer.
    """
    if getattr(layer.self_attn, "sliding_window", None) is not None:
        raise NotImplementedError("The curvature collector supports full attention, not sliding-window attention")
    
    operations = {}
    
    x_norm = layer.input_layernorm(x)
    _store_operation(operations, "layer_input", x_norm, operation_dtype)
    
    num_heads, num_kv_heads, repeat, head_dim = _resolve_attention_dims(layer, model)

    q = _reshape_for_heads(layer.self_attn.q_proj(x_norm), num_heads, head_dim)
    k = _reshape_for_heads(layer.self_attn.k_proj(x_norm), num_kv_heads, head_dim)
    v_linear = layer.self_attn.v_proj(x_norm)
    
    _store_operation(operations, "v_proj", v_linear, operation_dtype)
    
    v = _reshape_for_heads(v_linear, num_kv_heads, head_dim)
    del v_linear, x_norm

    if hasattr(layer.self_attn, "q_norm"):
        q, k = layer.self_attn.q_norm(q), layer.self_attn.k_norm(k)
    cos, sin = model.model.rotary_emb(x, position_ids)
    q, k = _apply_rotary_pos_emb(q, k, cos, sin)
    del cos, sin
    
    k, v = _repeat_kv(k, repeat), _repeat_kv(v, repeat)
    
    attn_output, attention = scaled_dot_product_attention(
        q, k, v, attn_mask=attention_mask,
        is_causal=attention_mask is None,
        scale=getattr(layer.self_attn, "scaling", None),
    )
    attn_context = _merge_heads(attn_output)
    _store_operation(operations, "A", attention)
    _store_operation(operations, "Att_out", attn_context[:, -1:], operation_dtype)
    del q, k, v, attention, attn_output

    projected = layer.self_attn.o_proj(attn_context)
    _store_operation(operations, "o_proj", projected[:, -1:], operation_dtype)
    del attn_context
    
    residual_multiplier = getattr(layer, "residual_multiplier", 1.0)
    residual = x + projected * residual_multiplier
    del projected
    x_out = residual + layer.mlp(layer.post_attention_layernorm(residual)) * residual_multiplier
    return x_out, operations
