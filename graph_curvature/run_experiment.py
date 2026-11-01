#!/usr/bin/env python3
"""Extract an LLM graph and calculate gate/up curvature without pruning weights."""

import argparse
import json
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from graph_curvature.curvature_utils import cal_curvature
from graph_curvature.curvature_utils.curv_dtype_utils import (
    curvature_torch_dtype, set_curvature_dtype,
)
from graph_curvature.curvature_utils.curv_tensor_utils import build_layer_cache
from graph_curvature.curvature_utils.graph_relation import GRAPH
from graph_curvature.curvature_utils.layerwrapper_curv import collect_layer_data
from graph_curvature.model_utils import capture_layer_inputs, required_cache_names


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--cache-dir", default=str(REPO_DIR / "llm_weights"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="The capital of France is Paris.")
    parser.add_argument("--prompts-file", help="JSON array of calibration prompt strings")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compute-device", default="cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--curvature-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--ops", nargs="+", choices=("gate_proj", "up_proj"),
                        default=["gate_proj", "up_proj"])
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, help="Exclusive layer index; defaults to all layers")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--sample-edge-num", type=int, default=100)
    parser.add_argument("--sample-edge-ratio", type=float, default=1.0)
    parser.add_argument("--shared-top-k", type=int, default=10)
    parser.add_argument("--shared-seq-select", choices=("top", "median", "last"), default="top")
    parser.add_argument("--l2-norm", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    args.ops = list(dict.fromkeys(args.ops))
    if args.workers < 1 or args.max_length < 1:
        parser.error("workers and max-length must be positive")
    if not 0 <= args.alpha < 1 or not 0 < args.sample_edge_ratio <= 1:
        parser.error("alpha must be in [0, 1); sample-edge-ratio must be in (0, 1]")
    if args.shared_top_k != -1 and args.shared_top_k < 1:
        parser.error("shared-top-k must be positive or -1 (all positions)")
    if args.sample_edge_num != -1 and args.sample_edge_num < 1:
        parser.error("sample-edge-num must be positive or -1 (no cap)")
    return args


@torch.no_grad()
def calculate_curvature(model, inputs, args):
    """Keep one decoder layer on the device; write and release each layer's scores."""
    output = Path(args.output_dir)
    layers = model.model.layers
    end = len(layers) if args.layer_end is None else args.layer_end
    if not 0 <= args.layer_start < end <= len(layers):
        raise ValueError("Expected 0 <= layer-start < layer-end <= model layer count")
    model.model.rotary_emb.to(args.device)
    for layer_id in range(end):
        layer = layers[layer_id].to(args.device)
        layer_cache, sp_cache, scores = {}, {}, {}
        try:
            for sample_idx, hidden in enumerate(inputs):
                positions = torch.arange(hidden.shape[1], device=args.device).unsqueeze(0)
                hidden_out, operations, q_heads, kv_heads, repeat, head_dim = collect_layer_data(
                    layer, hidden.to(args.device), None, positions, model,
                    operation_dtype=curvature_torch_dtype(),
                )
                inputs[sample_idx] = hidden_out.cpu()
                del hidden_out
                if layer_id < args.layer_start:
                    del operations
                    continue
                if not layer_cache:
                    build_layer_cache(model, operations, layer_id, layer_cache,
                                      device=args.compute_device,
                                      required_names=required_cache_names(args.ops, operations))
                for short in args.ops:
                    result = cal_curvature.compute_op_curvature(
                        operations=operations, short_name=short, layer_id=layer_id,
                        sample_idx=sample_idx, layer_cache=layer_cache, sp_cache=sp_cache,
                        alpha=args.alpha, device=args.compute_device, seq_len=hidden.shape[1],
                        num_q_heads=q_heads, num_kv_heads=kv_heads, head_dim=head_dim, repeat=repeat,
                        sample_edge_num=args.sample_edge_num, sample_edge_ratio=args.sample_edge_ratio,
                        dataset_name="prompts", l2_norm=args.l2_norm,
                        shared_top_k=args.shared_top_k, shared_seq_select=args.shared_seq_select,
                        analysis_dir=str(output / "analysis"),
                    )
                    curvature = result["curvature"] if isinstance(result, dict) else result
                    if short in scores:
                        torch.minimum(scores[short], curvature, out=scores[short])
                    else:
                        scores[short] = curvature
                    del result, curvature
                del operations
            if scores:
                path = output / f"layer_{layer_id:03d}.pt"
                torch.save({"layer_id": layer_id, "curvature_scores": scores}, path)
                print(f"Saved {path}", flush=True)
        finally:
            # The copied solver retains process-global views until explicitly reset.
            cal_curvature._reset_shared_state()
            layer_cache.clear()
            sp_cache.clear()
            scores.clear()
            layer.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    args = parse_args()
    torch.set_num_threads(1)
    set_seed(args.seed)
    set_curvature_dtype(args.curvature_dtype)
    cal_curvature.proc = args.workers
    prompts = json.loads(Path(args.prompts_file).read_text()) if args.prompts_file else [args.prompt]
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Calibration prompts must be a nonempty JSON array of nonempty strings")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty to avoid mixing runs: {output}")
    options = dict(cache_dir=args.cache_dir, local_files_only=args.local_files_only,
                   trust_remote_code=args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path, **options)
    dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True, **options,
    ).eval()
    model.config.use_cache = False
    args.layer_end = len(model.model.layers) if args.layer_end is None else args.layer_end
    inputs = capture_layer_inputs(model, tokenizer, prompts, args.max_length)
    settings = vars(args) | {"prompt_count": len(prompts),
                             "prompts": prompts, "edge_sampling_seed": 13,
                             "resolved_model_dtype": str(model.dtype),
                             "prompt_tokens": [x.shape[1] for x in inputs]}
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (output / "graph.json").write_text(json.dumps(GRAPH, indent=2) + "\n")
    calculate_curvature(model, inputs, args)


if __name__ == "__main__":
    main()
