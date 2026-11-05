#!/usr/bin/env python3
"""Load a model and LongBench prompts, collect layer values, and calculate curvature."""

import argparse
from itertools import islice
import json
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch
from lm_eval.tasks import TaskManager
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

    # Model and calibration data.
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--cache-dir", default=str(REPO_DIR / "llm_weights"))
    parser.add_argument("--task", default="longbench_lcc")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")

    # Curvature settings.
    parser.add_argument("--compute-device", default="cpu")
    parser.add_argument("--curvature-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--ops", nargs="+", choices=("gate_proj", "up_proj"),
                        default=["A"])
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, help="Exclusive layer index; defaults to all layers")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sample-edge-num", type=int, default=100)
    parser.add_argument("--sample-edge-ratio", type=float, default=1.0)
    parser.add_argument("--shared-top-k", type=int, default=10)
    parser.add_argument("--shared-seq-select", choices=("top", "median", "last"), default="top")
    parser.add_argument("--l2-norm", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    args.ops = list(dict.fromkeys(args.ops))

    # Reject settings that would silently skip work or make curvature undefined.
    for name in ("num_samples", "max_length", "workers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.alpha < 1:
        parser.error("--alpha must be in [0, 1)")
    if not 0 < args.sample_edge_ratio <= 1:
        parser.error("--sample-edge-ratio must be in (0, 1]")
    for name in ("sample_edge_num", "shared_top_k"):
        value = getattr(args, name)
        if value != -1 and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive or -1")

    return args


def load_model(args):
    """Keep model weights on CPU; the pipeline moves one layer at a time."""
    options = dict(
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path, **options)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=args.dtype, low_cpu_mem_usage=True, **options,
    ).eval()
    model.config.use_cache = False

    return model, tokenizer


def load_dataset(args):
    """Take the first N test prompts using the installed lm-eval task template.

    LCC has the shortest average inputs in the published LongBench statistics.
    Token truncation happens later, when capturing first-layer inputs.
    """
    task = TaskManager().load([args.task])["tasks"][args.task]
    return [task.doc_to_text(doc) for doc in islice(task.test_docs(), args.num_samples)]


@torch.no_grad()
def calculate_layer_curvature(model, layer_id, inputs, args):
    """Update inputs to the next hidden states and return per-operation scores.

    Scores have shape [out_features, in_features] and take the minimum across
    samples. Graph costs are built once for this layer and reused.
    """
    layer = model.model.layers[layer_id]
    layer_cache, sp_cache, scores = {}, {}, {}
    analysis_dir = str(Path(args.output_dir) / "analysis")

    try:
        for sample_idx, hidden in enumerate(inputs):
            positions = torch.arange(hidden.shape[1], device=args.device).unsqueeze(0)
            hidden_out, operations, q_heads, kv_heads, repeat, head_dim = collect_layer_data(
                layer, hidden.to(args.device), None, positions, model,
                operation_dtype=curvature_torch_dtype(),
            )
            inputs[sample_idx] = hidden_out.cpu()
            del hidden_out

            # Earlier layers only propagate hidden states to the requested range.
            if layer_id < args.layer_start:
                del operations
                continue

            if not layer_cache:
                build_layer_cache(
                    model, operations, layer_id, layer_cache,
                    device=args.compute_device,
                    required_names=required_cache_names(args.ops, operations),
                )

            for name in args.ops:
                curvature = cal_curvature.compute_op_curvature(
                    operations=operations, short_name=name, layer_id=layer_id,
                    sample_idx=sample_idx, layer_cache=layer_cache, sp_cache=sp_cache,
                    alpha=args.alpha, device=args.compute_device, seq_len=hidden.shape[1],
                    num_q_heads=q_heads, num_kv_heads=kv_heads, head_dim=head_dim, repeat=repeat,
                    sample_edge_num=args.sample_edge_num, sample_edge_ratio=args.sample_edge_ratio,
                    dataset_name=args.task, l2_norm=args.l2_norm,
                    shared_top_k=args.shared_top_k, shared_seq_select=args.shared_seq_select,
                    analysis_dir=analysis_dir,
                )
                if name in scores:
                    torch.minimum(scores[name], curvature, out=scores[name])
                else:
                    scores[name] = curvature
                del curvature

            del operations

        return scores
    finally:
        # The solver holds process-global tensor views; release them after each layer.
        cal_curvature._reset_shared_state()


def calculate_curvature(model, inputs, args):
    """Save each layer's scores immediately, keeping only one layer on the GPU."""
    output = Path(args.output_dir)
    model.model.rotary_emb.to(args.device)

    for layer_id in range(args.layer_end):
        layer = model.model.layers[layer_id].to(args.device)
        try:
            scores = calculate_layer_curvature(model, layer_id, inputs, args)
            if scores:
                path = output / f"layer_{layer_id:03d}.pt"
                torch.save({"layer_id": layer_id, "curvature_scores": scores}, path)
                print(f"Saved {path}", flush=True)
            del scores
        finally:
            layer.to("cpu")
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory to keep runs separate: {output}")

    torch.set_num_threads(1)
    set_seed(args.seed)
    set_curvature_dtype(args.curvature_dtype)
    cal_curvature.proc = args.workers

    # 1. Load model and dataset.
    model, tokenizer = load_model(args)
    prompts = load_dataset(args)
    args.layer_end = len(model.model.layers) if args.layer_end is None else args.layer_end
    if not 0 <= args.layer_start < args.layer_end <= len(model.model.layers):
        raise ValueError("Expected 0 <= layer-start < layer-end <= model layer count")

    # 2. Capture initial hidden states; later layers collect their values on demand.
    inputs = capture_layer_inputs(model, tokenizer, prompts, args.max_length)
    del prompts, tokenizer

    settings = vars(args) | {
        "prompt_count": len(inputs),
        "edge_sampling_seed": args.seed,
        "resolved_model_dtype": str(model.dtype),
        "prompt_tokens": [hidden.shape[1] for hidden in inputs],
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (output / "graph.json").write_text(json.dumps(GRAPH, indent=2) + "\n")

    # 3. Collect each layer's values, calculate curvature, and save the scores.
    calculate_curvature(model, inputs, args)


if __name__ == "__main__":
    main()
