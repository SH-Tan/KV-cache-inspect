#!/usr/bin/env python3
"""Load a model and LongBench prompts, collect layer values, and calculate A curvature."""

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

from graph_curvature.curvature_utils.cal_curvature_A import compute_a_curvature
from graph_curvature.curvature_utils.curv_analysis_utils import (
    _summarize_curvatures, rank_a_curvature, write_curvature_summary,
)
from graph_curvature.curvature_utils.curv_distribution_utils import _NEIGHBOR_KEEP_QUANTILE
from graph_curvature.curvature_utils.curv_dtype_utils import (
    curvature_torch_dtype, set_curvature_dtype,
)
from graph_curvature.curvature_utils.layerwrapper_curv import collect_layer_data
from graph_curvature.model_utils import capture_layer_inputs, build_a_layer_cache


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Model and calibration data.
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--cache-dir", default=str(REPO_DIR / "llm_weights"))
    parser.add_argument("--task", default="longbench_lcc")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-length", type=int,
                        help="Optional prompt token cap; defaults to the complete prompt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")

    # Curvature settings.
    parser.add_argument("--compute-device", default="cpu")
    parser.add_argument("--curvature-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, help="Exclusive layer index; defaults to all layers")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=48,
                        help="CPU curvature processes; default: 48 physical cores on the A40 host")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    # Reject settings that would silently skip work or make curvature undefined.
    for name in ("num_samples", "workers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.alpha < 1:
        parser.error("--alpha must be in [0, 1)")
    if args.max_length is not None and args.max_length < 1:
        parser.error("--max-length must be positive")

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
    """Take the first N test prompts from the selected lm-eval task."""
    task = TaskManager().load([args.task])["tasks"][args.task]
    return [task.doc_to_text(doc) for doc in islice(task.test_docs(), args.num_samples)]


@torch.no_grad()
def calculate_layer_curvature(model, layer_id, inputs, args):
    """Propagate prompts and save each prompt's last-row A scores independently."""
    layer = model.model.layers[layer_id]
    calculate = layer_id >= args.layer_start
    layer_cache = build_a_layer_cache(model, layer_id, args.compute_device) if calculate else None
    layer_dir = Path(args.output_dir) / f"layer_{layer_id:03d}"
    if calculate:
        layer_dir.mkdir(parents=True, exist_ok=True)
    prompt_records = []
    prompt_tokens = [hidden.shape[1] for hidden in inputs]

    for sample_idx, hidden in enumerate(inputs):
        positions = torch.arange(hidden.shape[1], device=args.device).unsqueeze(0)
        hidden_out, operations = collect_layer_data(
            layer, hidden.to(args.device), None, positions, model,
            operation_dtype=curvature_torch_dtype(),
        )
        inputs[sample_idx] = hidden_out.cpu()
        del hidden_out
        if not calculate:
            del operations
            continue

        curvature, metrics = compute_a_curvature(
            operations, layer_cache, alpha=args.alpha, device=args.compute_device,
            workers=args.workers,
        )
        metrics['curvature'] = _summarize_curvatures(curvature)
        prompt_records.append(metrics)
        path = layer_dir / f"prompt_{sample_idx:05d}.pt"
        torch.save({
            "layer_id": layer_id,
            "prompt_id": sample_idx,
            "token_count": hidden.shape[1],
            "query_index": hidden.shape[1] - 1,
            "attention_scores": operations['A'][0, :, 0].contiguous(),
            "curvature_scores": {"A": curvature},
            "curvature_ranking": rank_a_curvature(curvature),
        }, path)
        print(f"Saved {path}", flush=True)
        del curvature, operations

    if prompt_records:
        log_path = Path(args.output_dir) / "analysis" / f"layer_{layer_id:03d}_A_summary.txt"
        write_curvature_summary(log_path, prompt_tokens, prompt_records)


def calculate_curvature(model, inputs, args):
    """Save each layer's scores immediately, keeping only one layer on the GPU."""
    model.model.rotary_emb.to(args.device)

    for layer_id in range(args.layer_end):
        layer = model.model.layers[layer_id].to(args.device)
        try:
            calculate_layer_curvature(model, layer_id, inputs, args)
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

    # CPU curvature uses multiple processes, each with one Torch thread.
    torch.set_num_threads(1)
    set_seed(args.seed)
    set_curvature_dtype(args.curvature_dtype)

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
        "resolved_model_dtype": str(model.dtype),
        "prompt_tokens": [hidden.shape[1] for hidden in inputs],
        "total_token_length": sum(hidden.shape[1] for hidden in inputs),
        "neighbor_keep_quantile": _NEIGHBOR_KEEP_QUANTILE,
        "attention_edge_definition": "head_feature_sets_with_shortest_boundary_distances",
        "curvature_ranking_order": "descending",
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    # 3. Collect each layer's values, calculate curvature, and save the scores.
    calculate_curvature(model, inputs, args)


if __name__ == "__main__":
    main()
