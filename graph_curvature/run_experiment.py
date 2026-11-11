#!/usr/bin/env python3
"""Load a model and LongBench prompts, collect layer values, and calculate A curvature."""

import argparse
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from lm_eval.tasks import TaskManager
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

from graph_curvature.curvature_utils.cal_curvature_A import CURVATURE_ALGORITHM_VERSION, compute_a_curvature
from graph_curvature.curvature_utils.curv_analysis_utils import (
    _summarize_curvatures, save_a_curvature, write_curvature_summary,
)
from graph_curvature.curvature_utils.curv_distribution_utils import _NEIGHBOR_KEEP_QUANTILE
from graph_curvature.curvature_utils.curv_dtype_utils import (
    curvature_torch_dtype, set_curvature_dtype,
)
from graph_curvature.curvature_utils.layerwrapper_curv import collect_layer_data
from graph_curvature.model_utils import capture_layer_inputs, build_a_layer_cache
from kv_cache_experiment.comparison_utils import DEFAULT_METHODS, DEFAULT_RATIOS, configure_determinism
from src.utils.cuda_memory_utils import release_cuda_memory


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
    parser.add_argument("--evaluate", action="store_true", help="Compare saved curvature with kvpress after calculation")
    parser.add_argument("--presses", default=DEFAULT_METHODS)
    parser.add_argument("--compression-ratios", default=DEFAULT_RATIOS)
    parser.add_argument("--max-new-tokens", type=int, help="Evaluation generation cap; defaults to the task setting")
    parser.add_argument("--online-compression", action="store_true", help="Calculate and compress during prefill, then decode and score")
    parser.add_argument("--save-curvature", type=int, choices=(0, 1), default=1)
    parser.add_argument("--score-source", choices=("native", "curvature"), default="native")
    parser.add_argument("--curvature-reduction", choices=("min", "mean"), default="mean")
    parser.add_argument("--evaluation-output-dir")

    args = parser.parse_args()

    # Reject settings that would silently skip work or make curvature undefined.
    for name in ("num_samples", "workers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.alpha < 1:
        parser.error("--alpha must be in [0, 1)")
    if args.max_length is not None and args.max_length < 1:
        parser.error("--max-length must be positive")
    if args.evaluate and not args.online_compression and not args.save_curvature:
        parser.error("Offline evaluation requires --save-curvature 1; use --online-compression otherwise")

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
    layer_cache = None
    layer_dir = Path(args.output_dir) / f"layer_{layer_id:03d}"
    if calculate and args.save_curvature:
        layer_dir.mkdir(parents=True, exist_ok=True)
    prompt_records = []
    record_ids = []
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

        path = layer_dir / f"prompt_{sample_idx:05d}.pt"
        if args.save_curvature and path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if saved['token_count'] != hidden.shape[1]:
                raise ValueError(f"Saved curvature prompt length changed: {path}")
            prompt_records.append(saved["metrics"])
            record_ids.append(sample_idx)
            print(f"Reusing {path}", flush=True)
            del saved, operations
            continue
        if layer_cache is None:
            layer_cache = build_a_layer_cache(model, layer_id, args.compute_device)

        curvature, metrics = compute_a_curvature(
            operations, layer_cache, alpha=args.alpha, device=args.compute_device,
            workers=args.workers,
        )
        metrics['curvature'] = _summarize_curvatures(curvature)
        prompt_records.append(metrics)
        record_ids.append(sample_idx)
        if args.save_curvature:
            save_a_curvature(path, layer_id, sample_idx, operations, curvature, metrics)
            print(f"Saved {path}", flush=True)
        del curvature, operations

    if prompt_records:
        log_path = Path(args.output_dir) / "analysis" / f"layer_{layer_id:03d}_A_summary.txt"
        write_curvature_summary(log_path, [prompt_tokens[i] for i in record_ids],
                                prompt_records, prompt_ids=record_ids)


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


def calculate_or_reuse(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # CPU curvature uses multiple processes, each with one Torch thread.
    torch.set_num_threads(1)
    configure_determinism()
    set_seed(args.seed)
    set_curvature_dtype(args.curvature_dtype)

    config = AutoConfig.from_pretrained(args.model_path, cache_dir=args.cache_dir,
                                       local_files_only=args.local_files_only,
                                       trust_remote_code=args.trust_remote_code)
    args.layer_end = config.num_hidden_layers if args.layer_end is None else args.layer_end
    if not 0 <= args.layer_start < args.layer_end <= config.num_hidden_layers:
        raise ValueError("Expected 0 <= layer-start < layer-end <= model layer count")

    settings_path = output / "settings.json"
    settings = vars(args) | {"neighbor_keep_quantile": _NEIGHBOR_KEEP_QUANTILE,
                             "curvature_algorithm_version": CURVATURE_ALGORITHM_VERSION}
    if settings_path.exists():
        previous = json.loads(settings_path.read_text())
        for key in ("model_path", "tokenizer_path", "task", "num_samples", "max_length", "dtype",
                    "curvature_dtype", "alpha", "seed", "layer_start", "layer_end", "neighbor_keep_quantile",
                    "curvature_algorithm_version"):
            if previous.get(key) != settings[key]:
                raise ValueError(f"Cannot reuse curvature with changed {key}: {previous.get(key)} -> {settings[key]}. "
                                 "Use a new output directory, or evaluate the existing scores with "
                                 "kv_cache_experiment/evaluate_curvature.py --curvature-dir RUN_DIR.")
    else:
        if any(output.glob("layer_*/prompt_*.pt")):
            raise ValueError("Saved curvature has no settings.json to verify reuse")

    complete = args.save_curvature and settings_path.exists() and all(
        (output / f"layer_{layer:03d}" / f"prompt_{prompt:05d}.pt").is_file()
        for layer in range(args.layer_start, args.layer_end) for prompt in range(previous["prompt_count"])
    )
    if complete:
        print(f"All curvature files already exist in {output}; skipping calculation.", flush=True)
        return output

    # Load weights only when at least one curvature file is missing.
    model, tokenizer = load_model(args)
    prompts = load_dataset(args)

    # 2. Capture initial hidden states; later layers collect their values on demand.
    token_ids = []
    inputs = capture_layer_inputs(model, tokenizer, prompts, args.max_length, prompt_token_ids=token_ids)
    manifest = {"input_ids": token_ids,
                "prompt_sha256": [hashlib.sha256(text.encode()).hexdigest() for text in prompts]}
    manifest_path = output / "prompts.pt"
    if manifest_path.exists():
        saved_manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
        if saved_manifest["prompt_sha256"] != manifest["prompt_sha256"] or any(
            not torch.equal(old, new) for old, new in zip(saved_manifest["input_ids"], token_ids)
        ):
            raise ValueError("Prompt content or tokenization changed since the saved curvature run")
        del saved_manifest
    elif args.save_curvature:
        torch.save(manifest, manifest_path)
    del manifest, token_ids
    del prompts, tokenizer

    settings = vars(args) | {
        "prompt_count": len(inputs),
        "resolved_model_dtype": str(model.dtype),
        "prompt_tokens": [hidden.shape[1] for hidden in inputs],
        "total_token_length": sum(hidden.shape[1] for hidden in inputs),
        "neighbor_keep_quantile": _NEIGHBOR_KEEP_QUANTILE,
        "curvature_algorithm_version": CURVATURE_ALGORITHM_VERSION,
        "attention_edge_definition": "head_feature_sets_with_shortest_boundary_distances",
        "curvature_ranking_order": "descending",
    }
    if not settings_path.exists():
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    # 3. Collect each layer's values, calculate curvature, and save the scores.
    calculate_curvature(model, inputs, args)
    del model, inputs
    release_cuda_memory()
    return output


def main():
    args = parse_args()
    if args.online_compression:
        from kv_cache_experiment.evaluate_curvature import evaluate_run
        evaluate_run(args.output_dir, methods=args.presses, compression_ratios=args.compression_ratios,
                     device=args.device, max_new_tokens=args.max_new_tokens,
                     output_dir=args.evaluation_output_dir, online_settings=vars(args),
                     score_source=args.score_source, curvature_reduction=args.curvature_reduction)
        return
    output = calculate_or_reuse(args)
    if args.evaluate:
        from kv_cache_experiment.evaluate_curvature import evaluate_run
        evaluate_run(output, methods=args.presses, compression_ratios=args.compression_ratios,
                     device=args.device, max_new_tokens=args.max_new_tokens,
                     output_dir=args.evaluation_output_dir, score_source=args.score_source,
                     curvature_reduction=args.curvature_reduction)


if __name__ == "__main__":
    main()
