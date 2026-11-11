#!/usr/bin/env python3
"""Compare saved curvature and kvpress on identical prompts and total KV budgets."""

import argparse
from contextlib import nullcontext
from itertools import islice
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from kvpress.utils import compute_n_kept

from kv_cache_experiment.comparison_utils import DEFAULT_METHODS, DEFAULT_RATIOS, configure_determinism
from kv_cache_experiment.curvature_press import CURVATURE_SCORE_METHODS, SavedCurvaturePress, make_press
from kv_cache_experiment.output_logging import write_json
from src.utils.cuda_memory_utils import release_cuda_memory


def load_prompt_tokens(run_dir, settings, task, tokenizer, online=False):
    """Use the first N documents, exactly as the curvature collector does."""
    count = settings["num_samples"] if online else settings["prompt_count"]
    prompts = [task.doc_to_text(doc) for doc in islice(task.test_docs(), count)]
    path = run_dir / "prompts.pt"
    if not online and path.exists():
        manifest = torch.load(path, map_location="cpu", weights_only=True)
        hashes = [hashlib.sha256(text.encode()).hexdigest() for text in prompts]
        if hashes != manifest["prompt_sha256"]:
            raise ValueError("Task prompts differ from those used to calculate curvature")
        tokens, identity = manifest["input_ids"], "saved_input_ids_and_text_hashes"
    else:
        # Legacy runs did not save token IDs. Reconstruct from their recorded settings.
        tokens = [tokenizer(text, return_tensors="pt", truncation=settings["max_length"] is not None,
                            max_length=settings["max_length"])["input_ids"][0] for text in prompts]
        identity = "online_tokenized_input_ids" if online else "legacy_reconstructed_from_task_and_run_settings"
    if not online and [ids.numel() for ids in tokens] != settings["prompt_tokens"]:
        raise ValueError("Tokenized prompts do not match the saved curvature token counts")
    return prompts, tokens, identity


class CurvatureComparisonLM(HFLM):
    """Keep saved tokenization and verify physical cache budgets during generation."""

    def __init__(self, model, tokenizer, prompts, prompt_tokens, seed):
        super().__init__(pretrained=model, tokenizer=tokenizer, batch_size=1,
                         max_length=model.config.max_position_embeddings, logits_cache=False)
        self.prompts, self.prompt_tokens, self.seed = prompts, prompt_tokens, seed
        self.press = None
        self.prompt_id = 0
        self.curvature_provider = None

    def generate_until(self, requests, disable_tqdm=False):
        answers = []
        for request in requests:
            self.prompt_id = request.doc_id
            if request.args[0] != self.prompts[self.prompt_id]:
                raise ValueError("Evaluation changed the prompt text used for curvature")
            scorer = getattr(self.press, "curvature_scorer", self.press)
            if isinstance(scorer, SavedCurvaturePress):
                scorer.prompt_id = self.prompt_id
            if self.curvature_provider is not None:
                self.curvature_provider.prompt_id = self.prompt_id
            answer = super().generate_until([request], disable_tqdm=True)[0]
            answers.append(answer)
        return answers

    def tok_batch_encode(self, strings, padding_side="left", left_truncate_len=None, truncation=False):
        ids = self.prompt_tokens[self.prompt_id].unsqueeze(0)
        if left_truncate_len is not None and ids.shape[1] > left_truncate_len:
            raise ValueError("Prompt plus generation exceeds model context; do not silently truncate saved scores")
        return ids, torch.ones_like(ids)

    def _model_generate(self, context, max_length, stop, **kwargs):
        set_seed(self.seed)
        prompt_length = context.shape[1]
        ratio = self.press.compression_ratio if self.press is not None else 0.0
        expected = compute_n_kept(prompt_length, ratio)
        observed = False

        def record_cache(module, inputs, output):
            nonlocal observed
            if observed:
                return
            layers = output.past_key_values.layers
            # PyramidKV varies layer sizes; total elements must still match exactly.
            actual_elements = sum(layer.keys.numel() + layer.values.numel() for layer in layers)
            expected_elements = sum(
                2 * layer.keys.shape[0] * layer.keys.shape[1] * expected * layer.keys.shape[-1]
                for layer in layers
            )
            if actual_elements != expected_elements:
                raise ValueError(f"Unequal KV budget: expected {expected_elements}, got {actual_elements} elements")
            observed = True

        compression = self.press(self.model) if self.press is not None else nullcontext()
        needs_curvature = isinstance(self.press, SavedCurvaturePress) or hasattr(self.press, "curvature_scorer")
        capture = (self.curvature_provider.capture(prompt_length)
                   if self.curvature_provider is not None and needs_curvature else nullcontext())
        with torch.no_grad(), capture, compression, self.model.register_forward_hook(record_cache):
            generated = super()._model_generate(
                context, max_length, stop, logits_to_keep=1, output_scores=False,
                return_dict_in_generate=False, **kwargs,
            )
        if not observed:
            raise RuntimeError("Generation did not expose a KV cache for budget verification")
        return generated


def evaluate_run(curvature_dir, methods=DEFAULT_METHODS, compression_ratios=DEFAULT_RATIOS,
                 device="cuda:0", max_new_tokens=None, output_dir=None, *, online_settings=None,
                 score_source="native", curvature_reduction="mean"):
    run_dir = Path(curvature_dir)
    online = online_settings is not None
    if online:
        from graph_curvature.curvature_utils.cal_curvature_A import CURVATURE_ALGORITHM_VERSION
        from graph_curvature.curvature_utils.curv_distribution_utils import _NEIGHBOR_KEEP_QUANTILE
        from graph_curvature.curvature_utils.curv_dtype_utils import set_curvature_dtype
        from graph_curvature.online_curvature import OnlineCurvature
        if (run_dir / "settings.json").exists():
            raise ValueError("Online calculation requires a new OUTPUT_DIR; use saved-log evaluation to reuse a run")
        settings = dict(online_settings) | {
            "neighbor_keep_quantile": _NEIGHBOR_KEEP_QUANTILE,
            "curvature_algorithm_version": CURVATURE_ALGORITHM_VERSION,
            "attention_edge_definition": "head_feature_sets_with_shortest_boundary_distances",
            "curvature_ranking_order": "descending",
        }
        set_curvature_dtype(settings["curvature_dtype"])
        torch.set_num_threads(1)
    else:
        settings = json.loads((run_dir / "settings.json").read_text())
    configure_determinism()
    methods = list(dict.fromkeys(method.strip() for method in methods.split(",")))
    ratios = sorted(set(float(value) for value in compression_ratios.split(",")))
    if any(not 0 < ratio < 1 for ratio in ratios):
        raise ValueError("Compression ratios must be in (0, 1); baseline is evaluated separately")
    if settings["layer_start"] != 0:
        raise ValueError("Comparison requires saved curvature for every model layer")
    if not online:
        for layer in range(settings["layer_end"]):
            for prompt in range(settings["prompt_count"]):
                path = run_dir / f"layer_{layer:03d}" / f"prompt_{prompt:05d}.pt"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing curvature file: {path}")
    options = dict(cache_dir=settings["cache_dir"], local_files_only=settings["local_files_only"],
                   trust_remote_code=settings["trust_remote_code"])
    tokenizer = AutoTokenizer.from_pretrained(settings["tokenizer_path"] or settings["model_path"], **options)
    model = AutoModelForCausalLM.from_pretrained(
        settings["model_path"], dtype=settings["dtype"], device_map=device,
        attn_implementation="sdpa", low_cpu_mem_usage=True, **options,
    ).eval()
    if online and settings["layer_end"] is None:
        settings["layer_end"] = model.config.num_hidden_layers
    if settings["layer_end"] != model.config.num_hidden_layers:
        raise ValueError("Comparison requires saved curvature for every model layer")
    manager = TaskManager()
    task = manager.load([settings["task"]])["tasks"][settings["task"]]
    prompts, tokens, identity = load_prompt_tokens(run_dir, settings, task, tokenizer, online=online)
    if online:
        settings.update(prompt_count=len(prompts), prompt_tokens=[ids.numel() for ids in tokens],
                        total_token_length=sum(ids.numel() for ids in tokens), resolved_model_dtype=str(model.dtype))
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "settings.json", settings)
        if settings["save_curvature"]:
            torch.save({"input_ids": tokens,
                        "prompt_sha256": [hashlib.sha256(text.encode()).hexdigest() for text in prompts]},
                       run_dir / "prompts.pt")
    lm = CurvatureComparisonLM(model, tokenizer, prompts, tokens, settings["seed"])
    if online:
        lm.curvature_provider = OnlineCurvature(model, settings, run_dir)
    prompt_ids = list(range(len(prompts)))
    report = {
        "curvature_dir": str(run_dir.resolve()), "model": settings["model_path"],
        "task": settings["task"], "prompt_count": len(prompts),
        "prompt_selection": "first N test documents in dataset order",
        "prompt_identity": identity, "seed": settings["seed"],
        "tokenizer": settings["tokenizer_path"] or settings["model_path"],
        "dtype": settings.get("resolved_model_dtype", settings["dtype"]),
        "max_length": settings["max_length"],
        "curvature_files": "layer_{layer:03d}/prompt_{prompt:05d}.pt",
        "alpha": settings["alpha"],
        "curvature_dtype": settings["curvature_dtype"],
        "layer_start": settings["layer_start"], "layer_end": settings["layer_end"],
        "attention_edge_definition": settings.get("attention_edge_definition"),
        "neighbor_keep_quantile": settings["neighbor_keep_quantile"],
        "curvature_algorithm_version": settings.get("curvature_algorithm_version", 1),
        "window_attention_scaling": "model.scaling for SnapKV and PyramidKV",
        "compression_scope": "prefill_only; generated tokens are retained",
        "curvature_reduction": "min/mean across all query heads; same token scores for each KV head",
        "curvature_removal_order": "negative reduced curvature is importance; variants then apply pooling/window or sink/value-norm policies",
        "budget": "equal total prefill KV elements; PyramidKV rounding matched to uniform budget",
        "generation": {"do_sample": False, "max_gen_toks": max_new_tokens or task.config.generation_kwargs["max_gen_toks"]},
        "accuracy_metric": "LongBench task score (code similarity for LCC)",
        "accuracy_scale": "0-1",
        "online_compression": online,
        "save_curvature": bool(settings.get("save_curvature", True)),
        "score_source": score_source, "curvature_score_reduction": curvature_reduction,
        "curvature_score_methods": [method for method in methods
                                    if score_source == "curvature" and method in CURVATURE_SCORE_METHODS],
        "curvature_query": "last prompt query; curvature variants do not estimate attention for other queries",
        "curvature_attention_replacement": "SnapKV/PyramidKV pool negative curvature; ExpectedAttention applies softmax before value-norm scaling",
    }
    report = {"settings": report, "cases": []}
    output = Path(output_dir) if output_dir is not None else run_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    cases = [("baseline", 0.0)] + [(method, ratio) for ratio in ratios for method in methods]
    for method, ratio in cases:
        lm.press = None if method == "baseline" else make_press(
            method, ratio, run_dir, score_source=score_source, curvature_reduction=curvature_reduction,
            curvature_provider=lm.curvature_provider,
        )
        generation = dict(do_sample=False, temperature=0.0)
        if max_new_tokens is not None:
            generation["max_gen_toks"] = max_new_tokens
        results = simple_evaluate(
            model=lm, tasks=[task], task_manager=manager, num_fewshot=0, batch_size=1,
            samples={settings["task"]: prompt_ids}, gen_kwargs=generation,
            apply_chat_template=False, bootstrap_iters=0, log_samples=False,
            random_seed=settings["seed"], numpy_random_seed=settings["seed"],
            torch_random_seed=settings["seed"], fewshot_random_seed=settings["seed"],
        )
        metrics = results["results"][settings["task"]]
        report["cases"].append({"method": method, "removal_ratio": ratio,
                                "accuracy": metrics["score,none"]})
        write_json(output / "comparison.json", report)
        print(f"{method:20s} removal={ratio:.2f} score={metrics['score,none'] * 100:.4f}", flush=True)
        release_cuda_memory()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curvature-dir", type=Path, required=True)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--compression-ratios", default=DEFAULT_RATIOS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--score-source", choices=("native", "curvature"), default="native")
    parser.add_argument("--curvature-reduction", choices=("min", "mean"), default="mean")
    args = parser.parse_args()
    evaluate_run(**vars(args))


if __name__ == "__main__":
    main()
