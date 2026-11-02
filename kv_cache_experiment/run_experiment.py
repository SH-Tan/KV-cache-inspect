#!/usr/bin/env python3
"""lm-eval LongBench compression-rate sweep. Experiment settings live in run.sh."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import sys

# Set before importing torch or creating a CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import kvpress
import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from src.utils.cuda_memory_utils import release_cuda_memory
from output_logging import (
    GenerationStats, record_compressed_prompt, write_json, write_prompt_scores, write_score_table,
)


def configure_determinism():
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "tokenizer-path", "cache-dir", "device-map", "dtype",
                 "attn-implementation", "press", "output-dir", "tasks"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("limit", "max-context-length", "seed", "top-k"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--compression-ratios", "--compression-ratio", required=True)
    parser.add_argument("--press-kwargs", type=json.loads, required=True)
    parser.add_argument("--task-counts", type=json.loads, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    args.tasks = [name.strip() for name in args.tasks.split(",")]
    try:
        args.compression_ratios = [float(value) for value in args.compression_ratios.split(",")]
    except ValueError:
        parser.error("--compression-ratios must be comma-separated numbers")
    for ratio in args.compression_ratios:
        if not 0 <= ratio < 1:
            parser.error("compression ratios must be in [0, 1)")
    args.compression_ratios = sorted(set([0.0] + args.compression_ratios))
    if args.limit < 0:
        parser.error("limit must be nonnegative (0 disables detailed logs)")
    if any(args.task_counts[name] < 1 for name in args.tasks):
        parser.error("task counts must be positive")
    return args


class KVPressLM(HFLM):
    """Use lm-eval generation/scoring and capture compact next-token probabilities."""

    def __init__(self, args, model, tokenizer):
        super().__init__(pretrained=model, tokenizer=tokenizer, batch_size=1,
                         max_length=args.max_context_length, logits_cache=False)
        self.args = args
        self.press = None
        self.case = "baseline"
        self.compression_ratio = 0.0

    def generate_until(self, requests, disable_tqdm=False):
        answers = []
        for request in tqdm(requests, desc=self.case, disable=disable_tqdm):
            # lm-eval numbers requests within the selected subset.
            doc_id = self.sample_ids[request.doc_id]
            self.record = None
            if doc_id in self.log_ids:
                prompt_dir = Path(self.args.output_dir) / request.task_name / f"prompt_{doc_id}"
                prompt_dir.mkdir(parents=True, exist_ok=True)
                self.record = {
                    "do_sample": self.args.do_sample,
                    "temperature": self.args.temperature, "top_p": self.args.top_p,
                    "top_k": self.args.top_k, "seed": self.args.seed,
                    "model": self.args.model_path, "press": self.args.press,
                    "task": request.task_name, "doc_id": doc_id, "case": self.case,
                    "compression_ratio": self.compression_ratio, "compression_scope": "prefill_only",
                    "reference_answers": request.doc["answers"],
                    "probability_definition": "softmax(raw model logits), before temperature/top-p/top-k",
                }
            answer = super().generate_until([request], disable_tqdm=True)[0]
            if self.record is not None:
                write_json(prompt_dir / f"{self.case}.json", self.record)
            answers.append(answer)
            release_cuda_memory()
        return answers

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        set_seed(self.args.seed)
        prompt_length = context.shape[1]
        stats = GenerationStats() if self.record is not None else None
        compression = self.press(self.model) if self.press is not None else nullcontext()
        prompt_logging = (
            record_compressed_prompt(self.press, context, self.tokenizer, self.record)
            if self.press is not None and self.record is not None else nullcontext()
        )
        with torch.no_grad(), compression, prompt_logging:
            # Capture only generation forwards, after the press has initialized.
            hook = (
                self.model.register_forward_hook(stats.record)
                if stats is not None else nullcontext()
            )
            with hook:
                generated = super()._model_generate(
                    context, max_length, stop, output_scores=False, output_logits=False,
                    return_dict_in_generate=False, **generation_kwargs,
                )
        if self.record is not None:
            generated_ids = generated[0, prompt_length:].tolist()
            self.record["prompt_tokens"] = prompt_length
            self.record["generated_tokens"] = len(generated_ids)
            self.record["kv_cache_tokens"] = {
                "after_prefill_per_layer": stats.prefill_cache_tokens,
                "after_generation_per_layer": stats.final_cache_tokens,
            }
            self.record["checkpoints"] = stats.checkpoints(generated_ids, self.tokenizer)
        return generated


def main():
    args = parse_args()
    configure_determinism()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "settings.json", vars(args))
    press_class = getattr(kvpress, args.press)
    if press_class.forward_hook is not kvpress.BasePress.forward_hook:
        raise ValueError("This experiment requires the BasePress prefill-only hook, e.g. SnapKVPress/KnormPress")
    load_kwargs = dict(cache_dir=args.cache_dir, local_files_only=args.local_files_only,
                       trust_remote_code=args.trust_remote_code)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=args.dtype, device_map=args.device_map,
        attn_implementation=args.attn_implementation, low_cpu_mem_usage=True, **load_kwargs,
    ).eval()
    
    if args.max_context_length > model.config.max_position_embeddings:
        raise ValueError("MAX_CONTEXT_LENGTH exceeds model max_position_embeddings")
    
    lm = KVPressLM(args, model, tokenizer)
    manager = TaskManager()
    generation_kwargs = dict(do_sample=args.do_sample, temperature=args.temperature,
                             top_p=args.top_p, top_k=args.top_k, num_beams=1,
                             num_return_sequences=1)
    scores = {}
    for task_name in args.tasks:
        task = manager.load([task_name])["tasks"][task_name]
        # Dedicated RNG: the same subset is used in baseline and compressed runs.
        num_docs = len(task.test_docs())
        indices = sorted(random.Random(args.seed).sample(
            range(num_docs), min(args.task_counts[task_name], num_docs),
        ))
        lm.sample_ids = indices
        lm.log_ids = set(random.Random(args.seed).sample(indices, min(args.limit, len(indices))))
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        write_json(task_dir / "sample_ids.json", indices)
        write_json(task_dir / "log_ids.json", sorted(lm.log_ids))
        
        for ratio in args.compression_ratios:
            case = "baseline"
            lm.press = None
            if ratio > 0:
                case = f"compression_{ratio}"
                lm.press = press_class(compression_ratio=ratio, **args.press_kwargs)
            lm.case = case
            lm.compression_ratio = ratio
            
            results = simple_evaluate(
                model=lm, tasks=[task], task_manager=manager, num_fewshot=0, batch_size=1,
                samples={task_name: indices}, gen_kwargs=generation_kwargs,
                apply_chat_template=False, bootstrap_iters=0, log_samples=False,
                random_seed=args.seed, numpy_random_seed=args.seed,
                torch_random_seed=args.seed, fewshot_random_seed=args.seed,
            )
            write_prompt_scores(task, indices, lm.log_ids, task_dir, case)
            scores[(ratio, task_name)] = results["results"][task_name]["score,none"]
            write_score_table(
                output_dir / "scores.csv", args.tasks, args.compression_ratios, scores,
                model_path=args.model_path, press=args.press, do_sample=args.do_sample, seed=args.seed,
            )
            write_json(task_dir / f"{case}_results.json", results)
            print(f"{task_name} {case}: {results['results'][task_name]}")
            del results
            release_cuda_memory()
        del task


if __name__ == "__main__":
    main()
