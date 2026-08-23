#!/usr/bin/env python3
"""IBM Granite inspect-evals entrypoint.

This file is intentionally small: build model/task settings here, then reuse
inspect_evals/eval.py for task loading and execution.
"""

import argparse
import importlib.util
import os
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
EVAL_FILE = REPO_DIR / "inspect_evals" / "eval.py"
DEFAULT_IBM_MODEL = "ibm-granite/granite-3.3-2b-instruct"
DEFAULT_CACHE_DIR = REPO_DIR / "llm_weights"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("local_inspect_eval", EVAL_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load eval module from {EVAL_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected bool, got {value}")


def build_model_args(args):
    model_args = {
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path or args.model_path,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.backend == "hf":
        model_args["cache_dir"] = args.cache_dir
    elif args.backend == "vllm":
        model_args["download_dir"] = args.cache_dir
        model_args["gpu_memory_utilization"] = args.gpu_memory_utilization
        if args.batch_size is not None:
            model_args["max_num_seqs"] = args.batch_size
    if args.batch_size is not None:
        model_args["batch_size"] = args.batch_size
    if args.local_files_only and args.backend == "hf":
        model_args["local_files_only"] = True
    return model_args


def build_model_name(args):
    return f"{args.backend}/{args.model_name}"


def main():
    parser = argparse.ArgumentParser(description="Run IBM Granite evals with inspect_evals")
    parser.add_argument("--model-path", default=DEFAULT_IBM_MODEL)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--model-name", default="ibm-granite-local")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--trust-remote-code", type=parse_bool, default=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--think-mode", action="store_true")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("INSPECT_NO_SPAN", "1")
    os.environ.setdefault("XDG_DATA_HOME", "/tmp/inspect_ai_data")
    os.environ.setdefault("HF_HUB_CACHE", args.cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    eval_module = load_eval_module()
    tasks = [eval_module.load_task(task_name) for task_name in args.tasks]

    eval_module.main(
        [build_model_name(args)],
        tasks,
        debug=args.debug,
        model_args=build_model_args(args),
        limit=args.limit,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        system_message="Think step by step before answering." if args.think_mode else None,
        reasoning_effort="high" if args.think_mode else None,
    )


if __name__ == "__main__":
    main()
