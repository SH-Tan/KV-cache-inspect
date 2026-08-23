#!/usr/bin/env python3
"""Run evaluation sweep across models and tasks."""

import argparse
from importlib import import_module
from pathlib import Path
from posixpath import basename
import sys

import inspect_ai
from inspect_ai import Task
from inspect_ai.log import list_eval_logs, read_eval_log

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from token_length_log import append_token_length_summary

ALL_TASKS = {
    "pubmedqa": "inspect_evals.pubmedqa",
    "medqa": "inspect_evals.medqa",
    "aime2024": "inspect_evals.aime2024",
    "gsm8k": "inspect_evals.gsm8k",
    "math500": "inspect_evals.math500",
    "tab_fact": "inspect_evals.tab_fact",
    "legalbench": "inspect_evals.legalbench",
    "finben": "inspect_evals.finben",
    "livecodebench": "inspect_evals.livecodebench",
    "codeforces": "inspect_evals.codeforces",
    "polyglot": "inspect_evals.polyglot",
    "amc23": "inspect_evals.amc23",
    "humaneval": "inspect_evals.humaneval",
    "bigcodebench": "inspect_evals.bigcodebench",
    "mbpp": "inspect_evals.mbpp",
    "usaco": "inspect_evals.usaco",
}


def load_task(name: str) -> Task:
    if name == "livecodebench":
        livecodebench_dir = Path(__file__).parent / "src" / "inspect_evals" / "livecodebench" / "LiveCodeBench"
        sys.path.insert(0, str(livecodebench_dir))
    return getattr(import_module(ALL_TASKS[name]), name)


def parse_model_args(items: list[str]) -> dict[str, object]:
    model_args: dict[str, object] = {}
    for item in items:
        key, value = item.split("=", 1)
        if value in {"true", "false"}:
            model_args[key] = value == "true"
        else:
            try:
                model_args[key] = int(value)
            except ValueError:
                try:
                    model_args[key] = float(value)
                except ValueError:
                    model_args[key] = value
    return model_args


def get_run_name(model: str, task: Task, debug: bool = False) -> str:
    return f"{basename(model)}__{task.__name__}{'__debug' if debug else ''}"


def run_eval(
    model: str,
    task: Task,
    debug: bool,
    model_args: dict[str, object],
    limit: int | None,
    max_tokens: int | None,
    batch_size: int | None,
    system_message: str | None,
    reasoning_effort: str | None,
) -> None:
    run_name = get_run_name(model, task, debug)
    log_dir = f"logs/{run_name}"
    max_tokens = max_tokens or (2048 if "Qwen2.5-Math-7B" in model else 1024 * 16)
    print(f"running eval: {run_name}")
    inspect_ai.eval(
        task,
        model=model,
        model_args=model_args,
        log_dir=log_dir,
        max_tokens=max_tokens,
        max_connections=batch_size,
        system_message=system_message,
        reasoning_effort=reasoning_effort,
        limit=limit,
        display="none",
    )
    logs_infos = list_eval_logs(log_dir)
    latest = max(logs_infos, key=lambda x: x.mtime)
    log = read_eval_log(latest)
    if log.results and log.results.scores:
        metrics = log.results.scores[0].metrics
        if "accuracy" in metrics:
            print(f"  accuracy: {metrics['accuracy'].value}")
    append_token_length_summary(log)


def main(
    models: list[str],
    tasks: list[Task],
    debug: bool = False,
    model_args: dict[str, object] | None = None,
    limit: int | None = None,
    max_tokens: int | None = None,
    batch_size: int | None = None,
    system_message: str | None = None,
    reasoning_effort: str | None = None,
) -> None:
    model_args = model_args or {}
    for model in models:
        for task in tasks:
            run_eval(
                model,
                task,
                debug,
                model_args,
                limit,
                max_tokens,
                batch_size,
                system_message,
                reasoning_effort,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation sweep across models and tasks")
    parser.add_argument("--models", nargs="+", required=True, help="Models to evaluate")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(ALL_TASKS.keys()),
        choices=list(ALL_TASKS.keys()),
        help="Tasks to evaluate (default: all)",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--model-args", nargs="*", default=[], help="Model args as key=value pairs")
    parser.add_argument("--limit", type=int, default=None, help="Limit eval samples")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max generated tokens")
    parser.add_argument("--batch-size", type=int, default=None, help="Max concurrent samples")
    parser.add_argument("--system-message", type=str, default=None, help="Optional system message")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None)

    args = parser.parse_args()
    tasks = [load_task(t) for t in args.tasks]

    main(
        args.models,
        tasks,
        debug=args.debug,
        model_args=parse_model_args(args.model_args),
        limit=args.limit,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        system_message=args.system_message,
        reasoning_effort=args.reasoning_effort,
    )
