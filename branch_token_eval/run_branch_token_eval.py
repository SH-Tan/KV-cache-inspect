#!/usr/bin/env python3
"""Branch generation from top next-token alternatives during GSM8K eval."""

import argparse
import math
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

from inspect_ai.dataset import hf_dataset
from vllm import LLM, SamplingParams


REPO_DIR = Path(__file__).resolve().parents[1]
INSPECT_EVALS_SRC = REPO_DIR / "inspect_evals" / "src"
sys.path.insert(0, str(INSPECT_EVALS_SRC))

from inspect_evals.gsm8k.gsm8k import (  # noqa: E402
    MATH_PROMPT_TEMPLATE,
    record_to_sample,
    sample_to_fewshot,
    strip_chinese,
)


DEFAULT_MODEL = "ibm-granite/granite-3.3-2b-instruct"
DEFAULT_CACHE_DIR = REPO_DIR / "llm_weights"
PERCENTS = [0, 20, 50, 75, 90]


def parse_args():
    parser = argparse.ArgumentParser(description="Run GSM8K top-token branch eval with vLLM")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", default="logs/branch_token_eval")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-logprobs", type=int, default=10)
    parser.add_argument("--branch-top-k", type=int, default=5)
    parser.add_argument("--fewshot", type=int, default=10)
    parser.add_argument("--fewshot-seed", type=int, default=42)
    parser.add_argument("--think-mode", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def build_prompt(sample, fewshot_samples, tokenizer, think_mode: bool) -> str:
    messages = []
    system_parts = []
    if think_mode:
        system_parts.append("Think step by step before answering.")
    if fewshot_samples:
        system_parts.append("\n\n".join(sample_to_fewshot(s) for s in fewshot_samples))
    if system_parts:
        messages.append(
            {
                "role": "system",
                "content": "\n\n".join(system_parts),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": MATH_PROMPT_TEMPLATE.format(prompt=sample.input),
        }
    )
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def load_gsm8k(limit: int, start: int, fewshot: int, fewshot_seed: int):
    train = []
    if fewshot:
        train = list(
            hf_dataset(
                path="gsm8k",
                data_dir="main",
                split="train",
                sample_fields=record_to_sample,
                auto_id=True,
                shuffle=True,
                seed=fewshot_seed,
                limit=fewshot,
            )
        )
    test = hf_dataset(
        path="gsm8k",
        data_dir="main",
        split="test",
        sample_fields=record_to_sample,
        auto_id=True,
    )
    return list(test)[start : start + limit], train


def token_logprobs(position_logprobs, top_n: int):
    if not position_logprobs:
        return []
    rows = []
    for token_id, item in sorted(
        position_logprobs.items(), key=lambda kv: kv[1].logprob, reverse=True
    )[:top_n]:
        rows.append(
            {
                "token_id": token_id,
                "token": item.decoded_token,
                "logprob": item.logprob,
                "probability": math.exp(item.logprob),
                "rank": item.rank,
            }
        )
    return rows


def selected_positions(token_count: int):
    if token_count <= 0:
        return []
    seen = set()
    positions = []
    for percent in PERCENTS:
        index = int((token_count - 1) * percent / 100)
        if index not in seen:
            positions.append((percent, index))
            seen.add(index)
    return positions


def extract_answer(text: str) -> str:
    clean = strip_chinese(text)
    answer_match = re.search(r"ANSWER\s*:\s*([^\n]+)", clean, flags=re.IGNORECASE)
    if answer_match:
        clean = answer_match.group(1)
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", clean)
    return numbers[-1].replace(",", "") if numbers else clean.strip().rstrip(".")


def is_correct_answer(pred: str, target: str) -> bool:
    if pred == target:
        return True
    try:
        return Decimal(pred) == Decimal(target.replace(",", ""))
    except (InvalidOperation, ValueError):
        return False


def _write_top10(f, top10_tokens):
    f.write("top10_tokens:\n")
    for token in top10_tokens:
        f.write(
            "  "
            f"rank={token['rank']} "
            f"token_id={token['token_id']} "
            f"probability={token['probability']} "
            f"logprob={token['logprob']} "
            f"token={token['token']!r}\n"
        )
    f.write("\n")


def write_example_header(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("QUESTION\n\n")
        f.write(f"sample_index: {row['sample_index']}\n")
        f.write(f"sample_id: {row['sample_id']}\n")
        f.write(f"target: {row['target']}\n\n")
        f.write(f"{row['question']}\n\n")

        f.write("BASELINE\n\n")
        f.write(f"generated_tokens: {row['generated_tokens']}\n")
        f.write(f"final_answer: {row['final_answer']}\n")
        f.write(f"is_correct: {row['is_correct']}\n\n")
        f.write("full_response:\n\n")
        f.write(f"{row['full_response']}\n\n")


def append_position_log(path: Path, percent: int, token_index: int, top10_tokens):
    with path.open("a", encoding="utf-8") as f:
        f.write(f"POSITION percent={percent} token_index={token_index}\n\n")
        _write_top10(f, top10_tokens)


def append_branch_log(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(f"BRANCH choice_rank={row['choice_rank']}\n\n")
        f.write(f"choice_token_id: {row['choice_token_id']}\n")
        f.write(f"choice_token: {row['choice_token']!r}\n")
        f.write(f"choice_probability: {row['choice_probability']}\n")
        f.write(f"choice_logprob: {row['choice_logprob']}\n")
        f.write(f"final_generated_tokens: {row['final_generated_tokens']}\n")
        f.write(f"final_answer: {row['final_answer']}\n")
        f.write(f"is_correct: {row['is_correct']}\n\n")
        f.write("full_response:\n\n")
        f.write(f"{row['full_response']}\n\n")
        f.write("\n\n")


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_CACHE", args.cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", args.cache_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, fewshot_samples = load_gsm8k(
        args.limit, args.start, args.fewshot, args.fewshot_seed
    )
    llm = LLM(
        model=args.model_path,
        tokenizer=args.tokenizer_path or args.model_path,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        download_dir=args.cache_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,
        max_num_seqs=1,
    )
    tokenizer = llm.get_tokenizer()
    base_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        logprobs=args.top_logprobs,
    )

    for sample_index, sample in enumerate(samples, start=args.start):
        example_log_path = output_dir / "examples" / f"sample_{sample_index}.txt"
        prompt = build_prompt(sample, fewshot_samples, tokenizer, args.think_mode)
        base_output = llm.generate([prompt], base_params, use_tqdm=False)[0].outputs[0]
        base_token_ids = list(base_output.token_ids)
        base_text = base_output.text
        base_answer = extract_answer(base_text)
        base_correct = is_correct_answer(base_answer, sample.target)

        baseline_row = {
            "record_type": "baseline",
            "sample_index": sample_index,
            "sample_id": sample.id,
            "question": sample.input,
            "target": sample.target,
            "generated_tokens": len(base_token_ids),
            "final_answer": base_answer,
            "is_correct": base_correct,
            "full_response": base_text,
        }
        write_example_header(example_log_path, baseline_row)

        for percent, token_index in selected_positions(len(base_token_ids)):
            top10 = token_logprobs(base_output.logprobs[token_index], args.top_logprobs)
            append_position_log(example_log_path, percent, token_index, top10)
            prefix_ids = base_token_ids[:token_index]
            for choice_rank, choice in enumerate(top10[: args.branch_top_k], start=1):
                forced_ids = prefix_ids + [choice["token_id"]]
                forced_text = tokenizer.decode(forced_ids, skip_special_tokens=True)
                remaining_tokens = max(1, args.max_tokens - len(forced_ids))
                branch_prompt = prompt + forced_text
                branch_params = SamplingParams(
                    max_tokens=remaining_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                continuation = llm.generate([branch_prompt], branch_params, use_tqdm=False)[0].outputs[0]
                full_response = forced_text + continuation.text
                final_answer = extract_answer(full_response)
                is_correct = is_correct_answer(final_answer, sample.target)
                row = {
                    "record_type": "branch",
                    "sample_index": sample_index,
                    "sample_id": sample.id,
                    "question": sample.input,
                    "target": sample.target,
                    "baseline_generated_tokens": len(base_token_ids),
                    "baseline_answer": base_answer,
                    "baseline_correct": base_correct,
                    "percent": percent,
                    "token_index": token_index,
                    "choice_rank": choice_rank,
                    "choice_token_id": choice["token_id"],
                    "choice_token": choice["token"],
                    "choice_probability": choice["probability"],
                    "choice_logprob": choice["logprob"],
                    "top10_tokens": top10,
                    "final_generated_tokens": len(forced_ids) + len(continuation.token_ids),
                    "final_answer": final_answer,
                    "is_correct": is_correct,
                    "full_response": full_response,
                }
                append_branch_log(example_log_path, row)
        print(f"sample {sample_index}: baseline_tokens={len(base_token_ids)} branches_logged")

    print(f"example logs: {output_dir / 'examples'}")


if __name__ == "__main__":
    main()
