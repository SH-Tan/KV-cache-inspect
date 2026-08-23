import csv
from pathlib import Path


def _sample_score_value(sample):
    if not sample.scores:
        return None
    for score in sample.scores.values():
        value = score.value
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value == 1
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "correct", "c"}:
                return True
            if normalized in {"0", "false", "incorrect", "i"}:
                return False
    return None


def _sample_output_tokens(sample):
    if sample.model_usage:
        return sum(usage.output_tokens or 0 for usage in sample.model_usage.values())
    if sample.output and sample.output.usage:
        return sample.output.usage.output_tokens
    if sample.output:
        return len(sample.output.completion.split())
    return None


def _mean(values):
    return sum(values) / len(values) if values else ""


def _append_text_summary(row: dict[str, object], text_path: str) -> None:
    text_file = Path(text_path)
    text_file.parent.mkdir(parents=True, exist_ok=True)
    with text_file.open("a", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    f"model: {row['model']}",
                    f"task: {row['task']}",
                    f"log_file: {row['log_file']}",
                    f"total_samples: {row['total_samples']}",
                    f"scored_samples: {row['scored_samples']}",
                    f"correct_count: {row['correct_count']}",
                    f"incorrect_count: {row['incorrect_count']}",
                    f"unknown_count: {row['unknown_count']}",
                    f"avg_generated_tokens_correct: {row['avg_generated_tokens_correct']}",
                    f"avg_generated_tokens_incorrect: {row['avg_generated_tokens_incorrect']}",
                    f"avg_generated_tokens_unknown: {row['avg_generated_tokens_unknown']}",
                    "",
                ]
            )
        )


def append_token_length_summary(
    log,
    summary_path: str = "logs/token_length_summary.csv",
    text_summary_path: str = "logs/token_length_summary.txt",
) -> None:
    correct_lengths = []
    incorrect_lengths = []
    unknown_lengths = []
    total_samples = len(log.samples or [])

    for sample in log.samples or []:
        output_tokens = _sample_output_tokens(sample)
        if output_tokens is None:
            continue

        score_value = _sample_score_value(sample)
        if score_value is None:
            unknown_lengths.append(output_tokens)
        elif score_value:
            correct_lengths.append(output_tokens)
        else:
            incorrect_lengths.append(output_tokens)

    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model": log.eval.model,
        "task": log.eval.task,
        "log_file": log.location,
        "total_samples": total_samples,
        "scored_samples": len(correct_lengths) + len(incorrect_lengths),
        "correct_count": len(correct_lengths),
        "incorrect_count": len(incorrect_lengths),
        "unknown_count": len(unknown_lengths),
        "avg_generated_tokens_correct": _mean(correct_lengths),
        "avg_generated_tokens_incorrect": _mean(incorrect_lengths),
        "avg_generated_tokens_unknown": _mean(unknown_lengths),
    }
    write_header = not summary_file.exists()
    with summary_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "task",
                "log_file",
                "total_samples",
                "scored_samples",
                "correct_count",
                "incorrect_count",
                "unknown_count",
                "avg_generated_tokens_correct",
                "avg_generated_tokens_incorrect",
                "avg_generated_tokens_unknown",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    _append_text_summary(row, text_summary_path)
    print(f"  token length summary: {summary_file}")
    print(f"  token length text summary: {text_summary_path}")
