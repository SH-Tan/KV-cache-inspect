"""Compact token-probability logs and compression-rate score tables."""

import csv
from contextlib import contextmanager
import json

import torch
from kvpress import PyramidKVPress
from kvpress.utils import compute_n_kept


TASK_LABELS = {
    "longbench_narrativeqa": "NQA",
    "longbench_qasper": "Qser",
    "longbench_multifieldqa_en": "MF",
    "longbench_hotpotqa": "HpQA",
    "longbench_2wikimqa": "2WQA",
    "longbench_musique": "Mque",
    "longbench_gov_report": "GR",
    "longbench_qmsum": "QMS",
    "longbench_multi_news": "MN",
    "longbench_trec": "TREC",
    "longbench_triviaqa": "TQA",
    "longbench_samsum": "SSum",
    "longbench_passage_count": "PC",
    "longbench_passage_retrieval_en": "PR",
    "longbench_lcc": "Lcc",
    "longbench_repobench-p": "RBP",
}


def write_json(path, data):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, default=str)
        stream.write("\n")


@contextmanager
def record_compressed_prompt(press, context, tokenizer, record):
    """Decode retained positions from layer 0 / KV head 0, in prompt order."""
    score = press.score

    def capture_score(module, hidden_states, keys, values, attentions, kwargs):
        scores = score(module, hidden_states, keys, values, attentions, kwargs)
        if module.layer_idx == 0:
            n_kept = (press.get_layer_budget(module, keys.shape[2]) if isinstance(press, PyramidKVPress)
                      else compute_n_kept(keys.shape[2], press.compression_ratio))
            # Match ScorerPress's selection, including its handling of tied scores.
            positions = scores.topk(n_kept, dim=-1).indices[0, 0].sort().values
            token_ids = context[0].index_select(0, positions.to(context.device)).tolist()
            record["compressed_prompt"] = tokenizer.decode(token_ids)
            record["compressed_prompt_source"] = {"layer": 0, "kv_head": 0, "order": "original"}
        return scores

    press.score = capture_score
    try:
        yield
    finally:
        del press.score


def write_prompt_scores(task, sample_ids, log_ids, task_dir, case):
    """Add lm-eval's filtered-response metrics to the selected prompt logs."""
    for request in task.instances:
        doc_id = sample_ids[request.doc_id]
        if doc_id not in log_ids:
            continue
        path = task_dir / f"prompt_{doc_id}" / f"{case}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["scores"] = task.process_results(request.doc, [request.filtered_resps["none"]])
        write_json(path, record)


class GenerationStats:
    """Keep top-ten probabilities and cache-length snapshots on CPU, not KV tensors."""

    def __init__(self):
        self.steps = []
        self.prefill_cache_tokens = None
        self.final_cache_tokens = None

    def record(self, module, inputs, output):
        # Physical sequence dimension, not a cache's logical position counter.
        cache_tokens = [layer.keys.shape[-2] for layer in output.past_key_values.layers]
        if not self.steps:
            self.prefill_cache_tokens = cache_tokens
        self.final_cache_tokens = cache_tokens
        logits = output.logits[0, -1].detach().float()
        values, token_ids = torch.topk(logits, min(10, logits.numel()))
        # Equivalent to softmax over the whole vocabulary, not just the top ten.
        probabilities = torch.exp(values - torch.logsumexp(logits, dim=-1))
        self.steps.append((token_ids.tolist(), probabilities.tolist()))

    def checkpoints(self, generated_ids, tokenizer):
        checkpoints = []
        total_steps = len(generated_ids)
        for percent in (20, 50, 70, 90, 100):
            step = (percent * total_steps + 99) // 100
            token_ids, probabilities = self.steps[step - 1]
            top_tokens = []
            for token_id, probability in zip(token_ids, probabilities):
                top_tokens.append({"token_id": token_id, "token": tokenizer.decode([token_id]),
                                   "probability": probability})
            generated_id = generated_ids[step - 1]
            checkpoints.append({"percent": percent, "step": step,
                                "generated_token_id": generated_id,
                                "generated_token": tokenizer.decode([generated_id]),
                                "top_10": top_tokens})
        return checkpoints


def write_score_table(path, task_names, ratios, scores, *, model_path, press, do_sample, seed):
    """Write CSV and readable Markdown from the same percentage-score rows."""
    columns = [TASK_LABELS.get(task, task) for task in task_names]
    rows = []
    for ratio in ratios:
        values = [scores.get((ratio, task)) for task in task_names]
        values = ["" if value is None else 100 * value for value in values]
        average = sum(values) / len(values) if "" not in values else ""
        case = "baseline" if ratio == 0 else f"compression_{ratio}"
        rows.append([case, ratio] + values + [average])

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Model", "Press", "do_sample", "Seed", "Case", "C.R."] + columns + ["Avg."])
        for row in rows:
            writer.writerow([model_path, press, json.dumps(do_sample), seed] + row)

    with path.with_suffix(".md").open("w", encoding="utf-8") as stream:
        stream.write("# Scores vs. compression / 分数与压缩率\n\n")
        stream.write(f"Model / 模型: `{model_path}`  \n")
        stream.write(f"Press / 压缩方法: `{press}`  \n")
        stream.write(f"Sampling / 随机采样 (`do_sample`): `{json.dumps(do_sample)}`; Seed / 随机种子: `{seed}`\n\n")
        stream.write("C.R. is the fraction of prefill KV entries removed. `baseline` (0) uses no compression.\n\n")
        stream.write("C.R. 表示预填充阶段删除的 KV 条目比例；`baseline`（0）表示不压缩。\n\n")
        headers = ["Case / 条件", "C.R."] + [task.removeprefix("longbench_") for task in task_names] + ["Avg."]
        stream.write("| " + " | ".join(headers) + " |\n")
        stream.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for case, ratio, *values in rows:
            cells = [case, str(ratio)] + ["—" if value == "" else f"{value:.2f}" for value in values]
            stream.write("| " + " | ".join(cells) + " |\n")
        stream.write("\nScores are each task's metric × 100, not a shared accuracy metric. Avg. is the unweighted mean across tasks.\n\n")
        stream.write("各列为对应任务的指标 × 100，并非统一的准确率。Avg. 是各任务分数的算术平均值。\n\n")
        stream.write("— means pending; Avg. appears when all tasks in that row finish. CSV retains unrounded values.\n\n")
        stream.write("— 表示尚未完成；同一行的所有任务完成后才显示 Avg.。CSV 保留未四舍五入的数值。\n")
