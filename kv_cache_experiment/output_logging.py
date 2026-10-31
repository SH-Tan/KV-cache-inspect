"""Compact token-probability logs and compression-rate score tables."""

import csv
import json

import torch


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


class TokenProbabilities:
    """Keep only ten IDs/probabilities per step on CPU, never full-vocabulary scores."""

    def __init__(self):
        self.steps = []

    def record(self, module, inputs, output):
        logits = output.logits[0, -1].detach().float()
        values, token_ids = torch.topk(logits, min(10, logits.numel()))
        # Equivalent to softmax over the whole vocabulary, not just the top ten.
        probabilities = torch.exp(values - torch.logsumexp(logits, dim=-1))
        self.steps.append((token_ids.tolist(), probabilities.tolist()))

    def checkpoints(self, generated_ids, tokenizer):
        checkpoints = []
        total_steps = len(generated_ids)
        for percent in (10, 20, 50, 80, 90, 100):
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


def write_score_table(path, task_names, ratios, scores):
    """Write percentage scores; Avg. is the unweighted mean over all selected tasks."""
    columns = [TASK_LABELS.get(task, task) for task in task_names]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["C.R."] + columns + ["Avg."])
        for ratio in ratios:
            values = []
            for task in task_names:
                score = scores.get((ratio, task))
                values.append("" if score is None else 100 * score)
            average = ""
            if "" not in values:
                average = sum(values) / len(values)
            writer.writerow([ratio] + values + [average])
