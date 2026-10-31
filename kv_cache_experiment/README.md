# KV Cache Experiment

Run lm-eval LongBench with one shared Hugging Face model, batch size 1, and a sweep of prefill KV-cache compression rates. `TASK_COUNTS` explicitly sets the evaluation size per task: by default, 200 each for Qasper, HotpotQA, GovReport, TREC, and passage counting, and 500 for LCC (1,500 total). `LIMIT=5` controls only how many prompts per task receive detailed logs. Each rate uses the same evaluation and logging document IDs and generation settings.

## Tasks

Six tasks are active by default, one from each category: **Qasper, HotpotQA, GovReport, TREC, passage counting, and LCC**. The other ten tasks are listed in comments beside `TASKS` in `run.sh` and can be enabled through that variable.

The full table contains 16 English/code tasks from [LongBench v1](https://github.com/THUDM/LongBench/blob/main/LongBench/README.md). `C.R.` and `Avg.` are result columns, not tasks.

| Category | Columns | lm-eval tasks (prefix each with `longbench_`) |
| --- | --- | --- |
| SD-QA | NQA, Qser, MF | `narrativeqa`, `qasper`, `multifieldqa_en` |
| MD-QA | HpQA, 2WQA, Mque | `hotpotqa`, `2wikimqa`, `musique` |
| Summ. | GR, QMS, MN | `gov_report`, `qmsum`, `multi_news` |
| FSL | TREC, TQA, SSum | `trec`, `triviaqa`, `samsum` |
| Synth. | PC, PR | `passage_count`, `passage_retrieval_en` |
| Code | Lcc, RBP | `lcc`, `repobench-p` |

Uses the installed [lm-eval LongBench task definitions](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/longbench): prompts, reference answers, stopping rules, maximum new tokens, and metrics. No extra few-shot examples or chat template are added; few-shot task examples already appear in the dataset context.

## Run and compression rates

```bash
bash kv_cache_experiment/run.sh
COMPRESSION_RATIOS=0.1,0.3,0.5,0.7,0.9 LIMIT=20 bash kv_cache_experiment/run.sh
TASKS=longbench_qasper COMPRESSION_RATIOS=0.5 LIMIT=2 bash kv_cache_experiment/run.sh
TASKS=longbench_qasper TASK_COUNTS='{"longbench_qasper":10}' LIMIT=2 bash kv_cache_experiment/run.sh
```

The third example evaluates all 200 Qasper prompts and logs two; the fourth evaluates ten and logs two. Evaluation and logging subsets are selected with `SEED`. Evaluation counts must be positive and are capped at the available dataset size. `LIMIT=0` disables detailed prompt logs while keeping aggregate results. `TASK_COUNTS` defaults cover all 16 supported tasks: 150 for MultiFieldQA-en, 500 each for LCC and RepoBench-P, and 200 each for the others. An overridden `TASK_COUNTS` must include every selected task.

All configurable defaults live in `run.sh` and can be overridden with environment variables. Batch size is fixed at 1. Default rates are `0.1,0.2,0.5,0.8,0.9`. Rate `0` is automatically included as an uncompressed baseline, evaluated once per task. Duplicate rates are removed. `COMPRESSION_RATIO=0.5` remains supported for a single rate.

`C.R.` means the **fraction of prefill KV entries removed**: `0.8` requests removal of 80%, retaining about 20%. Values must be in `[0, 1)`. Actual retained counts are integers. The model is loaded once; every prompt/rate generates with a fresh cache, and each rate gets a fresh press instance.

The default press is `StreamingLLMPress`. It inherits the installed `kvpress/presses/base_press.py` hook, which checks `is_prefilling(cache_position, q_len)` and skips compression on decoding passes. `SnapKVPress` and `KnormPress` also inherit this hook. This runner accepts presses using that prefill-only hook. Generated-token KV entries are appended normally, so the cache grows during decoding; the configured removal fraction is not a fixed budget for the entire generation. With this post-attention prefill hook, the first output token's distribution is computed before the compressed cache is used by decoding.

## Logs and score curve

Default output directory:

```text
logs/kv_cache_experiment/<PRESS>/limit-5/<timestamp>/
  settings.json
  scores.csv
  longbench_qasper/
    sample_ids.json
    log_ids.json
    baseline_results.json
    compression_0.5_results.json
    prompt_<original_dataset_id>/
      baseline.json
      compression_0.5.json
      ...
```

`sample_ids.json` lists all evaluated document IDs; `log_ids.json` lists the subset receiving detailed logs. Aggregate scores use all evaluated prompts. Probability hooks and per-prompt output files are enabled only for the logging subset; lm-eval's full sample logs are disabled to avoid retaining extra copies of prompts and outputs.

Each logged prompt's baseline and compression JSON files contain:

- Full generated output and the output actually scored after lm-eval removes stopping strings.
- Reference answers, per-prompt metric scores, seed, rate, prompt-token count, and generated-token count.
- Top-10 token IDs, decoded tokens, and probabilities at 10%, 20%, 50%, 80%, 90%, and 100% of that run's actual generated length, plus the token actually generated at each checkpoint.

For an output of `N` tokens, checkpoint `p` uses 1-based step `ceil(p * N / 100)`. The first generated token is step 1. Counts include generated EOS/stop tokens before lm-eval text cleanup. Short outputs can share a step across several percentages. Baseline and compression can produce different lengths; percentages are calculated independently for each run.

Probabilities are `softmax(raw model logits)` over the **whole vocabulary**, before temperature, top-p, and top-k filtering. The top ten probabilities are not renormalized. Only ten token IDs/probabilities per step are retained on CPU until the final output length is known; only the six checkpoints are written. No full-vocabulary score history or KV tensors are retained in the logger.

`scores.csv` includes only the selected tasks. With the six defaults:

```text
C.R.,Qser,HpQA,GR,TREC,PC,Lcc,Avg.
```

Scores are percentages (`100 * lm-eval score`). They use each task's own metric: QA F1, ROUGE-L, classification/count/retrieval scores, or code similarity, rather than a single universal accuracy metric. `Avg.` is the unweighted mean over the selected task scores. Plot `C.R.` against a task column or `Avg.`. The CSV is updated after each task/rate; missing values and `Avg.` remain blank until their evaluations finish.

## Context, dependencies, and reproducibility

`MAX_CONTEXT_LENGTH=20480` is the default total input-plus-output token budget. lm-eval reserves each task's maximum new tokens and left-truncates longer inputs. Raise the budget within the model's position limit to retain more context. Logs store input token counts rather than the full prompt.

`LOCAL_FILES_ONLY=1` requires cached weights and LongBench data; set it to `0` to allow downloads. The Python environment needs `kvpress`, `transformers`, and `lm_eval[longbench]`.

`SEED` controls Python, NumPy, PyTorch CPU/CUDA, Transformers, document selection, and lm-eval's RNGs. It is set before model loading and reset before every generation. The launcher sets `PYTHONHASHSEED` and cuBLAS workspace configuration. Deterministic algorithms are required, cuDNN benchmarking is disabled, and CUDA/cuDNN TF32 are disabled.

Generation settings remain explicit: `DO_SAMPLE=0`, `TEMPERATURE=1.0`, `TOP_P=1.0`, `TOP_K=0`, one beam, and one returned sequence. Greedy generation ignores sampling filters. To sample reproducibly, set `DO_SAMPLE=1` and the desired temperature/top-p/top-k. Reproducibility assumes the same prompt, settings, compression method, and software/hardware stack.
