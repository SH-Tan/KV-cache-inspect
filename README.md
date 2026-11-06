# KV-cache-inspect

Tools for studying language-model behavior through KV-cache compression and graph curvature.

- [KV-cache experiments](kv_cache_experiment/README.md) compare model performance with and without cache compression.
- [Graph-curvature experiments](graph_curvature/README.md) collect model activations and calculate curvature over an operation graph.

## LongBench datasets

[LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://github.com/THUDM/LongBench/tree/main/LongBench) evaluates long-context understanding across six task categories. The table covers the 16 English/code tasks listed in this repository's KV-cache launcher. Task IDs in `lm_eval` use the prefix `longbench_` (for example, `longbench_qasper`).

| Dataset ID | Full name / description | Category | Score metric |
| --- | --- | --- | --- |
| `narrativeqa` | NarrativeQA: questions about stories and scripts | Single-document QA | Token F1 |
| `qasper` | Qasper: question answering over scientific papers | Single-document QA | Token F1 |
| `multifieldqa_en` | MultiFieldQA (English): questions across diverse domains | Single-document QA | Token F1 |
| `hotpotqa` | HotpotQA: questions requiring evidence from multiple documents | Multi-document QA | Token F1 |
| `2wikimqa` | 2WikiMultihopQA: multi-hop questions over Wikipedia documents | Multi-document QA | Token F1 |
| `musique` | MuSiQue: multi-hop questions composed from single-hop questions | Multi-document QA | Token F1 |
| `gov_report` | GovReport: government report summarization | Summarization | ROUGE-L F1 |
| `qmsum` | QMSum: query-based meeting summarization | Summarization | ROUGE-L F1 |
| `multi_news` | MultiNews: multi-document news summarization | Summarization | ROUGE-L F1 |
| `trec` | TREC (Text REtrieval Conference): question classification | Few-shot learning | Classification accuracy |
| `triviaqa` | TriviaQA: trivia question answering with examples | Few-shot learning | Token F1 |
| `samsum` | SAMSum: dialogue summarization with examples | Few-shot learning | ROUGE-L F1 |
| `passage_count` | PassageCount: count distinct passages in repeated text | Synthetic tasks | Counting accuracy |
| `passage_retrieval_en` | PassageRetrieval (English): identify the passage matching a summary | Synthetic tasks | Retrieval accuracy |
| `lcc` | Long Code Completion: predict the next code line | Code completion | Edit similarity |
| `repobench-p` | RepoBench-P: repository-level next-line code prediction | Code completion | Edit similarity |

### How the metrics are calculated

| Metric | Simple calculation (per example, 0–1) |
| --- | --- |
| Token F1 | Lowercase, remove punctuation and articles (`a`, `an`, `the`), then split into words. Count shared tokens, including repetitions. Precision `P = shared / predicted tokens`; recall `R = shared / reference tokens`; score `2PR / (P + R)`, or 0 with no overlap. |
| ROUGE-L F1 | Find the longest common subsequence: matching words in order, allowing gaps. With length `L`, use `P = L / predicted length`, `R = L / reference length`, and `2PR / (P + R)`. No overlap gives 0. |
| Classification accuracy | Find valid class labels appearing in the prediction, excluding shorter labels contained in the correct label. If the correct label appears among `k` matches, score `1/k`; otherwise 0. |
| Counting accuracy | Extract all predicted numbers. Score `occurrences of the correct count / total extracted numbers`; no numbers gives 0. |
| Retrieval accuracy | Extract all predicted numbers. Score `occurrences of the reference paragraph ID / total extracted numbers`; no numbers gives 0. |
| Edit similarity | Select the first generated code line without backticks, `#`, or `//`. Compare its characters with the reference using `fuzzywuzzy.fuzz.ratio / 100`; identical lines score 1. |

For example, predicting numbers `3, 4` when the correct count or paragraph ID is `3` scores `1/2 = 0.5`.

Higher scores are better. The official evaluator takes the best score across references for each example, then reports `100 × mean(example scores)`, rounded to two decimals. For TREC, TriviaQA, and SAMSum, it scores only the first prediction line after removing leading newlines. See the official [task descriptions](https://github.com/THUDM/LongBench/blob/main/LongBench/task.md), [metric mapping and aggregation](https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py), and [metric implementations](https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py).

## Getting started

1. Choose an experiment and install the dependencies listed in its README.
2. Configure its `run.sh` for your Python environment, model, dataset, available hardware, and output location.
3. Make the model and dataset available through local caches or enable downloads in the launcher settings.
4. Run the launcher from the repository root.

Settings can be edited in the launcher or overridden through environment variables. Use `PYTHON_BIN` to select your Python interpreter and `MODEL_PATH` to select a model identifier or local model directory. Check the launcher for current options and defaults.

Start with a small workload, inspect the results, and increase the experiment size to suit your CPU and GPU memory. Keep input selection and evaluation settings consistent when comparing runs.
