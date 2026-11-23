# Per-prompt last-k-row A curvature

Compute one final curvature per head and key token along `layer_input -> v_proj -> Att_out -> o_proj`, using the last `min(k, prompt_length)` query rows. Every prompt and every key token keeps its own score. Saved per-head rankings sort **raw curvature descending**; compression instead keeps **more negative curvature**. Scores are never minimized across tokens or prompts.

`run.sh` defaults to `LAST_K=1 QUERY_AGGREGATION=attention_mean`, using each head's original final A row unchanged and calculating and saving fresh scores (`RUN_CALCULATION=1 SAVE_CURVATURE=1 EVALUATE=0`). With one selected row, `attention_mean` passes it through without averaging across queries or heads. The Python CLI has the same row-selection defaults.

| `QUERY_AGGREGATION` / `--query-aggregation` | Calculation |
| --- | --- |
| `attention_mean` | Average only nonzero selected A values per head/token, then calculate curvature; all-zero tokens stay zero |
| `attention_max` | Take the maximum selected A value per head/token, then calculate curvature |
| `curvature_mean` | Calculate each selected row's curvature, then average finite results per head/token |
| `curvature_min` | Calculate each selected row's curvature, then take the minimum finite result per head/token |
| `curvature_both` | Calculate each selected row's curvature once, accumulating both finite mean and minimum; save in separate folders |

`attention_mean` divides each head/token's selected attention sum by its number of nonzero entries, excluding both causal zeros and other exact zeros. For example, `[0, 0, 0.6, 0.8]` becomes `0.7`; all-zero entries remain zero. Attention-first modes do not renormalize the reduced A values and use the final query's `o_proj` output neighbors. Curvature-first modes use each row's matching `o_proj` output and exclude future key positions from the reduction. Zero-attention entries are absent edges: no OT is run for them, and their curvature remains `NaN`. Mean/min exclude these entries and other undefined results; a token with no finite contribution remains `NaN`. Rows are processed individually without retaining a tensor of all per-row curvatures. Query aggregation is separate from `CURVATURE_REDUCTION`, which reduces heads during evaluation.

```bash
LAST_K=32 QUERY_AGGREGATION=curvature_both bash graph_curvature/run.sh
# Alternatives: attention_mean, attention_max, curvature_mean, curvature_min.
```

For `curvature_both`, each dataset output contains `curvature_min/` and `curvature_mean/`, each with its own `settings.json`, `layer_NNN/prompt_NNNNN.pt`, and analysis summaries. The model forward pass and per-row OT solves are shared; only running min, sum, and count tensors of shape `[heads, tokens]` are retained. Resume checks both folders and reuses completed files independently. `EVALUATE=1` evaluates both folders separately; the standalone evaluator can take either folder as `--curvature-dir`. Online compression requires a single aggregation mode.

Each folder saves `curvature_scores["A"]` with the same `[query_heads, key_tokens]` shape as the other modes. For head `h` and key token `t`, it contains the minimum or mean of finite curvatures of nonzero `A[h, selected_query, t]` edges. Heads and tokens are never combined during this query reduction. For evaluation, choose that saved query reduction through `CURVATURE_DIR`; `PRESSES=curvature_min,curvature_mean,...` still selects the independent reduction **across heads**, as before:

```bash
CURVATURE_DIR=logs/graph_curvature/my_run/longbench_lcc/curvature_min \
PRESSES=curvature_min,curvature_mean bash graph_curvature/run_eval.sh
# Set CURVATURE_DIR to the sibling curvature_mean folder to evaluate query means.
```

The selected k, aggregation mode, and algorithm version are checked on resume. Use a different `OUTPUT_DIR` for each setting combination.

## Run / 运行

Use Python with PyTorch, Transformers, Accelerate, NumPy, POT (`ot`), and `lm_eval[longbench]`:

```bash
PYTHON_BIN=python MODEL_PATH="<model-id-or-path>" bash graph_curvature/run.sh
```

`run.sh` accepts environment overrides; `run_experiment.py --help` lists CLI options. Reusing the same output directory and calculation settings skips existing `.pt` files and calculates only missing ones. Prompts are **not truncated by default**. Set `MAX_LENGTH` / `--max-length` only when an explicit token cap is wanted. Every positive selected attention entry is evaluated; there is no feature-edge or token sampling.

`run.sh` 支持环境变量覆盖，CLI 参数见 `run_experiment.py --help`。相同输出目录和计算设置下会复用已有 `.pt`，仅计算缺失文件。默认使用完整 prompt；仅在需要截断时设置 `MAX_LENGTH` / `--max-length`。

## Cache reuse and compression evaluation

`run.sh` calculates/saves curvature and evaluates it in separate processes. `run_online.sh` calculates curvature during prefill, compresses, then decodes and scores. Each launcher has its own editable settings and accepts environment overrides; neither sources or dispatches to the other:

| Variable | Purpose | Default |
| --- | --- | --- |
| `RUN_CALCULATION` | Calculate or resume offline curvature (`1`), or skip that stage (`0`); offline launcher only | `1` |
| `EVALUATE` | Run compression and accuracy comparison (`1`), or skip it (`0`) | Offline: `0` |
| `TASKS` | Comma-separated datasets, processed sequentially; `TASK` still selects one dataset | Offline: `longbench_lcc,longbench_trec` |
| `SAVE_CURVATURE` | Write curvature `.pt` files (`1`), or keep curvature transient (`0`) | Offline: `1`; online: `0` |
| `SCORE_SOURCE` | Use original method scores (`native`), or curvature scoring for SnapKV/PyramidKV/ExpectedAttention (`curvature`) | `curvature` |
| `CURVATURE_REDUCTION` | Reduce query heads with `min` or `mean` for those three variants; direct curvature methods retain their named reduction | `min` |
| `CURVATURE_RANKING` | Rank direct curvature scores per `layer`, or across the full `model` | `layer` |
| `OUTPUT_DIR` | Run root; multiple datasets get separate `OUTPUT_DIR/<task>` directories | New timestamped directory |
| `CURVATURE_DIR` | Saved score folder, or paired-output root when using `run.sh`; `run_eval.sh` takes one score folder | `OUTPUT_DIR` |
| `EVALUATION_OUTPUT_DIR` | Optional evaluation root; multiple datasets get separate `<task>` subdirectories | Each task's run directory plus `/evaluation` |

The offline launcher defaults to each task's complete LongBench test set: 500 prompts for LCC and RepoBench-P, 150 for MultiFieldQA-en, and 200 for the other tasks. Set `NUM_SAMPLES` to override the count for all selected tasks. Complete rendered prompts measured with the cached Granite 3.3 2B tokenizer (no truncation) have these mean token lengths:

| Dataset | Full test set | First 50 prompts |
| --- | ---: | ---: |
| LCC | 3,629 | 3,657 |
| Qasper | 5,729 | 6,191 |
| TREC | 8,055 | 7,487 |

These are shorter cached options than HotpotQA (16,276 tokens on average); individual prompts can still be longer. Override with `TASK=longbench_lcc` or `TASKS=longbench_lcc,longbench_qasper`. A single dataset keeps the original flat output layout. For a legacy saved directory containing `settings.json`, `RUN_CALCULATION=0 CURVATURE_DIR=... bash graph_curvature/run.sh` evaluates that one recorded dataset directly.

Offline calculation and comparison run in separate processes to release calculation memory before inference. By default, `run.sh` calculates and saves a fresh run; set `EVALUATE=1` to evaluate afterward. Use the same `OUTPUT_DIR` and settings to resume it. A complete score set skips loading the calculation model. Partial runs still propagate hidden states through the layers, but skip OT and projection-cost construction wherever scores already exist. Calculation settings, neighbor quantile, and algorithm version must match the saved run.

```bash
RUN_CALCULATION=1 OUTPUT_DIR=logs/graph_curvature/my_last32_run NUM_SAMPLES=50 bash graph_curvature/run.sh
# Resume the same calculation and evaluate on exactly those prompts:
RUN_CALCULATION=1 EVALUATE=1 OUTPUT_DIR=logs/graph_curvature/my_last32_run NUM_SAMPLES=50 bash graph_curvature/run.sh
```

To compare an existing score directory using the shell controls:

```bash
RUN_CALCULATION=0 EVALUATE=1 \
CURVATURE_DIR=logs/graph_curvature/20261107_190946 \
bash graph_curvature/run.sh
# Calculation only:
RUN_CALCULATION=1 EVALUATE=0 OUTPUT_DIR=logs/graph_curvature/my_last32_mean_run \
bash graph_curvature/run.sh
```

That legacy run contains **0.9-quantile, algorithm-version-1** curvature. New calculations use **0.70, version 7**. Version 7 leaves zero-attention curvature undefined and supports shared mean/min calculation. Comparison-only mode can evaluate older runs and records their original quantile/version. Resuming calculation requires matching versions so corrected OT results are not mixed with older files.

The launcher comparison is an uncompressed baseline plus `curvature_min`, `curvature_second_min`, `curvature_mean`, `SnapKVPress`, `ExpectedAttentionPress`, and `PyramidKVPress`, at removal ratios `0.3,0.5,0.7,0.9`. Configure `PRESSES`, `COMPRESSION_RATIOS`, and optional `MAX_NEW_TOKENS` in the launcher, or `--methods`, `--compression-ratios`, and `--max-new-tokens` in the standalone evaluator. Generation is greedy, with the saved seed and the task's generation cap unless overridden.

For online calculation without saving `.pt` files:

```bash
TASK=longbench_lcc PRESSES=curvature_mean COMPRESSION_RATIOS=0.5 \
bash graph_curvature/run_online.sh
# Replace attention scoring in SnapKV/PyramidKV/ExpectedAttention, online:
SCORE_SOURCE=curvature PRESSES=SnapKVPress,PyramidKVPress,ExpectedAttentionPress \
bash graph_curvature/run_online.sh
# The same variants using an existing saved log, with no new curvature calculation:
RUN_CALCULATION=0 CURVATURE_DIR=logs/graph_curvature/20261108_131908 \
SCORE_SOURCE=curvature PRESSES=SnapKVPress,PyramidKVPress,ExpectedAttentionPress \
bash graph_curvature/run.sh
```

Online mode computes curvature from the actual normalized layer input, native uncompressed V cache, last-query attention using the model's scale, and actual O-projection output. It immediately compresses that layer's prefill cache, then continues the forward pass and decoding. Only the current layer's curvature operations/projection costs are retained, and decoding does not calculate curvature. Each method/rate runs its own prefill and recomputes its curvature; use a smaller `PRESSES`/`COMPRESSION_RATIOS` selection to avoid repeating expensive OT. `SAVE_CURVATURE=1` uses the same score format as offline calculation so the resulting run can later be evaluated from disk. Online mode requires a fresh output directory and `EVALUATE=1`. Offline calculation with `SAVE_CURVATURE=0` requires `EVALUATE=0`.

Curvature variants use **the saved or online aggregated curvature**, with the query window and aggregation recorded in the run settings. The SnapKV variant uses finite-only pooling and retains priority for recent window tokens; PyramidKV also keeps its layer allocation and exact total KV budget. ExpectedAttention replaces its statistical attention predictor with softmax of negative reduced curvature, retaining its optional value-norm scaling and sink-token priority. StreamingLLM's position policy and Knorm's key norms do not use attention scores and remain unchanged. These are curvature variants of the methods, not their original attention algorithms; `comparison.json` records the score source, head reduction, and affected methods once.

The direct `curvature_min`, `curvature_second_min`, and `curvature_mean` methods **remove the highest raw-curvature tokens first** and retain the lowest-curvature tokens. After taking minimum, second minimum, or finite-only mean across query heads for each token, kvpress receives the negative of that reduced curvature because its top-k operation selects tokens to keep. The same token scores are shared by all KV heads without copying them. The second minimum counts tied values separately and requires two finite heads. The stored per-head descending `curvature_ranking` is not used for the cross-head reduction. Each layer's score file is memory-mapped on CPU when needed; full model attention matrices are not retained.

Set `CURVATURE_RANKING=model` (CLI: `--curvature-ranking model`) to rank those reduced scores jointly over all layers for each prompt. Layers can retain different numbers of tokens, including zero. The total retained budget remains `num_layers * max(1, int(prompt_length * (1 - removal_ratio)))`, matching the existing comparison methods and integer rounding. No minimum is imposed on individual layers. Equal scores prefer earlier layers, then earlier tokens; undefined scores have lowest retention priority. The default `layer` mode and the allocation/pooling/window/sink policies of other methods, including curvature-scored variants, remain unchanged. `comparison.json` records the ranking mode and the direct methods it applies to.

```bash
CURVATURE_RANKING=model PRESSES=curvature_min,curvature_mean \
bash graph_curvature/run_eval.sh
```

Full-model ranking retains only one CPU score per layer/token while ranking. Saved-score evaluation builds the selection before compressing the first layer and still compresses caches as each layer finishes. Online evaluation computes each layer's curvature once, waits for the final layer's scores, then compresses all caches before decoding; its peak prefill memory therefore includes the full uncompressed KV cache. It does not copy full caches or retain full-model curvature operations. Generated tokens remain uncompressed in both modes.

Every method uses the same first N task documents. New runs do not save `prompts.pt`; evaluation reconstructs tokenization from the recorded task/settings and checks token counts. Exact identity assumes the cached dataset and tokenizer are unchanged. Existing runs that contain `prompts.pt` can still use their saved token IDs and prompt-text hashes. Prompts are never silently truncated for evaluation: the prompt plus generation must fit the model context.

Compression happens once after prefill. All methods retain exactly the same **total number of KV elements** at a given rate; the evaluator verifies the physical cache. PyramidKV keeps its varying, nonincreasing layer allocation, with integer rounding adjusted to match that total. SnapKV and PyramidKV reconstruct window attention using the model's actual attention scale, including Granite's configured multiplier. Generated tokens subsequently grow the cache normally.

One `evaluation/comparison.json` stores shared data, seed, saved `.pt` location, curvature, and generation settings once. Each case contains only `method`, `removal_ratio`, and `accuracy` on a 0–1 scale. Scores are LongBench task metrics, such as code similarity for LCC, rather than a universal accuracy measure. Physical cache budgets are still verified during inference. No CSV or PDF copies are generated. Per-prompt metrics stored in new score files preserve complete summaries when calculation resumes.

The A40 host has 48 physical CPU cores / 96 logical CPUs. Both entry points default to 48 CPU workers, capped by active key tokens. Each worker builds one token's path blocks and solves OT locally, sharing inherited read-only inputs through `fork`. Queues carry token indices and compact results. Torch and OT use one thread per worker; the launcher also defaults OpenMP/BLAS to one thread. `WORKERS=1` / `--workers 1` uses serial execution.

A40 主机有 48 个物理 CPU 核心、96 个逻辑 CPU，默认使用 48 个 CPU worker，实际数量不超过有效 key token 数。worker 通过 `fork` 共享只读输入，在进程内计算路径和 OT；队列仅传递 token 索引和紧凑结果。Torch、OT 每个 worker 使用单线程，启动脚本中 OpenMP/BLAS 默认也为单线程。设置 `WORKERS=1` / `--workers 1` 使用串行计算。

Offline calculation moves model layers one at a time onto `DEVICE=cuda:0`; online compression uses the evaluator's inference model. Path calculation defaults to `COMPUTE_DEVICE=cpu`. CUDA path calculation, when selected, runs in the parent process; workers solve CPU OT. Curvature defaults to `float32`; use `CURVATURE_DTYPE=float64` / `--curvature-dtype float64` for higher precision. `alpha` defaults to `0.5` in `run.sh` and `0.0` in the Python CLI. `settings.json` records the value used.

## Graph and transport definition / 图和传输定义

- `layer_input` is the normalized input actually passed to the V projection. For key token `s`, incoming neighbor probabilities use **only `layer_input[0, s]`**.
- Outgoing neighbor candidates are **all features of the relevant query's `o_proj` output**, across all heads: the final query for attention-first modes, or the individual query for curvature-first modes.
- V and O projection costs remain independent FC matrices, `C_v = 1 / abs(W_v.T)` and `C_o = 1 / abs(W_o.T)`. Zero weights have infinite cost. The current attention cost is independently `c[h,s] = 1 / abs(A_selected[h,s])`, using either an individual row or the reduced attention values.
- Use the **0.70 quantile**: keep positive probabilities strictly above their 70th percentile (approximately the largest 30%), then renormalize the retained neighbor mass to `1-alpha`. The existing normalization, tie, and empty-neighbor behavior is retained. The quantile is applied over all input/output candidate features, not separately per head.
- Each A coefficient connects its V-head feature set to its query-head feature set. GQA maps query head `h` to V head `h // repeat`. Center masses do not depend on V or `Att_out` activation values. Distances to/from a head's feature set are shortest distances over its member features.

入邻居使用当前 key token 的 `layer_input[0, s]`，出邻居候选为对应 query 的全部 O 特征。V/O 的 FC 代价与当前 A 的代价独立计算，均使用 `1 / abs(weight)`。分位点为 **0.70**，保留严格大于第 70 百分位的正概率（约前 30%），再归一化为总质量 `1-alpha`。归一化、相同值和空邻居的处理保持原样。

For each token `s`, shortest paths preserve the actual feature graph: an attention edge connects matching feature positions within the appropriate heads. Construct `D_XO(s)` from the full `C_v -> c[:,s] -> C_o` adjacency. Take minima over complete feature paths; do not independently minimize the V and O matrices before composing them, which could join different feature positions incorrectly.

For fixed input/output neighbors, the OT matrix contains the shortest distance for every neighbor pair through the prev/current/next adjacency matrices. Infinite adjacency entries are missing connections: a pair's final cost is still finite whenever an alternative finite-cost path connects it. Only pairs with no finite route retain infinite shortest-path distance.

The implementation retains the adaptive blocked min-plus dynamic programming used by the earlier shortest-path utilities. For V feature `v`, it first computes the shortest `A -> O` continuation over matching repeated query heads, then composes it with V-projection costs:

```text
B[v,o] = min_h(c[h,s] + C_o[h*head_dim + v%head_dim, o])
          over heads h mapped to v's V head
D_XO[x,o] = min_v(C_v[x,v] + B[v,o])
```

Only active input/output neighbors are evaluated. The current attention adjacency is handled sparsely, without constructing a dense full adjacency matrix. GPU memory availability sets intermediate/output block sizes; additional row blocking bounds the broadcast temporary to 8 MiB on CPU or 64 MiB on CUDA with the default adaptive blocks. These bounds exclude the output matrices and reusable V/O caches. Value-only reductions avoid unused argmin indices; selected V rows are reused, and head-boundary costs are reduced on the compute device before CPU transfer.

每个 token 的 X→O 最短路径保留真实的特征连接关系，先组合完整的 `C_v -> c[:,s] -> C_o` 路径，再取最短路径；不能先对 V/O 的头内特征各自取最小值后再相加，否则可能错误连接不同特征。

For A entry `(h,s)`, let `U_h` be its V-head feature set and `T_h` its query-head feature set. The transport cost matrix is:

```text
                    O output neighbors       target center T_h
X[s] neighbors      D_XO(s)                   D_X,U_h + c[h,s]
source center U_h   D_U_h,O(s)                c[h,s]
```

`D_X,U_h` and `D_U_h,O(s)` are shortest distances to/from the source feature set in that same feature graph. Solve `W1 = OT(mu_s, nu, cost)` once per A entry and compute:

```text
curvature[h,s] = (1 - W1 / c[h,s]) / (1 - alpha)
```

With `alpha=0` and nonempty neighbor sets, only `D_XO(s)` contributes to OT. Its transport distance is shared across heads and solved once per key token; each head still gets its own curvature using its own `c[h,s]`.

Transport marginals use float64 normalization with center mass exactly `alpha`. Zero-mass centers are omitted from the OT solve. For infinite path costs, a bounded finite penalty allows the solver to find a feasible plan; the returned plan is checked for forbidden-edge flow. Infeasible transport remains undefined. This handling is recorded as curvature algorithm version 2.

当 `alpha=0` 且两侧邻居均非空时，OT 只使用共享的 `D_XO(s)`，因此每个 key token 仅求解一次 OT，再用各 head 自己的 A 代价得到独立曲率。

## Outputs / 输出

`settings.json` records the run configuration, prompt token lengths, quantile, algorithm version, graph definition, and raw descending ranking order. Scores are written immediately to:

```text
layer_000/prompt_00000.pt
layer_000/prompt_00001.pt
layer_001/prompt_00000.pt
...
```

Each file contains:

| Field | Meaning |
| --- | --- |
| `layer_id`, `prompt_id` | Layer and prompt indices |
| `token_count`, `query_index` | Prompt length and final query position |
| `query_start_index`, `query_count`, `query_aggregation` | Actual selected row window and reduction mode |
| `attention_scores` | Reduced A, `[query_heads, key_tokens]`; nonzero-only mean for curvature-first display; original dtype when k=1, curvature dtype otherwise |
| `attention_rows` | Original selected A rows and dtype, `[query_heads, query_count, key_tokens]`; present when query_count > 1 |
| `curvature_scores["A"]` | Aggregated curvature, `[query_heads, key_tokens]` |
| `curvature_ranking` | Key-token indices per head, highest curvature first |
| `metrics` | Per-prompt cost, neighbor, and curvature summaries, reused on resume |

Zero attention connections have no curvature (`NaN`) and are skipped by OT. Nonfinite attention, nonfinite reciprocal costs for nonzero attention, and unsuccessful OT solves also have `NaN` curvature. Invalid attention edges are excluded from shortest paths, so they cannot contaminate another head's finite route. Undefined final scores are excluded from ranking and appear as `-1` at the end of the ranking row. Equal finite curvatures preserve key-token order. Ranking is by curvature itself; raw attention rows are also saved.

每个文件保存一个 layer、一个 prompt 的聚合 A、对应曲率和 token 排序，形状为 `[query_heads, key_tokens]`，并保存选中的原始 A 行。无有效曲率时为 `NaN`，排序末尾以 `-1` 标记；相同曲率按 token 原顺序排列。

`analysis/layer_NNN_A_summary.txt` keeps separate curvature summaries for every prompt. Its overall section pools cost means, neighbor counts, and score counts only; it does not reduce prompt score tensors. `prev`, `cur`, and `next` cost means refer to V projection, A coefficients, and O projection respectively. Neighbor counts exclude center nodes. `total_edges` counts final head/token scores; in curvature-first modes, `edge_evaluations` counts evaluated finite A edges across the selected rows.

## Validation / 验证

Version 7 passed 24 small real-OT fixtures covering single and paired curvature reductions, attention-first modes, k=1/3/6, float32/float64 curvature, float32/bfloat16 attention, and MHA/GQA. Paired outputs matched explicit finite-only reductions of individually calculated rows, with one row solve per selected query. Additional checks covered all-zero attention without token tasks, multiprocessing parity, separate saved settings/manifests/rankings, complete and partial resume, version-6 rejection, and shell routing for paired/single runs and evaluation outputs. A final tiny two-layer Granite CPU integration checked real collection/OT/saving, per-head tensor shapes, all three direct evaluation scorers, curvature-variant loading, partial resume, k capped to prompt length, zero-edge exclusion, and unchanged model weights. No full dataset accuracy run was performed.

Version 6's nonzero-only attention mean passed 30 direct cases covering causal zeros, genuine zeros, all-zero tokens, k=1/3/32, and float32/float64/bfloat16 inputs. Curvature matched an explicitly reduced attention graph, saved scores matched the reduction, and native online/offline prefill agreed for k=1/3/32. Resuming version-5 results was rejected; temporary outputs were removed.

The initial version-5 audit at neighbor quantile 0.85 compared 480 curvature/aggregation cases against an independent full feature graph, Floyd–Warshall shortest paths, and a linear-program transport solver, plus 360 path-block cases and 150 transport cases. A further 192 graph/LP cases covered the corrected handling of NaN/infinite attention, all four aggregation modes, float32/float64, MHA/GQA, and serial/multiprocessing parity. Separate checks covered finite-only reduction after overflow and bounded storage for selected CPU slices.

On an A40, 128 CPU/GPU curvature cases matched, including GPU path construction with CPU OT workers. A real cached Granite 3.3 2B run through `run.sh` calculated and saved one prompt's first-layer curvature with 8 tokens and requested k=32, then resumed without recalculation. Its attention and projected features agreed with native model computations, saved curvature replayed exactly, and CPU/GPU curvature matched. This is a small model correctness check, not a full dataset accuracy run.

After the neighbor quantile changed to the current 0.70 setting, 192 additional independent distribution/graph/LP cases and 32 CPU/GPU cases passed. A fresh real Granite `run.sh` calculation saved finite scores with quantile 0.70 and rejected resuming results recorded with quantile 0.85.

CPU model checks compared Llama, Granite, and Qwen3 native hidden states/activations and attention with the collector in float32/bfloat16. Collection and all aggregation modes preserved model weights, input tensors, projection caches, and collected operations. End-to-end validation at quantile 0.85 used the real LongBench LCC scorer with a tiny two-layer Granite model and two prompts: 144 cases across all four aggregation modes, including saved-score evaluation, full-model ranking, and online compression. Physical KV budgets matched, and online/offline scores agreed. Complete/partial resume, changed-setting rejection, CLI options, and shell workflows also passed. Temporary validation scripts and outputs were removed.

The shortest-path optimization was checked against the preceding implementation: 56 blocked min-plus cases and 80 sparse MHA/GQA path cases, including empty neighbors and infinite edges, were unchanged. Twelve complete curvature cases and their metrics matched exactly in serial and multiprocessing execution. Full-adjacency Floyd–Warshall checks also passed within dtype precision. A one-thread CPU microbenchmark with Granite-like dimensions (V costs `2048x512`, O costs `2048x2048`, 32 query heads, head width 64, 614 active neighbors per side; median of five runs) measured 310.58 ms before and 50.37 ms after optimization. This is a shortest-path microbenchmark, not an end-to-end speed claim. A separate allocation check reduced the largest broadcast allocation from 18.75 MiB to 8 MiB. CUDA block-selection tiers were checked with mocked free-memory values; CUDA execution was not benchmarked.
