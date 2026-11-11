# Per-prompt last-row A curvature / 逐 prompt 的末行 A 曲率

Compute one curvature for each `A[head, last_query, key_token]` along `layer_input -> v_proj -> Att_out -> o_proj`. Every prompt and every key token keeps its own score. Saved per-head rankings sort **raw curvature descending**; compression instead keeps **more negative curvature**. Scores are never minimized across tokens or prompts.

沿 `layer_input -> v_proj -> Att_out -> o_proj` 路径，为每个 `A[head, last_query, key_token]` 计算一个曲率。每个 prompt、每个 key token 独立保留分数，每个 query head 内按曲率从高到低排序，不对 token 或 prompt 取最小值。

## Run / 运行

Use Python with PyTorch, Transformers, Accelerate, NumPy, POT (`ot`), and `lm_eval[longbench]`:

```bash
PYTHON_BIN=python MODEL_PATH="<model-id-or-path>" bash graph_curvature/run.sh
```

`run.sh` accepts environment overrides; `run_experiment.py --help` lists CLI options. Reusing the same output directory and calculation settings skips existing `.pt` files and calculates only missing ones. Prompts are **not truncated by default**. Set `MAX_LENGTH` / `--max-length` only when an explicit token cap is wanted. Every positive last-row attention entry is evaluated; there is no feature-edge or token sampling.

`run.sh` 支持环境变量覆盖，CLI 参数见 `run_experiment.py --help`。相同输出目录和计算设置下会复用已有 `.pt`，仅计算缺失文件。默认使用完整 prompt；仅在需要截断时设置 `MAX_LENGTH` / `--max-length`。所有正值末行注意力项都会计算，不再进行特征边或 token 采样。

## Cache reuse and compression evaluation

`run.sh` calculates/saves curvature and evaluates it in separate processes. `run_online.sh` calculates curvature during prefill, compresses, then decodes and scores. Each launcher has its own editable settings and accepts environment overrides; neither sources or dispatches to the other:

| Variable | Purpose | Default |
| --- | --- | --- |
| `RUN_CALCULATION` | Calculate or resume offline curvature (`1`), or skip that stage (`0`); offline launcher only | `0` |
| `EVALUATE` | Run compression and accuracy comparison (`1`), or skip it (`0`) | `1` |
| `TASKS` | Comma-separated datasets, processed sequentially; `TASK` still selects one dataset | `longbench_lcc` |
| `SAVE_CURVATURE` | Write curvature `.pt` files and prompt-token manifest (`1`), or keep curvature transient (`0`) | Offline: `1`; online: `0` |
| `SCORE_SOURCE` | Use original method scores (`native`), or curvature scoring for SnapKV/PyramidKV/ExpectedAttention (`curvature`) | `curvature` |
| `CURVATURE_REDUCTION` | Reduce query heads with `min` or `mean` for those three variants; `curvature_min`/`curvature_mean` retain their named reduction | `mean` |
| `OUTPUT_DIR` | Run root; multiple datasets get separate `OUTPUT_DIR/<task>` directories | New timestamped directory |
| `CURVATURE_DIR` | Saved run directory used for comparison; contains `settings.json` and layer `.pt` files | `OUTPUT_DIR` |
| `EVALUATION_OUTPUT_DIR` | Optional evaluation root; multiple datasets get separate `<task>` subdirectories | Each task's run directory plus `/evaluation` |

Both launchers default to LCC with 50 samples. Qasper and TREC remain optional shorter datasets. Complete rendered prompts measured with the cached Granite 3.3 2B tokenizer (no truncation) have these mean token lengths:

| Dataset | Full test set | First 50 prompts |
| --- | ---: | ---: |
| LCC | 3,629 | 3,657 |
| Qasper | 5,729 | 6,191 |
| TREC | 8,055 | 7,487 |

These are shorter cached options than HotpotQA (16,276 tokens on average); individual prompts can still be longer. Override with `TASK=longbench_lcc` or `TASKS=longbench_lcc,longbench_qasper`. A single dataset keeps the original flat output layout. For a legacy saved directory containing `settings.json`, `RUN_CALCULATION=0 CURVATURE_DIR=... bash graph_curvature/run.sh` evaluates that one recorded dataset directly.

Offline calculation and comparison run in separate processes to release calculation memory before inference. By default, `run.sh` evaluates the saved LCC curvature in `logs/graph_curvature/20261108_131908`. Set `RUN_CALCULATION=1` to calculate, save, and evaluate a fresh run; use the same `OUTPUT_DIR` and settings to resume it. A complete score set skips loading the calculation model. Partial runs still propagate hidden states through the layers, but skip OT and projection-cost construction wherever scores already exist. Calculation settings, neighbor quantile, and algorithm version must match the saved run.

```bash
RUN_CALCULATION=1 OUTPUT_DIR=logs/graph_curvature/my_quantile70_run NUM_SAMPLES=50 bash graph_curvature/run.sh
# Resume the same calculation and evaluate on exactly those prompts:
RUN_CALCULATION=1 OUTPUT_DIR=logs/graph_curvature/my_quantile70_run NUM_SAMPLES=50 bash graph_curvature/run.sh
```

To compare an existing score directory using the shell controls:

```bash
RUN_CALCULATION=0 EVALUATE=1 \
CURVATURE_DIR=logs/graph_curvature/20261107_190946 \
bash graph_curvature/run.sh
# Calculation only:
RUN_CALCULATION=1 EVALUATE=0 OUTPUT_DIR=logs/graph_curvature/my_quantile70_run \
bash graph_curvature/run.sh
```

That legacy run contains **0.9-quantile, algorithm-version-1** curvature. New calculations use **0.7, version 2**. Comparison-only mode can evaluate older runs and records their original quantile/version. Resuming calculation requires matching versions so corrected OT results are not mixed with older files.

The launcher comparison is an uncompressed baseline plus `curvature_min`, `curvature_mean`, `SnapKVPress`, `ExpectedAttentionPress`, and `PyramidKVPress`, at removal ratios `0.3,0.5,0.7,0.9`. Configure `PRESSES`, `COMPRESSION_RATIOS`, and optional `MAX_NEW_TOKENS` in the launcher, or `--methods`, `--compression-ratios`, and `--max-new-tokens` in the standalone evaluator. Generation is greedy, with the saved seed and the task's generation cap unless overridden.

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

Curvature variants use **only the last prompt query's curvature**, not SnapKV's original multi-query attention estimate. The SnapKV variant uses finite-only pooling and retains priority for recent window tokens; PyramidKV also keeps its layer allocation and exact total KV budget. ExpectedAttention replaces its statistical attention predictor with softmax of negative reduced curvature, retaining its optional value-norm scaling and sink-token priority. StreamingLLM's position policy and Knorm's key norms do not use attention scores and remain unchanged. These are curvature variants of the methods, not their original attention algorithms; `comparison.json` records the score source, head reduction, and affected methods once.

The direct `curvature_min` and `curvature_mean` methods **remove the highest raw-curvature tokens first** and retain the lowest-curvature tokens. After taking minimum or finite-only mean across query heads for each token, kvpress receives the negative of that reduced curvature because its top-k operation selects tokens to keep. The same token scores are shared by all KV heads without copying them. The stored per-head descending `curvature_ranking` is not used for the cross-head reduction. Each layer's score file is memory-mapped on CPU when needed; full model attention matrices are not retained.

Every method uses the same first N task documents and exact prompt token IDs. New runs save one `prompts.pt` containing input IDs and prompt-text hashes. Legacy runs reconstruct tokenization from their saved task/settings and check token counts; exact identity assumes the cached dataset and tokenizer are unchanged. Prompts are never silently truncated for evaluation: the saved prompt plus generation must fit the model context.

Compression happens once after prefill. All methods retain exactly the same **total number of KV elements** at a given rate; the evaluator verifies the physical cache. PyramidKV keeps its varying, nonincreasing layer allocation, with integer rounding adjusted to match that total. SnapKV and PyramidKV reconstruct window attention using the model's actual attention scale, including Granite's configured multiplier. Generated tokens subsequently grow the cache normally.

One `evaluation/comparison.json` stores shared data, seed, saved `.pt` location, curvature, and generation settings once. Each case contains only `method`, `removal_ratio`, and `accuracy` on a 0–1 scale. Scores are LongBench task metrics, such as code similarity for LCC, rather than a universal accuracy measure. Physical cache budgets are still verified during inference. No CSV or PDF copies are generated. Per-prompt metrics stored in new score files preserve complete summaries when calculation resumes.

The A40 host has 48 physical CPU cores / 96 logical CPUs. Both entry points default to 48 CPU workers, capped by active key tokens. Each worker builds one token's path blocks and solves OT locally, sharing inherited read-only inputs through `fork`. Queues carry token indices and compact results. Torch and OT use one thread per worker; the launcher also defaults OpenMP/BLAS to one thread. `WORKERS=1` / `--workers 1` uses serial execution.

A40 主机有 48 个物理 CPU 核心、96 个逻辑 CPU，默认使用 48 个 CPU worker，实际数量不超过有效 key token 数。worker 通过 `fork` 共享只读输入，在进程内计算路径和 OT；队列仅传递 token 索引和紧凑结果。Torch、OT 每个 worker 使用单线程，启动脚本中 OpenMP/BLAS 默认也为单线程。设置 `WORKERS=1` / `--workers 1` 使用串行计算。

Offline calculation moves model layers one at a time onto `DEVICE=cuda:0`; online compression uses the evaluator's inference model. Path calculation defaults to `COMPUTE_DEVICE=cpu`. CUDA path calculation, when selected, runs in the parent process; workers solve CPU OT. Curvature defaults to `float32`; use `CURVATURE_DTYPE=float64` / `--curvature-dtype float64` for higher precision. `alpha` defaults to `0.0` in both the launcher and Python CLI. `settings.json` records the value used.

## Graph and transport definition / 图和传输定义

- `layer_input` is the normalized input actually passed to the V projection. For key token `s`, incoming neighbor probabilities use **only `layer_input[0, s]`**.
- Outgoing neighbor candidates are **all features of the final query's `o_proj` output**, across all heads.
- V and O projection costs remain independent FC matrices, `C_v = 1 / abs(W_v.T)` and `C_o = 1 / abs(W_o.T)`. Zero weights have infinite cost. The current attention cost is independently `c[h,s] = 1 / abs(A[h,last_query,s])`.
- Use the **0.70 quantile**: keep positive probabilities strictly above their 70th percentile (approximately the largest 30%), then renormalize the retained neighbor mass to `1-alpha`. The existing normalization, tie, and empty-neighbor behavior is retained. The quantile is applied over all input/output candidate features, not separately per head.
- Each A coefficient connects its V-head feature set to its query-head feature set. GQA maps query head `h` to V head `h // repeat`. Center masses do not depend on V or `Att_out` activation values. Distances to/from a head's feature set are shortest distances over its member features.

入邻居使用当前 key token 的 `layer_input[0, s]`，出邻居候选为最后一个 query 的全部 O 特征。V/O 的 FC 代价与当前 A 的代价独立计算，均使用 `1 / abs(weight)`。分位点为 **0.70**，保留严格大于第 70 百分位的正概率（约前 30%），再归一化为总质量 `1-alpha`。归一化、相同值和空邻居的处理保持原样。

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
| `attention_scores` | Original last-row A values and dtype, `[query_heads, key_tokens]` |
| `curvature_scores["A"]` | One curvature per A entry, same shape |
| `curvature_ranking` | Key-token indices per head, highest curvature first |
| `metrics` | Per-prompt cost, neighbor, and curvature summaries, reused on resume |

Zero attention connections, nonfinite reciprocal costs, and unsuccessful OT solves have `NaN` curvature. They are excluded from ranking and appear as `-1` at the end of the ranking row. Equal finite curvatures preserve key-token order. Ranking is by curvature itself; raw attention values are also saved.

每个文件保存一个 layer、一个 prompt 的末行 A、对应曲率和 token 排序，形状为 `[query_heads, key_tokens]`。无有效连接或 OT 无有效解时曲率为 `NaN`，排序末尾以 `-1` 标记；相同曲率按 token 原顺序排列。按曲率本身排序，同时保留原始注意力值。

`analysis/layer_NNN_A_summary.txt` keeps separate curvature summaries for every prompt. Its overall section pools cost means, neighbor counts, and score counts only; it does not reduce prompt score tensors. `prev`, `cur`, and `next` cost means refer to V projection, A coefficients, and O projection respectively. Neighbor counts exclude center nodes. `total_edges` now counts A entries, not dense feature pairs.

## Validation / 验证

The version-2 audit compared 126 curvature cases against an independent full feature graph, Floyd–Warshall shortest paths, and a linear-program transport solver. Cases covered MHA/GQA, float32/float64, different center masses, zero attention, empty/constant/nonfinite activations, disconnected paths, and serial/multiprocessing parity. A further 100 transport cases checked feasibility and forbidden edges.

CPU model checks compared Llama, Granite, and Qwen3 native hidden states/activations and attention with the collector in float32/bfloat16, and checked that model weights stayed unchanged. End-to-end tests used the real LongBench scorer with a tiny Granite model for all 25 baseline/method/rate cases on two prompts. They verified identical prompts, measured equal cache budgets, complete/partial cache reuse, complete resumed summaries, and decoding with genuinely nonuniform PyramidKV budgets. All shell workflow modes and directory selections were checked. These are CPU correctness checks, not full-size model accuracy results; GPU validation requires a working CUDA driver. Temporary validation files are removed after use.

The shortest-path optimization was checked against the preceding implementation: 56 blocked min-plus cases and 80 sparse MHA/GQA path cases, including empty neighbors and infinite edges, were unchanged. Twelve complete curvature cases and their metrics matched exactly in serial and multiprocessing execution. Full-adjacency Floyd–Warshall checks also passed within dtype precision. A one-thread CPU microbenchmark with Granite-like dimensions (V costs `2048x512`, O costs `2048x2048`, 32 query heads, head width 64, 614 active neighbors per side; median of five runs) measured 310.58 ms before and 50.37 ms after optimization. This is a shortest-path microbenchmark, not an end-to-end speed claim. A separate allocation check reduced the largest broadcast allocation from 18.75 MiB to 8 MiB. CUDA block-selection tiers were checked with mocked free-memory values; CUDA execution was not benchmarked.
