# Per-prompt last-row A curvature / 逐 prompt 的末行 A 曲率

Compute one curvature for each `A[head, last_query, key_token]` along `layer_input -> v_proj -> Att_out -> o_proj`. Every prompt and every key token keeps its own score. Key tokens are ranked within each query head by **descending curvature**. Scores are never minimized across tokens or prompts.

沿 `layer_input -> v_proj -> Att_out -> o_proj` 路径，为每个 `A[head, last_query, key_token]` 计算一个曲率。每个 prompt、每个 key token 独立保留分数，每个 query head 内按曲率从高到低排序，不对 token 或 prompt 取最小值。

## Run / 运行

Use Python with PyTorch, Transformers, Accelerate, NumPy, POT (`ot`), and `lm_eval[longbench]`:

```bash
PYTHON_BIN=python MODEL_PATH="<model-id-or-path>" bash graph_curvature/run.sh
```

`run.sh` accepts environment overrides; `run_experiment.py --help` lists CLI options. Use an empty output directory for each run. Prompts are **not truncated by default**. Set `MAX_LENGTH` / `--max-length` only when an explicit token cap is wanted. Every positive last-row attention entry is evaluated; there is no feature-edge or token sampling.

`run.sh` 支持环境变量覆盖，CLI 参数见 `run_experiment.py --help`。每次运行使用空输出目录。默认使用完整 prompt；仅在需要截断时设置 `MAX_LENGTH` / `--max-length`。所有正值末行注意力项都会计算，不再进行特征边或 token 采样。

The A40 host has 48 physical CPU cores / 96 logical CPUs. Both entry points default to 48 CPU workers, capped by active key tokens. Each worker builds one token's path blocks and solves OT locally, sharing inherited read-only inputs through `fork`. Queues carry token indices and compact results. Torch and OT use one thread per worker; the launcher also defaults OpenMP/BLAS to one thread. `WORKERS=1` / `--workers 1` uses serial execution.

A40 主机有 48 个物理 CPU 核心、96 个逻辑 CPU，默认使用 48 个 CPU worker，实际数量不超过有效 key token 数。worker 通过 `fork` 共享只读输入，在进程内计算路径和 OT；队列仅传递 token 索引和紧凑结果。Torch、OT 每个 worker 使用单线程，启动脚本中 OpenMP/BLAS 默认也为单线程。设置 `WORKERS=1` / `--workers 1` 使用串行计算。

Model layers run one at a time on `DEVICE=cuda:0`; path calculation defaults to `COMPUTE_DEVICE=cpu`. CUDA path calculation, when selected, runs in the parent process; workers solve CPU OT. Curvature defaults to `float32`; use `CURVATURE_DTYPE=float64` / `--curvature-dtype float64` for higher precision. `alpha` defaults to `0.0` in both the launcher and Python CLI. `settings.json` records the value used.

## Graph and transport definition / 图和传输定义

- `layer_input` is the normalized input actually passed to the V projection. For key token `s`, incoming neighbor probabilities use **only `layer_input[0, s]`**.
- Outgoing neighbor candidates are **all features of the final query's `o_proj` output**, across all heads.
- V and O projection costs remain independent FC matrices, `C_v = 1 / abs(W_v.T)` and `C_o = 1 / abs(W_o.T)`. Zero weights have infinite cost. The current attention cost is independently `c[h,s] = 1 / abs(A[h,last_query,s])`.
- Use the **0.70 quantile**: keep positive probabilities strictly above their 70th percentile (approximately the largest 30%), then renormalize the retained neighbor mass to `1-alpha`. The existing normalization, tie, and empty-neighbor behavior is retained. The quantile is applied over all input/output candidate features, not separately per head.
- Each A coefficient connects its V-head feature set to its query-head feature set. GQA maps query head `h` to V head `h // repeat`. Center masses do not depend on V or `Att_out` activation values. Distances to/from a head's feature set are shortest distances over its member features.

入邻居使用当前 key token 的 `layer_input[0, s]`，出邻居候选为最后一个 query 的全部 O 特征。V/O 的 FC 代价与当前 A 的代价独立计算，均使用 `1 / abs(weight)`。分位点为 **0.70**，保留严格大于第 70 百分位的正概率（约前 30%），再归一化为总质量 `1-alpha`。归一化、相同值和空邻居的处理保持原样。

For each token `s`, shortest paths preserve the actual feature graph: an attention edge connects matching feature positions within the appropriate heads. Construct `D_XO(s)` from the full `C_v -> c[:,s] -> C_o` adjacency. Take minima over complete feature paths; do not independently minimize the V and O matrices before composing them, which could join different feature positions incorrectly.

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

当 `alpha=0` 且两侧邻居均非空时，OT 只使用共享的 `D_XO(s)`，因此每个 key token 仅求解一次 OT，再用各 head 自己的 A 代价得到独立曲率。

## Outputs / 输出

`settings.json` records the run configuration, prompt token lengths, quantile, graph definition, and descending ranking order. Scores are written immediately to:

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

Zero attention connections, nonfinite reciprocal costs, and unsuccessful OT solves have `NaN` curvature. They are excluded from ranking and appear as `-1` at the end of the ranking row. Equal finite curvatures preserve key-token order. Ranking is by curvature itself; raw attention values are also saved.

每个文件保存一个 layer、一个 prompt 的末行 A、对应曲率和 token 排序，形状为 `[query_heads, key_tokens]`。无有效连接或 OT 无有效解时曲率为 `NaN`，排序末尾以 `-1` 标记；相同曲率按 token 原顺序排列。按曲率本身排序，同时保留原始注意力值。

`analysis/layer_NNN_A_summary.txt` keeps separate curvature summaries for every prompt. Its overall section pools cost means, neighbor counts, and score counts only; it does not reduce prompt score tensors. `prev`, `cur`, and `next` cost means refer to V projection, A coefficients, and O projection respectively. Neighbor counts exclude center nodes. `total_edges` now counts A entries, not dense feature pairs.

## Validation / 验证

Validation compared scores with an independent full feature graph using Floyd–Warshall shortest paths and OT, checked MHA/GQA and both curvature dtypes, compared serial and multiprocessing results, and verified transport reuse and ranking. Offline two-layer Llama and Granite runs covered different prompt lengths, including more than 128 tokens. Granite checks also verified float32/bfloat16 model execution, unchanged model weights, and bit-for-bit preservation of saved A values and dtype. Temporary validation code and outputs were removed after use.
