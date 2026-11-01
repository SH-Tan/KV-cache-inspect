# Graph Curvature / 图曲率

Load an LLM, collect layer activations and projection weights, build the operation graph, and calculate optimal-transport curvature. This folder extracts the curvature workflow from `../LLM-pruning-with-graph-curvature`; it does not prune or save modified model weights.

加载 LLM，提取各层激活和投影权重，构建运算图并计算最优传输曲率。代码提取自 `../LLM-pruning-with-graph-curvature`，不进行剪枝或修改模型权重。

## Run / 运行

```bash
bash graph_curvature/run.sh
```

Set values in `run.sh` or override them with environment variables, as in `kv_cache_experiment`. The default model is the locally cached Granite 3.3 2B. `PYTHON_BIN` selects the environment; dependencies are PyTorch, Transformers, Accelerate, NumPy, and POT (`ot`). Optional plots in the copied utilities also require Matplotlib.

参数在 `run.sh` 中设置，也可以通过环境变量覆盖。默认使用本地缓存的 Granite 3.3 2B。通过 `PYTHON_BIN` 选择运行环境；依赖 PyTorch、Transformers、Accelerate、NumPy 和 POT (`ot`)。工具中的可选绘图功能还需要 Matplotlib。

```bash
MODEL_PATH=/path/to/model LAYER_END=1 SAMPLE_EDGE_NUM=20 \
  bash graph_curvature/run.sh

PROMPTS_FILE=/path/to/prompts.json COMPUTE_DEVICE=cuda:0 \
  bash graph_curvature/run.sh
```

`PROMPTS_FILE` is a JSON array of nonempty strings, for example `["The capital of France is", "Explain how a cache works."]`. When omitted, the runner uses `PROMPT`. Prompts are tokenized individually without padding or a chat template and truncated to `MAX_LENGTH` tokens. These are calibration inputs; no answers are generated.

`PROMPTS_FILE` 为非空字符串组成的 JSON 数组。未指定时使用 `PROMPT`。输入逐条分词，不补齐、不加聊天模板，并截断到 `MAX_LENGTH` 个 token；这些输入用于计算曲率，不生成回答。

## Settings / 参数

| Setting | Meaning |
| --- | --- |
| `MODEL_PATH`, `TOKENIZER_PATH`, `CACHE_DIR` | Model/tokenizer identifiers or local paths, and model cache directory. |
| `LOCAL_FILES_ONLY`, `TRUST_REMOTE_CODE` | Offline loading and custom model-code options; defaults are `1` and `0`. |
| `GPU`, `DEVICE` | Visible GPU and device for the current decoder layer; use `DEVICE=cpu` for CPU execution. |
| `COMPUTE_DEVICE` | Device for graph distance and shortest-path tensors; defaults to CPU. The OT solver runs on CPU. |
| `DTYPE`, `CURVATURE_DTYPE` | Model precision (`auto` by default) and graph/curvature precision (`float32` or `float64`). |
| `OPS` | Space-separated `gate_proj up_proj`, matching the source's active workflow; either can be selected alone. |
| `LAYER_START`, `LAYER_END` | Layer range `[start, end)`; empty end means all layers. Earlier layers still run to obtain correct inputs. |
| `SAMPLE_EDGE_NUM` | Cap on sampled edges per operation; default `100`, `-1` removes the cap. |
| `SAMPLE_EDGE_RATIO` | Fraction of eligible edges sampled before applying the cap; default `1.0`. |
| `SHARED_TOP_K` | Token positions considered per edge; default `10`, `-1` uses all positions. |
| `SHARED_SEQ_SELECT` | `top`, `median`, or `last`, using the source's sequence-selection metric. |
| `ALPHA` | Self-mass in the transport distribution, in `[0, 1)`; default `0`. |
| `L2_NORM` | `1` aggregates activations over positions using the source's per-example L2 mode. |
| `WORKERS` | OT worker processes; default `1` limits CPU memory and process overhead. |
| `SEED` | Model/runtime random seed. The copied edge sampler retains its source seed of `13`. |
| `OUTPUT_DIR` | Defaults to `logs/graph_curvature/<timestamp>`; must be new or empty so runs cannot mix artifacts. |

## Pipeline and files / 流程与文件

1. `model_utils.py` captures the real first-layer input using a temporary forward hook, then removes the hook.
2. `curvature_utils/layerwrapper_curv.py` collects attention/MLP activations, RoPE Q/K values, attention probabilities, and intermediate node values one layer at a time.
3. `graph_relation.py` defines operation connectivity. `curv_model_utils.py` converts weights to edge costs using `1 / abs(weight)`; zero weights have infinite cost. `curv_tensor_utils.py` and `curv_shortest_path_utils.py` construct and reuse the required graph and shortest-path matrices.
4. `cal_curvature.py` builds activation distributions and evaluates the source's OT curvature with sequence selection. For ordinary gate/up edges, the score is `(1 - W / d) / (1 - alpha)`, where `W` is the transport cost and `d` is the current edge cost. The source fixes gate beta to one; its separate `duv_beta` value is a logging diagnostic, not a score adjustment. Zero source/target activations retain the source's special score of one or two. Scores are reduced by minimum over selected positions and calibration prompts.

The source's graph abstraction and gate/up scoring rules are retained, including its minimum path cost across gate/up branches and activation-based neighbor reduction. This is an operation graph, not an exact representation of every bias, normalization, and residual operation in the model. The pruning masks, pruning pipeline, evaluation code, and calibration dataset download code are excluded. The runner supports `gate_proj`/`up_proj`; inherited helpers for other operations are internal extension code and require connectivity and shape validation before exposing another operation.

Review fixes prevent activation mutation, reject failed OT solves as undefined (`+inf`), preserve float64 analysis precision, and release shared memory on setup failures. Duplicate math helpers and unused diagnostics were removed. Successful finite gate/up calculations retain the original scoring semantics.

保留源代码的图抽象及 gate/up 评分规则，修复了原地修改激活、OT 失败结果处理、float64 统计精度和共享内存释放问题，并清理重复工具与无用诊断代码。未复制剪枝、评测或校准数据下载流程。独立运行器支持 `gate_proj`/`up_proj`；其他运算工具作为内部扩展代码保留，启用前需验证连接关系和张量形状。该图不逐项表示模型中的偏置、归一化和残差运算。

Small extraction adaptations respect the model's attention scaling, Granite residual multipliers, and Qwen3 Q/K normalization. The collector assumes a decoder with `model.layers`, rotary attention, separate Q/K/V/O projections, and a gated MLP. It uses full causal attention; sliding-window, MoE, fused-projection, and other architectures need their own collector adaptations.

提取部分支持模型定义的 attention 缩放、Granite 残差系数以及 Qwen3 的 Q/K 归一化。当前提取器假设模型使用旋转位置编码、独立 Q/K/V/O 投影和门控 MLP，并采用完整因果注意力；滑动窗口、MoE、融合投影等结构需要另行适配。

Outputs:

- `settings.json`: actual settings, resolved layer end and model dtype, calibration prompts, edge sampling seed, and token counts.
- `graph.json`: operation-level topology from the source.
- `layer_000.pt`, etc.: `layer_id` and `curvature_scores`, a mapping from operation names to CPU tensors shaped like their weights: `[out_features, in_features]`.
- `analysis/`: source calculation statistics for each operation and prompt length.

**Unsampled edges remain `+inf`; this is not a full curvature estimate unless every eligible edge is evaluated.** Zero-weight edges and undefined curvature can also remain infinite. Full coverage requires both `SAMPLE_EDGE_NUM=-1` and `SAMPLE_EDGE_RATIO=1.0` and can be expensive.

**未采样的边保留为 `+inf`，不代表已经计算了完整曲率。** 零权重边和未定义的曲率也可能为无穷大。完整覆盖需设置 `SAMPLE_EDGE_NUM=-1` 和 `SAMPLE_EDGE_RATIO=1.0`，计算量可能很大。

## Memory / 内存

The model stays on CPU except for the current decoder layer and rotary-embedding buffers. Captured inputs and activations are stored on CPU, only the required projection costs are built, and scores/caches are released after each layer is saved. CPU shortest-path calculations use chunks of at most 64, and OT defaults to one worker. Edge sampling uses flat indices before expanding coordinates; sequence-selection costs are gathered in chunks. Graph costs and score tensors are still dense; edge sampling reduces OT work but does not remove those allocations. Attention probabilities scale quadratically with prompt length, so start with short calibration inputs. The inherited worker implementation uses Linux `fork` and shared memory; runs are sequential within one Python process, not thread-safe.

模型除当前解码层外保留在 CPU。输入和激活保存到 CPU，只构建需要的投影代价；每层结果保存后释放分数与缓存。CPU 最短路径计算的分块不超过 64，OT 默认使用一个进程。图代价和分数仍为稠密矩阵；边采样减少 OT 计算量，但不会消除这些分配。注意力概率的内存随输入长度平方增长，建议先使用短输入。
