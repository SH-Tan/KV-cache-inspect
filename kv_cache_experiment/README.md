# KV Cache Experiment / KV 缓存压缩实验

Evaluate the same LongBench prompts with and without KV-cache compression. Each run loads one model and compares several compression rates against its uncompressed baseline.

对同一组 LongBench 输入，分别进行不压缩和不同压缩率下的评测。每次运行只加载一个模型，并以该模型的不压缩结果作为基线。

## Run / 运行

```bash
bash kv_cache_experiment/run.sh
```

Set experiment values in `run.sh`, or override them with environment variables. This README explains the parameters; `settings.json` records the values actually used in each run. Use `PYTHON_BIN` to select an environment with `kvpress`, `transformers`, and `lm_eval[longbench]`. `LOCAL_FILES_ONLY` controls whether downloads are allowed.

实验参数在 `run.sh` 中设置，也可以通过环境变量覆盖。README 只说明参数含义，每次运行实际使用的值记录在 `settings.json` 中。通过 `PYTHON_BIN` 选择已安装 `kvpress`、`transformers` 和 `lm_eval[longbench]` 的环境。`LOCAL_FILES_ONLY` 控制是否允许下载。

## Evaluation size and logging / 评测数量与日志数量

**`TASK_COUNTS` determines how many prompts are evaluated. `LIMIT` determines how many of those prompts receive detailed logs.** Scores always use every evaluated prompt.

**`TASK_COUNTS` 决定每个任务评测多少条输入；`LIMIT` 决定其中多少条保存详细日志。** 汇总分数始终使用全部已评测输入。

Examples of supported tasks and their metrics are listed below. The enabled tasks and evaluation counts are configured in `run.sh`.

下表列出部分支持的任务及其评分指标。实际启用的任务和评测数量在 `run.sh` 中配置。

| Task / 任务 | CSV column / CSV 列名 | Category / 类别 | Metric / 指标 |
| --- | --- | --- | --- |
| `longbench_qasper` | `Qser` | Single-document QA / 单文档问答 | F1 |
| `longbench_hotpotqa` | `HpQA` | Multi-document QA / 多文档问答 | F1 |
| `longbench_gov_report` | `GR` | Summarization / 摘要 | ROUGE-L |
| `longbench_trec` | `TREC` | Classification / 分类 | Accuracy / 准确率 |
| `longbench_passage_count` | `PC` | Passage counting / 段落计数 | Count score / 计数得分 |
| `longbench_lcc` | `Lcc` | Code completion / 代码补全 | Edit similarity / 编辑相似度 |

The same evaluated prompts and logged prompts are reused across compression rates. Both subsets are selected with `SEED`. `LIMIT=0` disables detailed logs while keeping the score tables.

所有压缩率使用同一组评测输入和同一组日志输入。这两组样本都由 `SEED` 控制选择。`LIMIT=0` 关闭详细日志，但仍保存分数表。

Evaluation counts must be positive and are capped at the dataset size. When overriding `TASK_COUNTS`, include every task selected by `TASKS`; the override replaces the whole mapping.

评测数量必须大于 0，超过数据集大小时使用全部样本。覆盖 `TASK_COUNTS` 时，需要包含 `TASKS` 中的每个任务，因为覆盖的是整个映射。

Other supported tasks are listed below. Add the `longbench_` prefix when setting `TASKS`.

下表列出其他可选任务。设置 `TASKS` 时需加上 `longbench_` 前缀。

| Category / 类别 | Other tasks / 其他任务 |
| --- | --- |
| Single-document QA / 单文档问答 | `narrativeqa`, `multifieldqa_en` |
| Multi-document QA / 多文档问答 | `2wikimqa`, `musique` |
| Summarization / 摘要 | `qmsum`, `multi_news` |
| Few-shot learning / 少样本学习 | `triviaqa`, `samsum` |
| Retrieval / 检索 | `passage_retrieval_en` |
| Code completion / 代码补全 | `repobench-p` |

Prompts, reference answers, stopping rules, output-length limits, and metrics come from the installed [lm-eval LongBench tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/longbench). No extra few-shot examples or chat template are added. See the [official LongBench statistics](https://github.com/THUDM/LongBench/blob/main/LongBench/task.md) for dataset sizes.

输入模板、参考答案、停止规则、输出长度上限和评分方法均来自已安装的 [lm-eval LongBench 任务定义](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/longbench)。本实验不额外添加少样本示例或聊天模板。数据集大小见 [LongBench 官方统计](https://github.com/THUDM/LongBench/blob/main/LongBench/task.md)。

## Main settings / 主要参数

| Setting / 参数 | Purpose / 用途 |
| --- | --- |
| `MODEL_PATH`, `TOKENIZER_PATH` | Model and tokenizer / 模型与分词器 |
| `TASKS`, `TASK_COUNTS` | Tasks and evaluation counts / 任务与评测数量 |
| `LIMIT` | Detailed-log count per task / 每个任务的详细日志数量 |
| `PRESS`, `PRESS_KWARGS` | Compression method and its options / 压缩方法及其参数 |
| `COMPRESSION_RATIOS` | Fractions of prefill KV entries removed / 预填充 KV 条目的删除比例 |
| `DO_SAMPLE` | Greedy decoding or sampling / 贪心解码或随机采样 |
| `TEMPERATURE`, `TOP_P`, `TOP_K` | Sampling controls / 采样参数 |
| `SEED` | Random seed for sample selection and generation / 样本选择与生成的随机种子 |
| `MAX_CONTEXT_LENGTH` | Input + output token budget / 输入与输出的总 token 预算 |
| `OUTPUT_DIR` | Run output directory / 运行结果目录 |

Temperature, top-p, and top-k affect generation only when `DO_SAMPLE=1`. Generation uses batch size 1, one beam, and one returned answer.

只有 `DO_SAMPLE=1` 时，温度、top-p 和 top-k 才会影响采样。生成固定使用 batch size 1、单束搜索，每条输入只生成一个答案。

Rate `0` is always added as the uncompressed baseline. Other rates specify the fraction of prefill KV entries to remove; actual retained counts are rounded to integers. Duplicate rates are removed. The singular variable `COMPRESSION_RATIO` also accepts one rate.

程序自动加入压缩率 `0` 作为不压缩基线。其他压缩率表示删除预填充 KV 条目的比例，实际保留数量需要取整。重复的压缩率会被去除。单数形式的 `COMPRESSION_RATIO` 也可以用来指定一个压缩率。

## What the output contains / 输出内容

Each run writes to its own directory. Set `OUTPUT_DIR` to choose the location.

每次运行的结果保存在独立目录中，也可以通过 `OUTPUT_DIR` 指定位置。

```text
logs/kv_cache_experiment/<PRESS>/limit-<LIMIT>/<timestamp>/
  settings.json                 # All run settings / 本次运行参数
  scores.md                     # Readable comparison table / 便于阅读的对比表
  scores.csv                    # Table for analysis / 用于数据分析的表格
  longbench_qasper/
    sample_ids.json             # Evaluated document IDs / 评测样本 ID
    log_ids.json                # Detailed-log document IDs / 详细日志样本 ID
    baseline_results.json       # lm-eval aggregate results / lm-eval 汇总结果
    compression_<rate>_results.json
    prompt_<original_dataset_id>/
      baseline.json            # Full-KV statistics / 不压缩时的统计
      compression_<rate>.json  # Compressed-KV statistics / 压缩后的统计
      ...
```

### Lengths and softmax / 长度与 softmax

Each prompt JSON starts with `do_sample`, followed by temperature, top-p, top-k, seed, model, and compression method. `false` means greedy decoding; `true` means sampling. The stored sampling parameters are the configured values and are inactive during greedy decoding.

每条输入的 JSON 日志以 `do_sample` 开头，随后记录温度、top-p、top-k、随机种子、模型和压缩方法。`false` 表示贪心解码，`true` 表示随机采样。日志中的采样参数是配置值，在贪心解码时不生效。

Full generated text and the reasoning path are not written to disk. The complete response is still used for evaluation. Here, “reasoning length” means the total number of generated tokens, including the answer; it is recorded as `generated_tokens`.

日志不保存完整生成文本或 reasoning path，但评测仍使用完整生成结果。这里的“reasoning 长度”指整段生成的 token 数，包含答案，记录在 `generated_tokens` 字段中。

Compare `baseline.json` and `compression_<rate>.json` in the same prompt directory. Both contain:

对比同一输入目录中的 `baseline.json` 和 `compression_<rate>.json`，即可查看压缩前后的结果。两者都包含：

| Field / 字段 | Meaning / 含义 |
| --- | --- |
| `reference_answers` | Dataset reference answers / 数据集参考答案 |
| `scores` | This prompt's task metrics / 该条输入的任务分数 |
| `prompt_tokens` | Input tokens actually used after truncation / 截断后实际使用的输入 token 数 |
| `generated_tokens` | Total generated tokens, including the answer / 整段生成的 token 数，包含答案 |
| `kv_cache_tokens.after_prefill_per_layer` | Actual retained KV length after prefill compression, by layer / prefill 压缩后各层实际保留的 KV 长度 |
| `kv_cache_tokens.after_generation_per_layer` | Actual KV length at the final generation forward, by layer / 最后一次生成 forward 后各层实际 KV 长度 |
| `checkpoints` | Top-10 tokens and their probabilities, plus the generated token / 概率最高的 10 个 token、各自概率，以及实际生成的 token |

All lengths are token counts. Cache-length arrays are ordered by layer, starting at layer 0; they describe stored sequence positions, not bytes or a sum across heads. The same fields are recorded for the full-KV `baseline` and each compression rate. The final generated token has not yet been fed back into the model, so it is normally absent from the final cache. With ordinary dynamic caches, final KV length is the retained prefill length plus `generated_tokens - 1`.

所有长度均以 token 为单位。KV 长度数组按层排列，从第 0 层开始；它表示缓存中的序列位置数，不是字节数，也不是所有 head 的长度之和。full-KV 的 `baseline` 和各压缩率都记录相同字段。最后一个生成的 token 尚未再次输入模型，因此通常还没有对应的 KV。对于普通动态缓存，结束时的 KV 长度等于 prefill 保留长度加上 `generated_tokens - 1`。

Checkpoint probabilities are **softmax over the entire vocabulary**, using raw model logits before temperature or top-p/top-k filtering. The ten displayed probabilities are not renormalized. With sampling, the generated token may be outside the displayed top ten.

检查点的概率是对原始 logits 在**整个词表上计算 softmax** 的结果，尚未经过温度或 top-p/top-k 处理。显示的 10 个概率不会重新归一化。开启采样后，实际生成的 token 可能不在这 10 个之中。

Checkpoints are at 10%, 20%, 50%, 80%, 90%, and 100% of each answer's generated length. For `N` generated tokens, the step is `ceil(percent × N / 100)`, counting from 1. Counts include generated EOS/stop tokens. Short answers can repeat a checkpoint step. If compression changes the answer length, the same percentage can refer to different token positions.

检查点位于每个答案生成长度的 10%、20%、50%、80%、90% 和 100%。生成 `N` 个 token 时，位置为 `ceil(percent × N / 100)`，从 1 开始计数，包含已生成的 EOS 或停止 token。短答案可能有多个百分比对应同一位置。压缩前后答案长度不同时，相同百分比不一定对应相同的 token 位置。

### Scores versus compression / 分数与压缩率

Open `scores.md` for a readable table. It identifies the model, compression method, sampling mode, and seed. Each row is one compression rate; `baseline` at `0` is the uncompressed model. Each task has its own column. `scores.csv` contains the same scores without rounding and includes the model and run labels in every row.

打开 `scores.md` 可直接查看对比表。表中标明模型、压缩方法、采样模式和随机种子，每行对应一个压缩率；`0` 对应的 `baseline` 是不压缩模型，每个任务单独占一列。`scores.csv` 保存相同分数的未四舍五入数值，并在每行注明模型和运行标识。

Scores are the task's metric multiplied by 100. They are not all accuracy: QA uses F1, summarization uses ROUGE-L, and code completion uses edit similarity. `Avg.` is the arithmetic mean across the selected tasks, giving each task equal weight regardless of its sample count.

分数等于该任务的指标乘以 100，并非所有任务都使用准确率：问答使用 F1，摘要使用 ROUGE-L，代码补全使用编辑相似度。`Avg.` 是所选任务分数的算术平均值，每个任务权重相同，与样本数无关。

Tables update after each task/rate finishes. Pending cells are blank in CSV and shown as `—` in Markdown; `Avg.` appears only when the whole row is complete. To compare models, run once per `MODEL_PATH` and compare their tables using the same tasks, sample counts, seed, context budget, and generation settings.

每完成一个任务在某个压缩率下的评测，表格就会更新。未完成的单元格在 CSV 中留空，在 Markdown 中显示为 `—`；整行完成后才显示 `Avg.`。比较不同模型时，分别设置 `MODEL_PATH` 运行，并使用相同任务、样本数、随机种子、上下文预算和生成参数来对比各自的表格。

## How to interpret the experiment / 如何理解实验结果

**Compression happens only after reading the prompt.** Generated tokens add new KV entries normally, so the cache grows during decoding. The compression rate describes removal from the prompt cache, not a fixed memory budget for the whole answer. The first output token's distribution is computed before later decoding uses the compressed cache. The runner accepts presses that use the installed KVPress prefill-only hook.

**压缩只发生在读取输入的预填充阶段。** 生成过程中，新 token 的 KV 条目正常追加，因此缓存仍会增长。压缩率描述的是输入缓存被删除的比例，并不代表整个生成过程都保持固定的内存大小。第一个输出 token 的分布，在后续解码使用压缩缓存之前就已算出。本程序接受使用当前 KVPress 预填充压缩 hook 的方法。

**Long inputs may be truncated.** lm-eval reserves space for the task's maximum output length, then removes tokens from the left of inputs that exceed the remaining budget. `prompt_tokens` records the length actually used. Keep `MAX_CONTEXT_LENGTH` within the model's supported position limit.

**过长的输入可能被截断。** lm-eval 先为任务的最大输出长度预留空间，再从输入左侧删除超出剩余预算的 token。`prompt_tokens` 记录实际使用的输入长度。`MAX_CONTEXT_LENGTH` 应设在模型支持的位置长度范围内。

**The seed is reset before each answer.** Repeating a run with identical inputs, settings, hardware, and software should reproduce the result. Using the same seed does not force compressed and uncompressed models to give the same answer: compression can change token probabilities. The runner enables deterministic PyTorch operations and disables TF32 and cuDNN benchmarking.

**每次生成答案前都会重置随机种子。** 在输入、参数、硬件和软件环境一致时，重复运行应能复现结果。但相同种子并不强制压缩前后生成相同答案，因为压缩可能改变 token 概率。程序启用 PyTorch 确定性运算，并关闭 TF32 和 cuDNN 自动调优。

**Detailed logging is limited to the selected prompts.** The logger keeps the top ten token IDs and probabilities per generated step on CPU, then writes the six checkpoints. It also keeps the prefill and final cache lengths as integer lists. It does not retain full-vocabulary probability histories or KV tensors. Each generation starts with a fresh cache.

**只有选中的输入会记录详细日志。** 日志模块在 CPU 上保留每步概率最高的 10 个 token ID 及其概率，最后写出六个检查点。prefill 和生成结束时的缓存长度只保存为整数列表，不保留整个词表的概率历史或 KV 张量。每次生成都使用新的缓存。
