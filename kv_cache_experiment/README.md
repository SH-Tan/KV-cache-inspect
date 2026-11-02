# KV Cache Experiment / KV 缓存压缩实验

Compare language-model performance on LongBench with and without KV-cache compression. Each run evaluates an uncompressed baseline and the selected compression settings on the same inputs.

在 LongBench 上比较语言模型使用和不使用 KV 缓存压缩时的表现。每次运行使用相同输入，评测不压缩基线及所选压缩配置。

## Setup and run / 配置与运行

Use a Python environment with PyTorch, Transformers, Accelerate, `kvpress`, and `lm_eval[longbench]`. Choose a model and compression method supported by the installed libraries and available hardware.

使用安装了 PyTorch、Transformers、Accelerate、`kvpress` 和 `lm_eval[longbench]` 的 Python 环境。选择已安装库和可用硬件支持的模型与压缩方法。

Configure `run.sh` with your model, tasks, evaluation size, compression settings, generation options, and output location. Review the cache and download settings for your environment. Then run from the repository root:

在 `run.sh` 中配置模型、任务、评测规模、压缩参数、生成选项和输出位置，并根据环境设置缓存与下载选项。然后在仓库根目录运行：

```bash
PYTHON_BIN=python MODEL_PATH="<model-id-or-path>" bash kv_cache_experiment/run.sh
```

Replace the model placeholder with your chosen model identifier or directory. Other settings can also be overridden through environment variables. Use the launcher and the Python entry point's `--help` for current options. Use a separate output directory for each run.

将模型占位符替换为所需的模型标识或目录。其他参数也可以通过环境变量覆盖。当前选项见启动脚本和 Python 入口的 `--help`。每次运行使用独立的输出目录。

## Evaluation and results / 评测与结果

Task definitions provide the prompts, reference answers, and evaluation metrics. `TASK_COUNTS` controls the number of evaluated examples for each selected task; `LIMIT` controls how many examples receive detailed logs. Logging fewer examples does not reduce the evaluation set.

任务定义提供输入模板、参考答案和评测指标。`TASK_COUNTS` 控制各个所选任务的评测样本数，`LIMIT` 控制保存详细日志的样本数。减少日志数量不会缩小评测集合。

Results include run settings, score comparisons, task-level results, and optional sample-level diagnostics. Use the comparison tables to assess quality across compression settings and the detailed logs to inspect generation and cache behavior.

结果包括运行参数、分数对比、任务级结果和可选的样本级诊断。通过对比表评估不同压缩配置下的质量，通过详细日志查看生成与缓存行为。

## Comparing runs / 对比实验

Keep tasks, sample selection, context limits, generation settings, and random seeds consistent when comparing models or compression methods. Task metrics may differ, so interpret scores according to each task.

比较模型或压缩方法时，应保持任务、样本选择、上下文限制、生成参数和随机种子一致。不同任务可能使用不同指标，应结合各任务的指标理解分数。

Compression is applied to the prompt cache during prefill. The cache continues to grow as new tokens are generated, so a compression ratio is not a fixed memory budget for the entire answer. Long prompts may be truncated to fit the configured context limit. Start with a small evaluation before scaling up.

压缩作用于预填充阶段的输入缓存。随着新 token 生成，缓存会继续增长，因此压缩率不代表整个回答过程中的固定内存预算。过长输入可能按上下文限制截断。建议先验证小规模评测，再扩大实验规模。
