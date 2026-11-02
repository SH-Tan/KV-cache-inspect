# Graph Curvature / 图曲率

Collect language-model activations and projection weights, construct an operation graph, and calculate optimal-transport curvature. The workflow processes layers incrementally to limit memory use.

提取语言模型的激活和投影权重，构建运算图，并计算最优传输曲率。流程逐层处理，以控制内存占用。

## Setup and run / 配置与运行

Use a Python environment with PyTorch, Transformers, Accelerate, NumPy, POT (`ot`), and `lm_eval[longbench]`. Optional plotting utilities use Matplotlib. The curvature workers require a platform that supports multiprocessing with `fork` and shared memory.

使用安装了 PyTorch、Transformers、Accelerate、NumPy、POT（`ot`）和 `lm_eval[longbench]` 的 Python 环境。可选绘图功能使用 Matplotlib。曲率计算进程需要支持 `fork` 和共享内存的平台。

Configure `run.sh` with your model, LongBench task, sample selection, layer range, computation devices, and output location. Review the cache and download settings for your environment. Then run from the repository root:

在 `run.sh` 中配置模型、LongBench 任务、样本选择、层范围、计算设备和输出位置，并根据环境设置缓存与下载选项。然后在仓库根目录运行：

```bash
PYTHON_BIN=python MODEL_PATH="<model-id-or-path>" bash graph_curvature/run.sh
```

Replace the model placeholder with your chosen model identifier or directory. Other settings can also be overridden through environment variables. Use the launcher and the Python entry point's `--help` for current options. Choose a new or empty output directory for each run.

将模型占位符替换为所需的模型标识或目录。其他参数也可以通过环境变量覆盖。当前选项见启动脚本和 Python 入口的 `--help`。每次运行使用新的或空的输出目录。

## Workflow and results / 流程与结果

The pipeline loads the model and dataset, captures layer inputs, collects the values required by the graph, and calculates curvature. Results include run settings, graph structure, layer scores, and analysis summaries.

流程依次加载模型与数据集、捕获层输入、提取图计算所需数值，并计算曲率。结果包括运行参数、图结构、逐层分数和分析汇总。

Input truncation and edge sampling control the workload. Sampled runs do not provide full edge coverage; infinite scores can indicate unsampled edges or undefined curvature. Start with a small workload before increasing context length, sample count, or layer coverage.

输入截断和边采样用于控制计算量。采样运行不覆盖全部边；无穷分数可能表示未采样的边或未定义的曲率。建议先验证小规模运行，再增加上下文长度、样本数或层覆盖范围。

## Extending the pipeline / 扩展流程

Start with `main()` in `run_experiment.py` to follow the loading, collection, and calculation stages. Reusable helpers handle activation collection, graph costs, and curvature calculations. Keep collection and calculation organized by layer to avoid retaining all activations at once.

从 `run_experiment.py` 的 `main()` 阅读加载、提取和计算流程。可复用工具负责激活提取、图代价和曲率计算。扩展时保持逐层提取与计算，避免同时保存所有激活。

The current runner calculates gate/up projection curvature for compatible decoder models with full causal attention and gated MLPs. Other model architectures or graph operations may need collector and graph adaptations.

当前程序针对兼容的解码器模型计算 gate/up 投影曲率，要求完整因果注意力和门控 MLP。其他模型结构或图运算可能需要调整提取器和图定义。
