# KV-cache-inspect

Tools for studying language-model behavior through KV-cache compression and graph curvature.

- [KV-cache experiments](kv_cache_experiment/README.md) compare model performance with and without cache compression.
- [Graph-curvature experiments](graph_curvature/README.md) collect model activations and calculate curvature over an operation graph.

## Getting started

1. Choose an experiment and install the dependencies listed in its README.
2. Configure its `run.sh` for your Python environment, model, dataset, available hardware, and output location.
3. Make the model and dataset available through local caches or enable downloads in the launcher settings.
4. Run the launcher from the repository root.

Settings can be edited in the launcher or overridden through environment variables. Use `PYTHON_BIN` to select your Python interpreter and `MODEL_PATH` to select a model identifier or local model directory. Check the launcher for current options and defaults.

Start with a small workload, inspect the results, and increase the experiment size to suit your CPU and GPU memory. Keep input selection and evaluation settings consistent when comparing runs.
