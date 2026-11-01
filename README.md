# KV-cache-inspect

Two standalone experiment folders:

- [kv_cache_experiment](kv_cache_experiment/README.md): existing KV-cache compression experiments.
- [graph_curvature](graph_curvature/README.md): LLM activation extraction, graph construction, and curvature calculation copied from `../LLM-pruning-with-graph-curvature`.

Each experiment has its own `run.sh`, `run_experiment.py`, and README. Downloaded models stay in `llm_weights/`; run outputs go to `logs/`.

The KV experiment folder is unchanged. `src/utils/cuda_memory_utils.py` and its package initializer remain because the existing KV runner imports that helper. Legacy evaluation code and scripts have been removed.
