# Downstream Task Test

This folder contains an isolated vLLM downstream task test adapted from:

`/home/tans5/Parameters-efficient-post-training/downstream_eval`

It is not imported by the inspect eval runner. Use it to run downstream accuracy tests separately.

Default dataset:

`downstream_test/dataset/mathqa500/test.parquet`

This file was copied from:

`/home/tans5/Parameters-efficient-post-training/dataset/mathqa500/test.parquet`

Run vLLM downstream eval:

```sh
bash downstream_test/run_vllm_downstream.sh /path/to/hf_model_or_checkpoint
```

Useful overrides:

```sh
MODEL_PATH=/path/to/model \
DOWNSTREAM_TASK_DATA=downstream_test/dataset/mathqa500/test.parquet \
DOWNSTREAM_MAX_EXAMPLES=50 \
DOWNSTREAM_BATCH_SIZE=8 \
DOWNSTREAM_GENERATION_MAX_BATCH_TOKENS=65536 \
VLLM_TENSOR_PARALLEL_SIZE=1 \
VLLM_GPU_MEMORY_UTILIZATION=0.7 \
VLLM_PYTHON=/home/tans5/anaconda3/envs/vllm/bin/python \
bash downstream_test/run_vllm_downstream.sh
```

Outputs default to:

`downstream_test/results/`
