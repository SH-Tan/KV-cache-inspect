#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/tans5/anaconda3/envs/vllm/bin/python}"

IBM_MODEL_DIR="${IBM_MODEL_DIR:-ibm-granite/granite-3.3-2b-instruct}"
MODEL="${MODEL:-vllm/ibm-granite-local}"
TASKS="${TASKS:-gsm8k}"
GPU="${GPU:-0}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
LIMIT="${LIMIT:-}"

export INSPECT_NO_SPAN=1
export CUDA_VISIBLE_DEVICES="$GPU"

limit_arg=""
if [ -n "$LIMIT" ]; then
  limit_arg="--limit $LIMIT"
fi

echo "Running IBM local eval"
echo "  model dir: $IBM_MODEL_DIR"
echo "  tasks: $TASKS"
echo "  gpu: $GPU"

"$PYTHON_BIN" "$SCRIPT_DIR/eval.py" \
  --models "$MODEL" \
  --tasks $TASKS \
  --model-args \
    model_path="$IBM_MODEL_DIR" \
    tokenizer_path="$IBM_MODEL_DIR" \
    dtype="$DTYPE" \
    trust_remote_code=true \
    gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    max_model_len="$MAX_MODEL_LEN" \
    max_num_seqs="$MAX_NUM_SEQS" \
  $limit_arg
