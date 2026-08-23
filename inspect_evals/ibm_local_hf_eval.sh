#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

IBM_MODEL_DIR="${IBM_MODEL_DIR:-ibm-granite/granite-3.3-2b-instruct}"
MODEL="${MODEL:-hf/ibm-granite-local}"
TASKS="${TASKS:-gsm8k}"
GPU="${GPU:-0}"
DTYPE="${DTYPE:-auto}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
LIMIT="${LIMIT:-500}"

export INSPECT_NO_SPAN=1
export CUDA_VISIBLE_DEVICES="$GPU"

limit_arg=""
if [ -n "$LIMIT" ]; then
  limit_arg="--limit $LIMIT"
fi

echo "Running IBM local HF eval"
echo "  model dir: $IBM_MODEL_DIR"
echo "  tasks: $TASKS"
echo "  gpu: $GPU"

"$PYTHON_BIN" "$SCRIPT_DIR/eval.py" \
  --models "$MODEL" \
  --tasks $TASKS \
  --max-tokens "$MAX_TOKENS" \
  --model-args \
    model_path="$IBM_MODEL_DIR" \
    tokenizer_path="$IBM_MODEL_DIR" \
    dtype="$DTYPE" \
    trust_remote_code=true \
  $limit_arg
