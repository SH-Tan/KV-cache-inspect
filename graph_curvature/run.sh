#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Experiment settings; environment variables override these defaults.
PYTHON_BIN="${PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
MODEL_PATH="${MODEL_PATH:-ibm-granite/granite-3.3-2b-instruct}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/llm_weights}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
COMPUTE_DEVICE="${COMPUTE_DEVICE:-cpu}"
DTYPE="${DTYPE:-auto}"
CURVATURE_DTYPE="${CURVATURE_DTYPE:-float32}"
TASK="${TASK:-longbench_lcc}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
MAX_LENGTH="${MAX_LENGTH:-}"
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-}"
ALPHA="${ALPHA:-0.0}"
# This A40 host has 48 physical cores / 96 logical CPUs.
WORKERS="${WORKERS:-48}"
SEED="${SEED:-29}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/logs/graph_curvature/$(date +%Y%m%d_%H%M%S)}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONHASHSEED="$SEED"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

flags=()
if [ "$LOCAL_FILES_ONLY" = 1 ]; then
  flags+=(--local-files-only)
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
if [ "$TRUST_REMOTE_CODE" = 1 ]; then
  flags+=(--trust-remote-code)
fi
if [ -n "$LAYER_END" ]; then
  flags+=(--layer-end "$LAYER_END")
fi
if [ -n "$MAX_LENGTH" ]; then
  flags+=(--max-length "$MAX_LENGTH")
fi

"$PYTHON_BIN" "$SCRIPT_DIR/run_experiment.py" \
  --model-path "$MODEL_PATH" --tokenizer-path "$TOKENIZER_PATH" \
  --cache-dir "$CACHE_DIR" --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" --compute-device "$COMPUTE_DEVICE" \
  --dtype "$DTYPE" --curvature-dtype "$CURVATURE_DTYPE" \
  --task "$TASK" --num-samples "$NUM_SAMPLES" \
  --layer-start "$LAYER_START" \
  --alpha "$ALPHA" \
  --workers "$WORKERS" --seed "$SEED" "${flags[@]}"
