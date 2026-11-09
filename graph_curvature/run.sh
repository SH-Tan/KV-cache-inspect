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
# RUN_CALCULATION=0 EVALUATE=1 compares an existing CURVATURE_DIR without calculation.
RUN_CALCULATION="${RUN_CALCULATION:-1}"
EVALUATE="${EVALUATE:-1}"
CURVATURE_DIR="${CURVATURE_DIR:-$OUTPUT_DIR}"
EVALUATION_OUTPUT_DIR="${EVALUATION_OUTPUT_DIR:-}"
PRESSES="${PRESSES:-curvature_min,curvature_mean,SnapKVPress,StreamingLLMPress,KnormPress,PyramidKVPress}"
COMPRESSION_RATIOS="${COMPRESSION_RATIOS:-0.3,0.5,0.7,0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"

case "$RUN_CALCULATION:$EVALUATE" in
  0:0|0:1|1:0|1:1) ;;
  *) echo "RUN_CALCULATION and EVALUATE must be 0 or 1" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
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

if [ "$RUN_CALCULATION" = 1 ]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/run_experiment.py" \
  --model-path "$MODEL_PATH" --tokenizer-path "$TOKENIZER_PATH" \
  --cache-dir "$CACHE_DIR" --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" --compute-device "$COMPUTE_DEVICE" \
  --dtype "$DTYPE" --curvature-dtype "$CURVATURE_DTYPE" \
  --task "$TASK" --num-samples "$NUM_SAMPLES" \
  --layer-start "$LAYER_START" \
  --alpha "$ALPHA" \
  --workers "$WORKERS" --seed "$SEED" "${flags[@]}"
fi

if [ "$EVALUATE" = 1 ]; then
  eval_flags=()
  if [ -n "$MAX_NEW_TOKENS" ]; then
    eval_flags+=(--max-new-tokens "$MAX_NEW_TOKENS")
  fi
  if [ -n "$EVALUATION_OUTPUT_DIR" ]; then
    eval_flags+=(--output-dir "$EVALUATION_OUTPUT_DIR")
  fi
  "$PYTHON_BIN" "$REPO_DIR/kv_cache_experiment/evaluate_curvature.py" \
    --curvature-dir "$CURVATURE_DIR" --device "$DEVICE" \
    --methods "$PRESSES" --compression-ratios "$COMPRESSION_RATIOS" "${eval_flags[@]}"
fi
