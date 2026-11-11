#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Online settings are independent of run.sh; environment variables override them.
PYTHON_BIN="${PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
MODEL_PATH="${MODEL_PATH:-ibm-granite/granite-3.3-2b-instruct}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/llm_weights}"
# Select physical GPU 0 or 1 here, or set GPU when launching.
GPU="${GPU:-0}"
DEVICE="cuda:0"  # CUDA_VISIBLE_DEVICES maps the selected GPU to this local index.
COMPUTE_DEVICE="${COMPUTE_DEVICE:-cpu}"
DTYPE="${DTYPE:-auto}"
CURVATURE_DTYPE="${CURVATURE_DTYPE:-float32}"
TASKS="${TASKS:-${TASK:-longbench_lcc}}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
MAX_LENGTH="${MAX_LENGTH:-}"
ALPHA="${ALPHA:-0.0}"
WORKERS="${WORKERS:-48}"
SEED="${SEED:-29}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
SAVE_CURVATURE="${SAVE_CURVATURE:-0}"
SCORE_SOURCE="${SCORE_SOURCE:-curvature}"
CURVATURE_REDUCTION="${CURVATURE_REDUCTION:-min}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/logs/graph_curvature/$(date +%Y%m%d_%H%M%S)_online}"
EVALUATION_OUTPUT_DIR="${EVALUATION_OUTPUT_DIR:-}"
PRESSES="${PRESSES:-curvature_min,SnapKVPress,ExpectedAttentionPress,PyramidKVPress}"
COMPRESSION_RATIOS="${COMPRESSION_RATIOS:-0.3,0.5,0.7,0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"

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
if [ -n "$MAX_LENGTH" ]; then
  flags+=(--max-length "$MAX_LENGTH")
fi
if [ -n "$MAX_NEW_TOKENS" ]; then
  flags+=(--max-new-tokens "$MAX_NEW_TOKENS")
fi

IFS=',' read -r -a task_names <<< "${TASKS// /}"
for task in "${task_names[@]}"; do
  output="$OUTPUT_DIR"
  evaluation="$EVALUATION_OUTPUT_DIR"
  if [ "${#task_names[@]}" -gt 1 ]; then
    output="$output/$task"
    if [ -n "$evaluation" ]; then
      evaluation="$evaluation/$task"
    fi
  fi
  output_flags=()
  if [ -n "$evaluation" ]; then
    output_flags+=(--evaluation-output-dir "$evaluation")
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/run_experiment.py" \
    --online-compression --model-path "$MODEL_PATH" --tokenizer-path "$TOKENIZER_PATH" \
    --cache-dir "$CACHE_DIR" --output-dir "$output" \
    --device "$DEVICE" --compute-device "$COMPUTE_DEVICE" \
    --dtype "$DTYPE" --curvature-dtype "$CURVATURE_DTYPE" \
    --task "$task" --num-samples "$NUM_SAMPLES" \
    --alpha "$ALPHA" --workers "$WORKERS" --seed "$SEED" \
    --save-curvature "$SAVE_CURVATURE" --score-source "$SCORE_SOURCE" \
    --curvature-reduction "$CURVATURE_REDUCTION" \
    --presses "$PRESSES" --compression-ratios "$COMPRESSION_RATIOS" \
    "${flags[@]}" "${output_flags[@]}"
done
