#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Experiment settings; environment variables override these defaults.
PYTHON_BIN="${PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
MODEL_PATH="${MODEL_PATH:-ibm-granite/granite-3.3-2b-instruct}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/llm_weights}"
# Select physical GPU 0 or 1 here, or set GPU when launching.
GPU="${GPU:-1}"
DEVICE="cuda:0"  # CUDA_VISIBLE_DEVICES maps the selected GPU to this local index.
COMPUTE_DEVICE="${COMPUTE_DEVICE:-cpu}"
DTYPE="${DTYPE:-auto}"
CURVATURE_DTYPE="${CURVATURE_DTYPE:-float32}"
# TASK still selects one dataset; TASKS selects a comma-separated list.
TASKS="${TASKS:-${TASK:-longbench_lcc,longbench_qasper,longbench_trec,longbench_multi_news}}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
MAX_LENGTH="${MAX_LENGTH:-}"
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-}"
ALPHA="${ALPHA:-0.5}"
# This A40 host has 48 physical cores / 96 logical CPUs.
WORKERS="${WORKERS:-48}"
SEED="${SEED:-29}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
SAVE_CURVATURE="${SAVE_CURVATURE:-1}"
# Use curvature importance in SnapKV, PyramidKV, and ExpectedAttention by default.
SCORE_SOURCE="${SCORE_SOURCE:-curvature}"
CURVATURE_REDUCTION="${CURVATURE_REDUCTION:-min}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/logs/graph_curvature/$(date +%Y%m%d_%H%M%S)}"
# Calculate and save fresh scores; set RUN_CALCULATION=0 and CURVATURE_DIR to reuse a run.
RUN_CALCULATION="${RUN_CALCULATION:-1}"
EVALUATE="${EVALUATE:-0}"
if [ "$RUN_CALCULATION" = 0 ]; then
  CURVATURE_DIR="${CURVATURE_DIR:-$REPO_DIR/logs/graph_curvature/20261108_131908}"
else
  CURVATURE_DIR="${CURVATURE_DIR:-$OUTPUT_DIR}"
fi
EVALUATION_OUTPUT_DIR="${EVALUATION_OUTPUT_DIR:-}"
PRESSES="${PRESSES:-curvature_min,curvature_mean,SnapKVPress,ExpectedAttentionPress,PyramidKVPress}"
COMPRESSION_RATIOS="${COMPRESSION_RATIOS:-0.3,0.5,0.7,0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"

case "$RUN_CALCULATION:$EVALUATE" in
  0:0|0:1|1:0|1:1) ;;
  *) echo "RUN_CALCULATION and EVALUATE must be 0 or 1" >&2; exit 2 ;;
esac
case "$SAVE_CURVATURE" in
  0|1) ;;
  *) echo "SAVE_CURVATURE must be 0 or 1" >&2; exit 2 ;;
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

generation_flags=()
if [ -n "$MAX_NEW_TOKENS" ]; then
  generation_flags+=(--max-new-tokens "$MAX_NEW_TOKENS")
fi
IFS=',' read -r -a task_names <<< "${TASKS// /}"

run_evaluation() {
  local curvature_dir="$1" output="$2"
  local output_flags=()
  if [ -n "$output" ]; then
    output_flags+=(--output-dir "$output")
  fi
  "$PYTHON_BIN" "$REPO_DIR/kv_cache_experiment/evaluate_curvature.py" \
    --curvature-dir "$curvature_dir" --device "$DEVICE" \
    --methods "$PRESSES" --compression-ratios "$COMPRESSION_RATIOS" \
    --score-source "$SCORE_SOURCE" --curvature-reduction "$CURVATURE_REDUCTION" \
    "${generation_flags[@]}" "${output_flags[@]}"
}

if [ "$RUN_CALCULATION:$EVALUATE:$SAVE_CURVATURE" = 1:1:0 ]; then
  echo "Offline calculation followed by evaluation requires SAVE_CURVATURE=1" >&2
  exit 2
fi

if [ "$RUN_CALCULATION" = 0 ] && [ -f "$CURVATURE_DIR/settings.json" ]; then
  if [ "$EVALUATE" = 1 ]; then
    run_evaluation "$CURVATURE_DIR" "$EVALUATION_OUTPUT_DIR"
  fi
  exit 0
fi

for task in "${task_names[@]}"; do
  output="$OUTPUT_DIR"
  curvature="$CURVATURE_DIR"
  evaluation="$EVALUATION_OUTPUT_DIR"
  if [ "${#task_names[@]}" -gt 1 ]; then
    output="$output/$task"
    curvature="$curvature/$task"
    if [ -n "$evaluation" ]; then
      evaluation="$evaluation/$task"
    fi
  fi
  if [ "$RUN_CALCULATION" = 1 ]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/run_experiment.py" \
      --model-path "$MODEL_PATH" --tokenizer-path "$TOKENIZER_PATH" \
      --cache-dir "$CACHE_DIR" --output-dir "$output" \
      --device "$DEVICE" --compute-device "$COMPUTE_DEVICE" \
      --dtype "$DTYPE" --curvature-dtype "$CURVATURE_DTYPE" \
      --task "$task" --num-samples "$NUM_SAMPLES" --layer-start "$LAYER_START" \
      --alpha "$ALPHA" --workers "$WORKERS" --seed "$SEED" \
      --save-curvature "$SAVE_CURVATURE" --score-source "$SCORE_SOURCE" \
      --curvature-reduction "$CURVATURE_REDUCTION" "${flags[@]}"
  fi
  if [ "$EVALUATE" = 1 ]; then
    run_evaluation "$curvature" "$evaluation"
  fi
done
