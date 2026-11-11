#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Evaluate the saved Qasper run; no curvature calculation is performed.
PYTHON_BIN="${PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
CURVATURE_DIR="${CURVATURE_DIR:-$REPO_DIR/logs/graph_curvature/20261109_150356/longbench_qasper}"
GPU="${GPU:-0}"
PRESSES="${PRESSES:-curvature_min,curvature_mean,SnapKVPress,ExpectedAttentionPress,PyramidKVPress}"
COMPRESSION_RATIOS="${COMPRESSION_RATIOS:-0.3,0.5,0.7,0.9}"
# Curvature min/mean still load saved scores; other methods use native scoring.
SCORE_SOURCE="${SCORE_SOURCE:-native}"
CURVATURE_REDUCTION="${CURVATURE_REDUCTION:-min}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
EVALUATION_OUTPUT_DIR="${EVALUATION_OUTPUT_DIR:-$CURVATURE_DIR/evaluation/$(date +%Y%m%d_%H%M%S)}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONHASHSEED=29
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

generation_flags=()
if [ -n "$MAX_NEW_TOKENS" ]; then
  generation_flags+=(--max-new-tokens "$MAX_NEW_TOKENS")
fi

# Model, task, prompts, dtype, and seed are loaded from the saved run.
exec "$PYTHON_BIN" "$REPO_DIR/kv_cache_experiment/evaluate_curvature.py" \
  --curvature-dir "$CURVATURE_DIR" --device cuda:0 \
  --methods "$PRESSES" --compression-ratios "$COMPRESSION_RATIOS" \
  --score-source "$SCORE_SOURCE" --curvature-reduction "$CURVATURE_REDUCTION" \
  --output-dir "$EVALUATION_OUTPUT_DIR" "${generation_flags[@]}"
