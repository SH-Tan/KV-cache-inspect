#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Evaluate saved scores; no curvature calculation is performed.
PYTHON_BIN="${PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
# For curvature_both, select the saved .../curvature_min or .../curvature_mean folder.
CURVATURE_DIR="${CURVATURE_DIR:-$REPO_DIR/logs/graph_curvature/20261120_163717/longbench_lcc}"
GPU="${GPU:-0}"
# These methods reduce the saved scores across heads, independently of query aggregation.
PRESSES="${PRESSES:-curvature_min,curvature_mean,curvature_second_min,SnapKVPress,ExpectedAttentionPress,PyramidKVPress}" # SnapKVPress,ExpectedAttentionPress,PyramidKVPress
COMPRESSION_RATIOS="${COMPRESSION_RATIOS:-0.3,0.5,0.7,0.9}"
# Direct curvature methods load saved scores; other methods use native scoring.
SCORE_SOURCE="${SCORE_SOURCE:-curvature}"
CURVATURE_REDUCTION="${CURVATURE_REDUCTION:-min}"
CURVATURE_RANKING="${CURVATURE_RANKING:-layer}"  # model shares the direct curvature budget across layers.
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
  --curvature-ranking "$CURVATURE_RANKING" \
  --output-dir "$EVALUATION_OUTPUT_DIR" "${generation_flags[@]}"
