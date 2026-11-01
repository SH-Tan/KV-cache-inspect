#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# All experiment settings live here; environment variables can override them.
PYTHON_BIN="${PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
MODEL_PATH="${MODEL_PATH:-ibm-granite/granite-3.3-2b-instruct}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/llm_weights}"
GPU="${GPU:-0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
PRESS="${PRESS:-StreamingLLMPress}"
COMPRESSION_RATIOS="${COMPRESSION_RATIOS:-${COMPRESSION_RATIO:-0.1,0.2,0.5,0.8,0.9}}"
PRESS_KWARGS=${PRESS_KWARGS:-'{}'}
# LIMIT controls detailed prompt logs per task, not evaluation size.
LIMIT="${LIMIT:-20}"
TASK_COUNTS=${TASK_COUNTS:-'{
  "longbench_narrativeqa": 200,
  "longbench_qasper": 200,
  "longbench_multifieldqa_en": 150,
  "longbench_hotpotqa": 200,
  "longbench_2wikimqa": 200,
  "longbench_musique": 200,
  "longbench_gov_report": 200,
  "longbench_qmsum": 200,
  "longbench_multi_news": 200,
  "longbench_trec": 200,
  "longbench_triviaqa": 200,
  "longbench_samsum": 200,
  "longbench_passage_count": 200,
  "longbench_passage_retrieval_en": 200,
  "longbench_lcc": 200,
  "longbench_repobench-p": 500
}'}
# One task per category: SD-QA, MD-QA, Summ., FSL, Synth., Code.
TASKS="${TASKS:-longbench_qasper,longbench_hotpotqa,longbench_gov_report,longbench_trec,longbench_passage_count,longbench_lcc}"
# Other tasks available through TASKS:
# SD-QA: longbench_narrativeqa,longbench_multifieldqa_en
# MD-QA: longbench_2wikimqa,longbench_musique
# Summ.: longbench_qmsum,longbench_multi_news
# FSL: longbench_triviaqa,longbench_samsum
# Synth.: longbench_passage_retrieval_en
# Code: longbench_repobench-p
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-102400}"
SEED="${SEED:-29}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/logs/kv_cache_experiment/$PRESS/limit-$LIMIT/$(date +%Y%m%d_%H%M%S)}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE="$LOCAL_FILES_ONLY" TRANSFORMERS_OFFLINE="$LOCAL_FILES_ONLY" HF_DATASETS_OFFLINE="$LOCAL_FILES_ONLY"
flags=()
if [ "$LOCAL_FILES_ONLY" = 1 ]; then
  flags+=(--local-files-only)
fi
if [ "$TRUST_REMOTE_CODE" = 1 ]; then
  flags+=(--trust-remote-code)
fi
if [ "$DO_SAMPLE" = 1 ]; then
  flags+=(--do-sample)
fi

"$PYTHON_BIN" "$SCRIPT_DIR/run_experiment.py" \
  --model-path "$MODEL_PATH" --tokenizer-path "$TOKENIZER_PATH" \
  --cache-dir "$CACHE_DIR" --device-map "$DEVICE_MAP" --dtype "$DTYPE" \
  --attn-implementation "$ATTN_IMPLEMENTATION" \
  --press "$PRESS" --compression-ratios "$COMPRESSION_RATIOS" --press-kwargs "$PRESS_KWARGS" \
  --tasks "$TASKS" --task-counts "$TASK_COUNTS" --limit "$LIMIT" --max-context-length "$MAX_CONTEXT_LENGTH" \
  --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" \
  --seed "$SEED" --output-dir "$OUTPUT_DIR" "${flags[@]}"
