#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/tans5/anaconda3/envs/vllm/bin/python}"

# ── models to sweep ───────────────────────────────────────────────────────────
MODELS="${MODELS:-vllm/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"

# ── tasks to sweep ────────────────────────────────────────────────────────────
# Available: pubmedqa medqa aime2024 gsm8k math500 tab_fact legalbench finben
#            livecodebench codeforces polyglot amc23 humaneval bigcodebench mbpp usaco
TASKS="${TASKS:-aime2024}"

export INSPECT_NO_SPAN=1

# ── run every model across all tasks ─────────────────────────────────────────
for MODEL in $MODELS; do
  echo "Running eval  |  model: $MODEL  |  tasks: $TASKS"
  "$PYTHON_BIN" "$SCRIPT_DIR/eval.py" --models "$MODEL" --tasks $TASKS
done
