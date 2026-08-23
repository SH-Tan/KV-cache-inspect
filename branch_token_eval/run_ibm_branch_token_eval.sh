#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/tans5/anaconda3/envs/vllm/bin/python"
if [ -z "${PYTHON_BIN:-}" ] && [ -x "$DEFAULT_PYTHON" ]; then
  PYTHON_BIN="$DEFAULT_PYTHON"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

IBM_MODEL_DIR="${IBM_MODEL_DIR:-ibm-granite/granite-3.3-2b-instruct}"
GPU="${GPU:-0}"
DTYPE="${DTYPE:-auto}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/llm_weights}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/logs/branch_token_eval/run_$RUN_TIMESTAMP}"
OUTPUT_BASE="$OUTPUT_DIR"
OUTPUT_SUFFIX=1
while [ -e "$OUTPUT_DIR" ]; do
  OUTPUT_DIR="${OUTPUT_BASE}_$OUTPUT_SUFFIX"
  OUTPUT_SUFFIX=$((OUTPUT_SUFFIX + 1))
done
TASK_NUM="${TASK_NUM:-50}"
START="${START:-0}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
TOP_LOGPROBS="${TOP_LOGPROBS:-10}"
BRANCH_TOP_K="${BRANCH_TOP_K:-5}"
FEWSHOT="${FEWSHOT:-10}"
THINK_MODE="${THINK_MODE:-1}"

think_mode_arg=""
if [ "$THINK_MODE" = "1" ]; then
  think_mode_arg="--think-mode"
fi

if [ -z "${CUDA_HOME:-}" ]; then
  CUDA_CANDIDATE="$("$PYTHON_BIN" -c 'import pathlib, sys; p=pathlib.Path(sys.prefix) / "lib" / ("python%d.%d" % sys.version_info[:2]) / "site-packages" / "nvidia" / "cu13"; print(p if (p / "bin" / "nvcc").exists() else "")')"
  if [ -n "$CUDA_CANDIDATE" ]; then
    export CUDA_HOME="$CUDA_CANDIDATE"
    export PATH="$CUDA_HOME/bin:$PATH"
  fi
fi

if [ -n "${CUDA_HOME:-}" ]; then
  if [ ! -d "$CUDA_HOME/lib64" ] && [ -d "$CUDA_HOME/lib" ]; then
    REAL_CUDA_HOME="$CUDA_HOME"
    CUDA_WRAPPER_HOME="/tmp/vllm_cuda_home_${USER:-$(id -u)}"
    mkdir -p "$CUDA_WRAPPER_HOME/lib64/stubs"
    ln -sfn "$REAL_CUDA_HOME/bin" "$CUDA_WRAPPER_HOME/bin"
    ln -sfn "$REAL_CUDA_HOME/include" "$CUDA_WRAPPER_HOME/include"
    ln -sfn "$REAL_CUDA_HOME/lib" "$CUDA_WRAPPER_HOME/lib"
    for lib in "$REAL_CUDA_HOME"/lib/*; do
      ln -sfn "$lib" "$CUDA_WRAPPER_HOME/lib64/$(basename "$lib")"
    done
    if [ ! -e "$CUDA_WRAPPER_HOME/lib64/libcudart.so" ] && [ -e "$REAL_CUDA_HOME/lib/libcudart.so.13" ]; then
      ln -sfn "$REAL_CUDA_HOME/lib/libcudart.so.13" "$CUDA_WRAPPER_HOME/lib64/libcudart.so"
    fi
    if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
      ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so "$CUDA_WRAPPER_HOME/lib64/libcuda.so"
    fi
    export CUDA_HOME="$CUDA_WRAPPER_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
  fi
  if [ -d "$CUDA_HOME/lib" ]; then
    export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  fi
  if [ -d "$CUDA_HOME/lib64" ]; then
    export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  fi
fi
if [ -d /usr/lib/x86_64-linux-gnu ]; then
  export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

export XDG_DATA_HOME="${XDG_DATA_HOME:-/tmp/inspect_ai_data}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_DIR}"

echo "Running IBM branch token eval"
echo "  model path: $IBM_MODEL_DIR"
echo "  gpu: $GPU"
echo "  cache dir: $CACHE_DIR"
echo "  run timestamp: $RUN_TIMESTAMP"
echo "  output dir: $OUTPUT_DIR"
echo "  examples: $TASK_NUM"
echo "  max tokens: $MAX_TOKENS"
echo "  gpu memory utilization: $GPU_MEMORY_UTILIZATION"
echo "  think mode: $THINK_MODE"
echo "  cuda home: ${CUDA_HOME:-unset}"

"$PYTHON_BIN" "$SCRIPT_DIR/run_branch_token_eval.py" \
  --model-path "$IBM_MODEL_DIR" \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --limit "$TASK_NUM" \
  --start "$START" \
  --gpu "$GPU" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-tokens "$MAX_TOKENS" \
  --top-logprobs "$TOP_LOGPROBS" \
  --branch-top-k "$BRANCH_TOP_K" \
  --fewshot "$FEWSHOT" \
  $think_mode_arg
