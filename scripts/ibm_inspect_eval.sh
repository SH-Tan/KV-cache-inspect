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
TASKS="${TASKS:-gsm8k}"
GPU="${GPU:-0}"
DTYPE="${DTYPE:-auto}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/llm_weights}"
BACKEND="${BACKEND:-vllm}"
MODEL_NAME="${MODEL_NAME:-ibm-granite-local}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TASK_NUM="${TASK_NUM:-${LIMIT:-}}"
BATCH_SIZE="${BATCH_SIZE:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
THINK_MODE="${THINK_MODE:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

limit_arg=""
if [ -n "$TASK_NUM" ]; then
  limit_arg="--limit $TASK_NUM"
fi

local_files_arg=""
if [ "$LOCAL_FILES_ONLY" = "1" ]; then
  local_files_arg="--local-files-only"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

batch_size_arg=""
if [ -n "$BATCH_SIZE" ]; then
  batch_size_arg="--batch-size $BATCH_SIZE"
fi

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

echo "Running IBM inspect eval"
echo "  model path: $IBM_MODEL_DIR"
echo "  backend: $BACKEND"
echo "  tasks: $TASKS"
echo "  gpu: $GPU"
echo "  cache dir: $CACHE_DIR"
echo "  task num: ${TASK_NUM:-all}"
echo "  batch size: ${BATCH_SIZE:-inspect/vllm default}"
echo "  gpu memory utilization: $GPU_MEMORY_UTILIZATION"
echo "  think mode: $THINK_MODE"
echo "  cuda home: ${CUDA_HOME:-unset}"

export XDG_DATA_HOME="${XDG_DATA_HOME:-/tmp/inspect_ai_data}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CACHE_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_DIR}"

"$PYTHON_BIN" "$REPO_DIR/run_ibm_inspect_eval.py" \
  --model-path "$IBM_MODEL_DIR" \
  --backend "$BACKEND" \
  --model-name "$MODEL_NAME" \
  --tasks $TASKS \
  --dtype "$DTYPE" \
  --cache-dir "$CACHE_DIR" \
  --max-tokens "$MAX_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --gpu "$GPU" \
  $batch_size_arg \
  $think_mode_arg \
  $local_files_arg \
  $limit_arg
