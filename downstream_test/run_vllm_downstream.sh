#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

model_path="${MODEL_PATH:-${1:-meta-llama/Meta-Llama-3-8B}}"
dataset_path="${DOWNSTREAM_TASK_DATA:-downstream_test/dataset/mathqa500/test.parquet}"
prompt_key="${DOWNSTREAM_PROMPT_KEY:-prompt}"
response_key="${DOWNSTREAM_RESPONSE_KEY:-}"
reward_score_dir="${DOWNSTREAM_REWARD_SCORE_DIR:-}"
output_dir="${DOWNSTREAM_OUTPUT_DIR:-downstream_test/results}"

vllm_python="${VLLM_PYTHON:-}"
if [ -z "$vllm_python" ]; then
    if [ -x "/home/tans5/anaconda3/envs/vllm/bin/python" ]; then
        vllm_python="/home/tans5/anaconda3/envs/vllm/bin/python"
    else
        vllm_python="python"
    fi
fi

max_examples="${DOWNSTREAM_MAX_EXAMPLES:-500}"
start_index="${DOWNSTREAM_START_INDEX:-0}"
shuffle="${DOWNSTREAM_SHUFFLE:-0}"
batch_size="${DOWNSTREAM_BATCH_SIZE:-64}"
generation_max_batch_tokens="${DOWNSTREAM_GENERATION_MAX_BATCH_TOKENS:-491520}"
max_prompt_length="${DOWNSTREAM_MAX_PROMPT_LENGTH:-2048}"
max_new_tokens="${DOWNSTREAM_MAX_NEW_TOKENS:-8192}"
temperature="${DOWNSTREAM_TEMPERATURE:-0.0}"
top_p="${DOWNSTREAM_TOP_P:-1.0}"
top_k="${DOWNSTREAM_TOP_K:-0}"
response_log_max="${DOWNSTREAM_RESPONSE_LOG_MAX:-0}"
tensor_parallel_size="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.7}"
vllm_dtype="${VLLM_DTYPE:-auto}"

mkdir -p "$output_dir"
safe_model_name="$(basename "$model_path" | tr '/:' '__')"
output_path="$output_dir/${safe_model_name}_responses.jsonl"
metrics_path="$output_dir/${safe_model_name}_metrics.json"

cmd=(
    "$vllm_python" -m downstream_test.vllm_accuracy_runner
    --model_path "$model_path"
    --dataset_path "$dataset_path"
    --output_path "$output_path"
    --metrics_path "$metrics_path"
    --prompt_key "$prompt_key"
    --start_index "$start_index"
    --max_examples "$max_examples"
    --batch_size "$batch_size"
    --generation_max_batch_tokens "$generation_max_batch_tokens"
    --max_prompt_length "$max_prompt_length"
    --max_new_tokens "$max_new_tokens"
    --temperature "$temperature"
    --top_p "$top_p"
    --top_k "$top_k"
    --response_log_max "$response_log_max"
    --tensor_parallel_size "$tensor_parallel_size"
    --gpu_memory_utilization "$gpu_memory_utilization"
    --dtype "$vllm_dtype"
)

if [ -n "$response_key" ]; then
    cmd+=(--response_key "$response_key")
fi
if [ -n "$reward_score_dir" ]; then
    cmd+=(--reward_score_dir "$reward_score_dir")
fi
if [ "$shuffle" = "1" ]; then
    cmd+=(--shuffle)
fi

echo "Running downstream vLLM eval"
echo "model_path=$model_path"
echo "dataset_path=$dataset_path"
echo "output_path=$output_path"
echo "metrics_path=$metrics_path"

"${cmd[@]}"
