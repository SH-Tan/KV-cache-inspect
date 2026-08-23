#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="$PROJECT_DIR/inspect_evals/src/inspect_evals"
TEMP_DIR="$(mktemp -d)"
TASKS=(
    agieval
    gsm8k
    ifeval
    math500
    mgsm
    mmlu
    onet
    race_h
    sevenllm
    truthfulqa
    winogrande
)
COMMON_PATHS=(
    constants.py
    metadata.py
    utils
)

trap 'rm -rf "$TEMP_DIR"' EXIT

git clone \
    --depth 1 \
    --filter=blob:none \
    --sparse \
    https://github.com/UKGovernmentBEIS/inspect_evals.git \
    "$TEMP_DIR/repo"

cd "$TEMP_DIR/repo"
git sparse-checkout set \
    "${TASKS[@]/#/src/inspect_evals/}" \
    "${COMMON_PATHS[@]/#/src/inspect_evals/}"

mkdir -p "$TARGET_DIR"

for path in "${COMMON_PATHS[@]}"; do
    if [ -e "$TEMP_DIR/repo/src/inspect_evals/$path" ]; then
        rm -rf "$TARGET_DIR/$path"
        cp -a \
            "$TEMP_DIR/repo/src/inspect_evals/$path" \
            "$TARGET_DIR/$path"
        echo "Updated: $path"
    else
        echo "Common path not found upstream: $path" >&2
    fi
done

for task in "${TASKS[@]}"; do
    if [ -d "$TEMP_DIR/repo/src/inspect_evals/$task" ]; then
        rm -rf "$TARGET_DIR/$task"
        cp -a \
            "$TEMP_DIR/repo/src/inspect_evals/$task" \
            "$TARGET_DIR/$task"
        echo "Updated: $task"
    else
        echo "Task not found upstream: $task" >&2
    fi
done
