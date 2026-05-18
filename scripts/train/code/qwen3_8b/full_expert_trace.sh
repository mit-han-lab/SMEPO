#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../common.sh"
TASK_NAME="code-qwen3-8b"
EXPERIMENT_NAME="full_expert_trace"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the base model path}"
TRAIN_PATH="${TRAIN_PATH:-$ROOT_DIR/data/raw/code_teacher.parquet}"
USE_TEACHER_PREFIX=true
PREFIX_RATIO="${PREFIX_RATIO:-1.0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
run_smepo_train "$@"
