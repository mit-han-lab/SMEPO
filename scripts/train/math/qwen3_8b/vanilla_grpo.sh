#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../common.sh"
TASK_NAME="math-qwen3-8b"
EXPERIMENT_NAME="vanilla_grpo"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the base model path}"
TRAIN_PATH="${TRAIN_PATH:-$ROOT_DIR/data/raw/math_teacher.parquet}"
USE_TEACHER_PREFIX=false
run_smepo_train "$@"
