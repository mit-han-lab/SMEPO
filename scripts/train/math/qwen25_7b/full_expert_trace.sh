#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../common.sh"
TASK_NAME="math-qwen25-7b"
EXPERIMENT_NAME="full_expert_trace"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the base model path}"
TRAIN_PATH="${TRAIN_PATH:-$ROOT_DIR/data/raw/math_teacher.parquet}"
USE_TEACHER_PREFIX=true
PREFIX_RATIO="${PREFIX_RATIO:-1.0}"
run_smepo_train "$@"
