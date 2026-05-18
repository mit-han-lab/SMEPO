#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../common.sh"
TASK_NAME="math-qwen3-8b"
EXPERIMENT_NAME="smepo"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the base model path}"
TRAIN_PATH="${TRAIN_PATH:-$ROOT_DIR/data/masked/math_teacher_masked.parquet}"
USE_TEACHER_PREFIX=true
PREFIX_RATIO="${PREFIX_RATIO:-1.0}"
run_smepo_train "$@"
