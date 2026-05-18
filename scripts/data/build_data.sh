#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RAW_DIR="${RAW_DIR:-$ROOT_DIR/data/raw}"
MASKED_DIR="${MASKED_DIR:-$ROOT_DIR/data/masked}"

MATH_RAW_PARQUET="${MATH_RAW_PARQUET:-$RAW_DIR/math_teacher.parquet}"
CODE_RAW_PARQUET="${CODE_RAW_PARQUET:-$RAW_DIR/code_teacher.parquet}"

mkdir -p "$RAW_DIR" "$MASKED_DIR"

maybe_download() {
  local out_parquet="$1"
  local repo="${2:-}"
  local subset="${3:-}"
  local split="${4:-train}"
  local filename="${5:-}"

  if [[ -f "$out_parquet" ]]; then
    return 0
  fi

  if [[ -z "$repo" ]]; then
    return 0
  fi

  python "$SCRIPT_DIR/download_from_hf.py" \
    --repo "$repo" \
    --split "$split" \
    ${subset:+--subset "$subset"} \
    ${filename:+--filename "$filename"} \
    --out-parquet "$out_parquet"
}

maybe_download "$MATH_RAW_PARQUET" "${SMEPO_MATH_HF_REPO:-mit-han-lab/SMEPO}" "${SMEPO_MATH_HF_SUBSET:-}" "${SMEPO_MATH_HF_SPLIT:-train}" "${SMEPO_MATH_HF_FILENAME:-math.parquet}"
maybe_download "$CODE_RAW_PARQUET" "${SMEPO_CODE_HF_REPO:-mit-han-lab/SMEPO}" "${SMEPO_CODE_HF_SUBSET:-}" "${SMEPO_CODE_HF_SPLIT:-train}" "${SMEPO_CODE_HF_FILENAME:-code.parquet}"

if [[ ! -f "$MATH_RAW_PARQUET" ]]; then
  echo "Missing math raw parquet: $MATH_RAW_PARQUET" >&2
  echo "Place the raw file there or use the default mit-han-lab/SMEPO source." >&2
  exit 1
fi

if [[ ! -f "$CODE_RAW_PARQUET" ]]; then
  echo "Missing code raw parquet: $CODE_RAW_PARQUET" >&2
  echo "Place the raw file there or use the default mit-han-lab/SMEPO source." >&2
  exit 1
fi

python "$SCRIPT_DIR/prepare_math_teacher_ds.py" \
  --in-file "$MATH_RAW_PARQUET" \
  --out-parquet "$MASKED_DIR/math_teacher_masked.parquet" \
  --mask-teacher-ds

python "$SCRIPT_DIR/prepare_code_teacher_ds.py" \
  --in-file "$CODE_RAW_PARQUET" \
  --out-parquet "$MASKED_DIR/code_teacher_masked.parquet" \
  --mask-teacher-ds
