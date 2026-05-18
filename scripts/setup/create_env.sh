#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda env create -f "$ROOT_DIR/environment.yml" || conda env update -f "$ROOT_DIR/environment.yml"
conda activate smepo

python -m pip install --upgrade pip
python -m pip install \
  torch \
  transformers \
  datasets \
  tensordict \
  omegaconf \
  hydra-core \
  wandb \
  ray \
  vllm \
  pandas \
  pyarrow \
  scipy \
  accelerate

python -m pip install ninja packaging
python -m pip install flash-attn --no-build-isolation
