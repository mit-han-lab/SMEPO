# SMEPO

Official codebase for `Hide to Guide: Learning via Semantic Masking`.

Paper: https://arxiv.org/abs/2605.25198

Data: https://huggingface.co/datasets/mit-han-lab/SMEPO

Semantic Masked Expert Policy Optimization (SMEPO) is an expert-guided RLVR method that masks reward-relevant semantic spans in expert traces while preserving their procedural structure. This repository provides code for data preparation, semantic masking, and training with masked or full expert-trace guidance.

## Overview

SMEPO supports three training settings used in the paper:

- `vanilla GRPO`: standard RLVR training
- `full expert trace`: RL conditioned on the full expert trace
- `SMEPO`: RL conditioned on semantically masked expert traces

The prefix length is directly controllable through `PREFIX_RATIO`, while the sentence/newline-aligned prefix truncation behavior used in our experiments is preserved in the training code.

## Repository Structure

- `scripts/data`: dataset construction scripts
- `scripts/masking`: semantic masking components used by SMEPO
- `scripts/train`: training entry points for all experiment settings
- `scripts/setup`: environment setup scripts
- `verl`: training stack used by SMEPO

## Environment

The conda environment name is `smepo`.

```bash
conda env create -f environment.yml
conda activate smepo
```

You can also use the helper script:

```bash
bash scripts/setup/create_env.sh
```

The training environment includes `flash-attn`:

```bash
pip install flash-attn --no-build-isolation
```

## Data

SMEPO uses task-specific expert traces stored in `teacher_ds`.

Released raw datasets use the same schema in all domains:

- `question`
- `reward_model`
- `teacher_ds`

To construct the masked datasets from raw expert traces:

```bash
bash scripts/data/build_data.sh
```

The released raw datasets are hosted at `mit-han-lab/SMEPO`. After downloading `math.parquet`, the standard data build script can be used directly:

```bash
python scripts/data/download_from_hf.py \
  --repo mit-han-lab/SMEPO \
  --filename math.parquet \
  --out-parquet data/raw/math_teacher.parquet
```

Then build the masked dataset:

```bash
bash scripts/data/build_data.sh
```

## Training

The repository includes training scripts for the main experiment settings in the paper. Below is an example reproduction flow for the math task with Qwen3-8B:

```bash
conda activate smepo
MODEL_PATH=/path/to/Qwen3-8B-Base \
bash scripts/train/math/qwen3_8b/smepo.sh
```

Related scripts for the same setting:

- `scripts/train/math/qwen3_8b/vanilla_grpo.sh`
- `scripts/train/math/qwen3_8b/full_expert_trace.sh`
- `scripts/train/math/qwen3_8b/smepo.sh`

Additional scripts for other tasks and model settings are provided in `scripts/train/`.

## Prefix Control

SMEPO exposes the expert-prefix controls directly from the shell:

```bash
MODEL_PATH=/path/to/model \
PREFIX_RATIO=0.5 \
bash scripts/train/math/qwen3_8b/smepo.sh
```

Useful knobs:

- `PREFIX_RATIO`: fraction of the expert trace used as the prefix
- `PREFIX_INTRO`: prompt string inserted before the prefix
- `PREFIX_TAIL`: prompt string inserted after the prefix
- `TRAINER_LOGGER`: logging backend selection
