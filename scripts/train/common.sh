#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SMEPO_CONDA_ENV="${SMEPO_CONDA_ENV:-smepo}"

activate_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available in PATH" >&2
    exit 1
  fi
  eval "$(conda shell.bash hook)"
  conda activate "$SMEPO_CONDA_ENV"
}

run_smepo_train() {
  : "${TASK_NAME:?TASK_NAME is required}"
  : "${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
  : "${MODEL_PATH:?MODEL_PATH is required}"
  : "${TRAIN_PATH:?TRAIN_PATH is required}"
  activate_env

  export NCCL_IBEXT_DISABLE="${NCCL_IBEXT_DISABLE:-1}"
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
  export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
  export UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1}"
  export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
  export PYTHONNOUSERSITE=1
  export NNODES="${NNODES:-1}"
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
  export RAY_LOGGING_LEVEL="${RAY_LOGGING_LEVEL:-DEBUG}"
  export HYDRA_FULL_ERROR=1
  export WANDB_DIR="${WANDB_DIR:-$ROOT_DIR}"

  local train_batch_size="${TRAIN_BATCH_SIZE:-32}"
  local ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-8}"
  local ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE:-8}"
  local n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
  local max_prompt_length="${MAX_PROMPT_LENGTH:-4096}"
  local max_response_length="${MAX_RESPONSE_LENGTH:-12288}"
  local save_freq="${SAVE_FREQ:-10}"
  local total_epochs="${TOTAL_EPOCHS:-1000000}"
  local total_training_steps="${TOTAL_TRAINING_STEPS:-500}"
  local gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.8}"
  local use_teacher_prefix="${USE_TEACHER_PREFIX:-false}"
  local teacher_key="${TEACHER_KEY:-ds}"
  local prefix_ratio="${PREFIX_RATIO:-1.0}"
  local prefix_intro="${PREFIX_INTRO:-Here is an example solution:\n}"
  local prefix_tail="${PREFIX_TAIL:-<|im_end|>\n<|im_start|>assistant}"
  local trainer_logger="${TRAINER_LOGGER:-[\"console\"]}"

  local project_name="smepo-${TASK_NAME}"
  local ckpt_root="${CKPT_ROOT:-$ROOT_DIR/outputs/checkpoints}"
  local rollout_root="${ROLLOUT_ROOT:-$ROOT_DIR/outputs/rollouts}"
  local ckpt_dir="${ckpt_root}/${TASK_NAME}/${EXPERIMENT_NAME}"
  local rollout_dir="${rollout_root}/${TASK_NAME}/${EXPERIMENT_NAME}"

  export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
  cd "$ROOT_DIR"

  python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$TRAIN_PATH']" \
    data.val_files="[]" \
    data.shuffle=true \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.train_batch_size="${train_batch_size}" \
    data.val_batch_size=1 \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    algorithm.use_kl_in_reward=false \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.0 \
    actor_rollout_ref.actor.clip_ratio_high=5.0 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 2)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 3)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 3)) \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ppo_micro_batch_size}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_checkpointing=true \
    trainer.logger="${trainer_logger}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=false \
    trainer.test_freq=0 \
    trainer.save_freq="${save_freq}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.total_training_steps="${total_training_steps}" \
    trainer.default_local_dir="${ckpt_dir}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=0 \
    +trainer.use_teacher_prefix="${use_teacher_prefix}" \
    +trainer.teacher_key="${teacher_key}" \
    +trainer.prefix_ratio="${prefix_ratio}" \
    +trainer.prefix_intro="${prefix_intro}" \
    +trainer.prefix_tail="${prefix_tail}" \
    trainer.rollout_data_dir="${rollout_dir}" \
    +trainer.checkpoint.save_contents='["hf_model"]' \
    "$@"
}
