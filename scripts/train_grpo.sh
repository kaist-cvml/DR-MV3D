#!/usr/bin/env bash
# Reinforcement learning with the dense reward, starting from a fine-tuned checkpoint.
#
#     CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_grpo.sh <sft-checkpoint> [train.parquet]
#
# Group Relative Policy Optimization samples a group of responses per prompt,
# scores each with drmv3d/grpo/reward.py, and uses the group's spread as the
# advantage. There is no value network, so the reward is the only critic.
#
# Defaults match the released run. Override any of them from the environment.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SFT_CHECKPOINT="${1:?usage: train_grpo.sh <sft-checkpoint> [train.parquet]}"
TRAIN_DATA="${2:-data/grpo/train.parquet}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
export CUDA_VISIBLE_DEVICES
NUM_GPUS=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .)

# --- Hyperparameters -------------------------------------------------------
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
EPOCHS="${EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"      # prompts per optimisation step
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"       # per GPU; raise if memory allows
ROLLOUT_N="${ROLLOUT_N:-8}"                     # responses sampled per prompt
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
TEMPERATURE="${TEMPERATURE:-0.7}"               # must be > 0 so the group varies
SAVE_EVERY="${SAVE_EVERY:-100}"

# veRL initialises every logger it is given, so experiment tracking is opt-in:
# set LOGGER=console,wandb once you have logged in.
LOGGER="${LOGGER:-console}"

OUTPUT_DIR="${OUTPUT_DIR:-experiments/grpo/results/$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-$(basename "$OUTPUT_DIR")}"
mkdir -p "$OUTPUT_DIR"

# --- Reward term weights ---------------------------------------------------
# Read by drmv3d/grpo/reward.py at import time.
export DRMV3D_WEIGHT_GLOBAL="${DRMV3D_WEIGHT_GLOBAL:-1.0}"
export DRMV3D_WEIGHT_LOCAL="${DRMV3D_WEIGHT_LOCAL:-1.0}"
export DRMV3D_WEIGHT_ANSWER="${DRMV3D_WEIGHT_ANSWER:-5.0}"
export DRMV3D_WEIGHT_FORMAT="${DRMV3D_WEIGHT_FORMAT:-1.0}"

# --- Distributed setup -----------------------------------------------------
# On a node whose peer-to-peer links work, leave these alone: forcing loopback
# costs real throughput. They are set here because on the machine the released run
# used, enabling peer-to-peer and InfiniBand hung the collective rather than
# falling back. Override any of them from the environment.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 10000))}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_INIT_TIMEOUT_SEC=1800
# UCX watches /tmp with inotify, which exhausts the watch limit inside a
# container. TCP transport avoids it.
export UCX_TLS=self,tcp
export UCX_POSIX_USE_PROC_LINK=no
export RAY_ADDRESS=local
export RAY_DEDUP_LOGS=1
export HYDRA_FULL_ERROR=1

# The fine-tuning trainer writes the image processor and the fast tokenizer to its
# run directory, not into each checkpoint, while the rollout engine loads them from
# the checkpoint it is given. Copy whatever is missing from the base model.
for NAME in preprocessor_config.json chat_template.json tokenizer.json; do
  if [[ ! -f "$SFT_CHECKPOINT/$NAME" ]]; then
    echo "-- $SFT_CHECKPOINT is missing $NAME; copying it from $BASE_MODEL"
    BASE_MODEL="$BASE_MODEL" python3 - "$SFT_CHECKPOINT" "$NAME" <<'PY'
import os
import shutil
import sys

from huggingface_hub import hf_hub_download

destination, name = sys.argv[1], sys.argv[2]
shutil.copy(hf_hub_download(os.environ["BASE_MODEL"], name), os.path.join(destination, name))
PY
  fi
done

if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "❌ $TRAIN_DATA not found. Build it first:"
  echo "   python scripts/prepare_grpo_data.py -i data/prompts/MindCube_train_drmv3d.jsonl -o $TRAIN_DATA"
  exit 1
fi

cat <<INFO
==============================================================================
DR-MV3D reinforcement learning
==============================================================================
  start from      : $SFT_CHECKPOINT
  training data   : $TRAIN_DATA
  GPUs            : $CUDA_VISIBLE_DEVICES ($NUM_GPUS)
  output          : $OUTPUT_DIR
  learning rate   : $LEARNING_RATE over $EPOCHS epochs
  batch           : $TRAIN_BATCH_SIZE prompts x $ROLLOUT_N samples
  reward weights  : global=$DRMV3D_WEIGHT_GLOBAL local=$DRMV3D_WEIGHT_LOCAL \
answer=$DRMV3D_WEIGHT_ANSWER format=$DRMV3D_WEIGHT_FORMAT
  logging to      : $LOGGER
==============================================================================
INFO

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$TRAIN_DATA" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=false \
  data.truncation=error \
  data.image_key=images \
  actor_rollout_ref.model.path="$SFT_CHECKPOINT" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature="$TEMPERATURE" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.free_cache_engine=true \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=true \
  trainer.total_epochs="$EPOCHS" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.project_name=drmv3d \
  trainer.experiment_name="$RUN_NAME" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_EVERY" \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  trainer.critic_warmup=0 \
  "trainer.logger=[$LOGGER]" \
  ++custom_reward_function.path="$PWD/drmv3d/grpo/reward.py" \
  ++custom_reward_function.name=compute_score

echo "✅ Done. Checkpoints in $OUTPUT_DIR"
