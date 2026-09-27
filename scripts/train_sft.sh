#!/usr/bin/env bash
# Supervised fine-tuning: teach the model the four-block reasoning format.
#
#     CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_sft.sh [configs/sft.sh]
#
# Reads its hyperparameters from the config file, defaulting to configs/sft.sh,
# which holds the values the released checkpoint was trained with. The effective
# batch size is held fixed across GPU counts by deriving gradient accumulation
# from it, so a run on eight GPUs takes the same optimisation steps as one on
# four.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$PWD"

CONFIG_FILE="${1:-configs/sft.sh}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "❌ config not found: $CONFIG_FILE"
  exit 1
fi
# shellcheck source=/dev/null
source "$CONFIG_FILE"

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
export CUDA_VISIBLE_DEVICES
NUM_PROCESSES=$(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .)

# Derived after the GPU count is known, so the effective batch size is what the
# config says regardless of how many GPUs are in use.
DENOMINATOR=$((NUM_PROCESSES * PER_DEVICE_BATCH_SIZE))
if (( EFFECTIVE_BATCH_SIZE % DENOMINATOR != 0 )); then
  echo "❌ EFFECTIVE_BATCH_SIZE ($EFFECTIVE_BATCH_SIZE) is not divisible by"
  echo "   GPUs x per-device batch ($NUM_PROCESSES x $PER_DEVICE_BATCH_SIZE = $DENOMINATOR)."
  exit 1
fi
GRAD_ACCUM_STEPS=$((EFFECTIVE_BATCH_SIZE / DENOMINATOR))

TRAINER_ROOT="$REPO_ROOT/third_party/qwen_vl_finetune"
ENTRY_POINT="$TRAINER_ROOT/qwenvl/train/train_qwen.py"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$TRAINER_ROOT/scripts/zero3.json}"

# The trainer resolves dataset names through this manifest instead of having
# them written into its own source. See third_party/qwen_vl_finetune/PROVENANCE.md.
export DRMV3D_SFT_REGISTRY="${DRMV3D_SFT_REGISTRY:-$REPO_ROOT/configs/sft_registry.json}"
export DRMV3D_DATA_ROOT="${DRMV3D_DATA_ROOT:-$REPO_ROOT/data}"

OUTPUT_DIR="${OUTPUT_DIR:-experiments/sft/results/$(date +%Y%m%d_%H%M%S)_${TASK_NAME}}"
mkdir -p "$OUTPUT_DIR"

# Single node, so loopback only. On a node whose peer-to-peer links work, leave
# these alone: forcing loopback costs real throughput. They default to off because
# on some single hosts enabling peer-to-peer and InfiniBand hangs the collective
# rather than falling back. Override any of them from the environment.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 10000))}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
# UCX watches /tmp with inotify, which exhausts the watch limit inside a
# container. TCP transport avoids it.
export UCX_TLS="${UCX_TLS:-self,tcp}"
export UCX_POSIX_USE_PROC_LINK="${UCX_POSIX_USE_PROC_LINK:-no}"

cat <<INFO
==============================================================================
DR-MV3D supervised fine-tuning
==============================================================================
  base model      : $BASE_MODEL
  dataset         : $TASK_NAME (from $DRMV3D_SFT_REGISTRY)
  GPUs            : $CUDA_VISIBLE_DEVICES ($NUM_PROCESSES)
  output          : $OUTPUT_DIR
  learning rate   : $LEARNING_RATE, $LR_SCHEDULER, warmup $WARMUP_RATIO
  batch           : $PER_DEVICE_BATCH_SIZE per device x $NUM_PROCESSES GPUs \
x $GRAD_ACCUM_STEPS accumulated = $EFFECTIVE_BATCH_SIZE
  epochs          : $EPOCHS
  image budget    : $MIN_PIXELS to $MAX_PIXELS pixels
  sequence length : $MODEL_MAX_LENGTH
  DeepSpeed       : $DEEPSPEED_CONFIG
==============================================================================
INFO

# The --tune_mm_* flags unfreeze the vision tower, the merger and the language
# model.

PYTHONPATH="$TRAINER_ROOT${PYTHONPATH:+:$PYTHONPATH}" torchrun \
  --nproc_per_node="$NUM_PROCESSES" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$ENTRY_POINT" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --model_name_or_path "$BASE_MODEL" \
  --dataset_use "$TASK_NAME" \
  --data_flatten True \
  --tune_mm_vision True \
  --tune_mm_mlp True \
  --tune_mm_llm True \
  --bf16 \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
  --max_pixels "$MAX_PIXELS" \
  --min_pixels "$MIN_PIXELS" \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --warmup_ratio "$WARMUP_RATIO" \
  --max_grad_norm 1 \
  --lr_scheduler_type "$LR_SCHEDULER" \
  --logging_steps 1 \
  --model_max_length "$MODEL_MAX_LENGTH" \
  --gradient_checkpointing True \
  --dataloader_num_workers "$DATALOADER_WORKERS" \
  --seed "$SEED" \
  --run_name "$(basename "$OUTPUT_DIR")" \
  --report_to none

echo "✅ Done. Checkpoints in $OUTPUT_DIR"
