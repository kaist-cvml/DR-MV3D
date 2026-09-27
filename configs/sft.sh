# Supervised fine-tuning configuration for the released model.
#
# These are the values the released checkpoint was trained with, read back from
# the checkpoint itself. Changing any of them changes what you get.

TASK_NAME="drmv3d"

BASE_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

LEARNING_RATE=1e-5
LR_SCHEDULER="cosine"
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.0
EPOCHS=3

# Effective batch size, held fixed across GPU counts: the launcher derives
# gradient accumulation from this, the GPU count, and the per-device batch size.
EFFECTIVE_BATCH_SIZE=128
PER_DEVICE_BATCH_SIZE=1

# Image-token budget during training. Inference uses a larger budget; see
# INFERENCE_MAX_PIXELS in drmv3d/constants.py.
MAX_PIXELS=90000
MIN_PIXELS=784

MODEL_MAX_LENGTH=8192

SAVE_STEPS=10
SAVE_TOTAL_LIMIT=6

DATALOADER_WORKERS=4
SEED=42
