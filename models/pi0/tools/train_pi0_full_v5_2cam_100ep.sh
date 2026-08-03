#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LEROBOT_ROOT=${LEROBOT_ROOT:-/home/shukun01.huang/lerobot}
CONDA=${CONDA:-/home/shukun01.huang/miniforge3/bin/conda}
ENV_NAME=${ENV_NAME:-lerobot}
DATASET_NAME=${DATASET_NAME:-so100_rdk_camera_box_on_black_mcu_pi0_v5_2cam_100ep_10s}
DATASET_ROOT=${DATASET_ROOT:-/home/shukun01.huang/datasets/$DATASET_NAME}
BASE_MODEL=${BASE_MODEL:-/home/shukun01.huang/gemma/pi0_base}
OUTPUT_DIR=${OUTPUT_DIR:-$LEROBOT_ROOT/outputs/train/pi0_so100_rdk_camera_box_on_black_mcu_full_v5_2cam_100ep_b1_15k}
JOB_NAME=${JOB_NAME:-pi0_so100_rdk_camera_box_on_black_mcu_full_v5_2cam_100ep_b1_15k}
STEPS=${STEPS:-15000}
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_FREQ=${SAVE_FREQ:-5000}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_ENABLE=${WANDB_ENABLE:-true}
WANDB_PROJECT=${WANDB_PROJECT:-lerobot-pi0-so100}
WANDB_ENTITY=${WANDB_ENTITY:-}

WANDB_ARGS=(
  --wandb.enable=$WANDB_ENABLE
  --wandb.project=$WANDB_PROJECT
)
if [[ -n "$WANDB_ENTITY" ]]; then
  WANDB_ARGS+=(--wandb.entity=$WANDB_ENTITY)
fi

test -f $DATASET_ROOT/meta/info.json
test -f $BASE_MODEL/config.json
test ! -e $OUTPUT_DIR

cd $LEROBOT_ROOT
exec $CONDA run --no-capture-output -n $ENV_NAME python "$SCRIPT_DIR/train_pi0_memory_efficient.py" \
  --dataset.repo_id=local/$DATASET_NAME \
  --dataset.root=$DATASET_ROOT \
  --dataset.video_backend=torchcodec \
  --policy.type=pi0 \
  --policy.push_to_hub=false \
  --policy.pretrained_path=$BASE_MODEL \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_model=false \
  --policy.train_expert_only=false \
  --policy.optimizer_lr=1e-5 \
  --policy.scheduler_decay_lr=1e-6 \
  --policy.use_relative_actions=false \
  --policy.empty_cameras=0 \
  --policy.num_inference_steps=10 \
  --policy.input_features='{observation.images.base_0_rgb: {type: VISUAL, shape: [3, 224, 224]}, observation.images.left_wrist_0_rgb: {type: VISUAL, shape: [3, 224, 224]}, observation.images.right_wrist_0_rgb: {type: VISUAL, shape: [3, 224, 224]}, observation.state: {type: STATE, shape: [32]}}' \
  --rename_map='{observation.images.front: observation.images.base_0_rgb, observation.images.side: observation.images.left_wrist_0_rgb}' \
  --use_policy_training_preset=false \
  --optimizer.type=memory_efficient_adafactor \
  --optimizer.lr=1e-5 \
  --optimizer.weight_decay=0.01 \
  --optimizer.grad_clip_norm=1.0 \
  --optimizer.scale_parameter=false \
  --optimizer.relative_step=false \
  --optimizer.warmup_init=false \
  --optimizer.max_chunk_elements=8388608 \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=1000 \
  --scheduler.num_decay_steps=30000 \
  --scheduler.peak_lr=1e-5 \
  --scheduler.decay_lr=1e-6 \
  --output_dir=$OUTPUT_DIR \
  --job_name=$JOB_NAME \
  --steps=$STEPS \
  --batch_size=$BATCH_SIZE \
  --num_workers=$NUM_WORKERS \
  --save_checkpoint=true \
  --save_freq=$SAVE_FREQ \
  --log_freq=20 \
  "${WANDB_ARGS[@]}"
