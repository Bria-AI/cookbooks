#!/bin/bash

# Activate the virtual environment
source controlnet_venv/bin/activate

# Set environment variables
export BASE_MODEL_NAME="briaai/BRIA-2.3"
export CONTROLNET_MODEL_NAME="briaai/BRIA-2.3-ControlNet-Recoloring"

export OUTPUT_DIR="XXXX"
export DATASET_NAME="XXX"

accelerate launch \
  --mixed_precision="no" \
  diffusers/examples/controlnet/train_controlnet_sdxl.py \
  --pretrained_model_name_or_path="$BASE_MODEL_NAME" \
  --controlnet_model_name_or_path="$CONTROLNET_MODEL_NAME" \
  --dataset_name="$DATASET_NAME" \
  --output_dir="$OUTPUT_DIR" \
  --conditioning_image_column="conditioning_image" \
  --image_column="image" \
  --caption_column="prompt" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --allow_tf32 \
  --use_8bit_adam \
  --learning_rate=1e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=4010 \
  --checkpointing_steps=500 \
  --seed=1337 \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention 
