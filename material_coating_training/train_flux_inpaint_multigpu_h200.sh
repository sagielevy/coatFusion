#!/bin/bash
#SBATCH --mem=380gb
#SBATCH -c16
#SBATCH --gres=gpu:h200:2
#SBATCH --time=2-0
#SBATCH --job-name=mat_coat_flux_inpaint_lora
#SBATCH --output=logs/log_job%A.txt
#SBATCH --error=logs/error_job%A.txt
#SBATCH --killable
#SBATCH --requeue
# #SBATCH --dependency=afterok:30488228

# --mem= should be ~50G per GPU, so --mem=400gb for 8 l40s..
# Load necessary modules. Use profile for this.
source /etc/profile

# Use default because it seems not all nodes support the same module versions.
module load cuda/default # cuda/12.4.1
module load nvidia/default #nvidia/550.144.03
nvidia-smi
echo "Running on $(hostname) with $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) GPUs"

# Activate virtual environment
source .venv/bin/activate

## PRE-TRAINING DATA MOVE
#SRC="material-coating-lora-inpaint-triptych-256-v8/checkpoint-30000"
#DEST_DIR="material-coating-lora-inpaint-triptych-512-v8"
#
#if [ -d "$SRC" ]; then
#    echo "Moving checkpoint-30000 to new output directory..."
#    mkdir -p "$DEST_DIR"
#    mv "$SRC" "$DEST_DIR/"
#else
#    echo "Warning: Source checkpoint not found at $SRC"
#fi

# Batch size depends on dataset size, variance between images, etc. Try 32-64 batch size and 5e-4 to 1e-4.
# LoRa Rank range: 8-128
# --main_process_port=0 assumes we use a single node.
# Note: for res=1024 use batch=2 MAX.
accelerate launch --config_file=~/.cache/huggingface/accelerate/default_config_h200.yaml --main_process_port=0 material_coating_training/train_dreambooth_inpaint_lora_flux.py \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-Fill-dev" \
    --jsonl_for_train="material_coating_training/train_dataset_config.jsonl" \
    --output_dir="material-coating-lora-inpaint-triptych-512-v8" \
    --tracker_project_name="material-coating-lora-inpaint-triptych-512" \
    --mixed_precision="bf16" \
    --train_batch_size=2 \
    --rank=128 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1e-4 \
    --report_to="wandb" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=300 \
    --max_train_steps=35000 \
    --dataloader_num_workers=4 \
    --normal_drop=0.8 \
    --seed=0 \
    --resolution=512 \
    --resume_from_checkpoint="latest" \
    --enable_data_augmentation \
    --use_best_uv_mapping \
    --filter_replace_task \
    --validation_steps=2000 \
    --checkpointing_steps=1000