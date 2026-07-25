#!/bin/bash
#SBATCH --mem=48gb
#SBATCH -c8
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=1:00:00
#SBATCH --job-name=flux_inpaint_lora_inference
#SBATCH --output=log_inference_inpaint_job%A.txt
#SBATCH --error=error_inference_inpaint_job%A.txt
#SBATCH --killable

# Load necessary modules
source /etc/profile
module load cuda/default
module load nvidia/default
nvidia-smi
echo "Running inpaint inference on $(hostname) with $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) GPUs"

# Will use the latest checkpoint
LORA_WEIGHTS_PATH="material-coating-lora-inpaint-v4-transmissive-only"
OUTPUT_DIR="inference_inpaint_results_$(date +%Y%m%d_%H%M%S)"

ADD_SOURCE_IMAGES=(
#    "validation_dataset/sample_0/coating_0.png"
#    "validation_dataset/sample_1/coating_0.png"
#    "validation_dataset/sample_2/coating_0.png"
#    "validation_dataset/sample_3/coating_0.png"
#    "validation_dataset/sample_4/coating_0.png"
#    "validation_dataset/sample_5/coating_0.png"
#    "validation_dataset/sample_6/coating_0.png"
#    "validation_dataset/sample_7/coating_0.png"
#    "validation_dataset/sample_8/coating_0.png"
#    "validation_dataset/sample_9/coating_0.png"
#    "validation_dataset/sample_10/coating_0.png"
#    "validation_dataset/sample_11/coating_0.png"

  # Add tasks
  "validation_dataset/fire_hydrant/coating_0.png"
  "validation_dataset/fingernails/coating_0.png"
  "validation_dataset/old_car/coating_0.png"
  "validation_dataset/old_furniture/coating_0.png"
  "validation_dataset/room/coating_0.png"
  "validation_dataset/room/coating_0.png"
)

REMOVE_SOURCE_IMAGES=(
  "validation_dataset/corroded_pole/coating_0.png"
  "validation_dataset/painted_floor/coating_0.png"
  "validation_dataset/painted_wall_1/coating_0.png"
  "validation_dataset/painted_wall_2/coating_0.png"
)

MAT_FUSION_SOURCE_IMAGES=(
  "validation_dataset/material_fusion/bunny/coating_0.png"
  "validation_dataset/material_fusion/gray_car/coating_0.png"
  "validation_dataset/material_fusion/pumpkin/coating_0.png"
  "validation_dataset/material_fusion/red_car/coating_0.png"
  "validation_dataset/material_fusion/rubber_flamingo/coating_0.png"
  "validation_dataset/material_fusion/shoes/coating_0.png"
)

MAT_SWAP_SOURCE_IMAGES=(
#  "validation_dataset/mat_swap/room_1/coating_0.png"
#  "validation_dataset/mat_swap/room_1/coating_0.png"
#  "validation_dataset/mat_swap/room_2/coating_0.png"
#  "validation_dataset/mat_swap/room_3/coating_0.png"
#  "validation_dataset/mat_swap/room_3/coating_0.png"

  "validation_dataset/matswap_baseline_compare/row_1_image.png"
  "validation_dataset/matswap_baseline_compare/row_2_image.png"
  "validation_dataset/matswap_baseline_compare/row_3_image.png"
  "validation_dataset/matswap_baseline_compare/row_4_image.png"
  "validation_dataset/matswap_baseline_compare/row_5_image.png"
  "validation_dataset/matswap_baseline_compare/row_6_image.png"
  "validation_dataset/matswap_baseline_compare/row_7_image.png"
  "validation_dataset/matswap_baseline_compare/row_8_image.png"
  "validation_dataset/matswap_baseline_compare/row_9_image.png"
)

ADD_COATING_MASKS=(
#    "validation_dataset/sample_0/coating_mask.png"
#    "validation_dataset/sample_1/coating_mask.png"
#    "validation_dataset/sample_2/coating_mask.png"
#    "validation_dataset/sample_3/coating_mask.png"
#    "validation_dataset/sample_4/coating_mask.png"
#    "validation_dataset/sample_5/coating_mask.png"
#    "validation_dataset/sample_6/coating_mask.png"
#    "validation_dataset/sample_7/coating_mask.png"
#    "validation_dataset/sample_8/coating_mask.png"
#    "validation_dataset/sample_9/coating_mask.png"
#    "validation_dataset/sample_10/coating_mask.png"
#    "validation_dataset/sample_11/coating_mask.png"

  # Add tasks
  "validation_dataset/fire_hydrant/coating_mask.png"
  "validation_dataset/fingernails/coating_mask.png"
  "validation_dataset/old_car/coating_mask.png"
  "validation_dataset/old_furniture/coating_mask.png"
  "validation_dataset/room/coating_mask_1.png"
  "validation_dataset/room/coating_mask_2.png"
)

REMOVE_COATING_MASKS=(
  "validation_dataset/corroded_pole/coating_mask.png"
  "validation_dataset/painted_floor/coating_mask.png"
  "validation_dataset/painted_wall_1/coating_mask.png"
  "validation_dataset/painted_wall_2/coating_mask.png"
)

MAT_FUSION_COATING_MASKS=(
  "validation_dataset/material_fusion/bunny/coating_mask.png"
  "validation_dataset/material_fusion/gray_car/coating_mask.png"
  "validation_dataset/material_fusion/pumpkin/coating_mask.png"
  "validation_dataset/material_fusion/red_car/coating_mask.png"
  "validation_dataset/material_fusion/rubber_flamingo/coating_mask.png"
  "validation_dataset/material_fusion/shoes/coating_mask.png"
)

MAT_SWAP_COATING_MASKS=(
#  "validation_dataset/mat_swap/room_1/coating_mask_0.png"
#  "validation_dataset/mat_swap/room_1/coating_mask_1.png"
#  "validation_dataset/mat_swap/room_2/coating_mask.png"
#  "validation_dataset/mat_swap/room_3/coating_mask_0.png"
#  "validation_dataset/mat_swap/room_3/coating_mask_1.png"

  "validation_dataset/matswap_baseline_compare/row_1_mask.png"
  "validation_dataset/matswap_baseline_compare/row_2_mask.png"
  "validation_dataset/matswap_baseline_compare/row_3_mask.png"
  "validation_dataset/matswap_baseline_compare/row_4_mask.png"
  "validation_dataset/matswap_baseline_compare/row_5_mask.png"
  "validation_dataset/matswap_baseline_compare/row_6_mask.png"
  "validation_dataset/matswap_baseline_compare/row_7_mask.png"
  "validation_dataset/matswap_baseline_compare/row_8_mask.png"
  "validation_dataset/matswap_baseline_compare/row_9_mask.png"
)

MAT_SWAP_PROJECTED_ALBEDOS=(
  "validation_dataset/matswap_baseline_compare/projected_row_1_albedo.png"
  "validation_dataset/matswap_baseline_compare/projected_row_2_albedo.png"
  "validation_dataset/matswap_baseline_compare/projected_row_3_albedo.png"
  "validation_dataset/matswap_baseline_compare/projected_row_4_albedo.png"
  "validation_dataset/matswap_baseline_compare/projected_row_5_albedo.png"
  "validation_dataset/matswap_baseline_compare/projected_row_6_albedo.png"
  "validation_dataset/matswap_baseline_compare/projected_row_7_albedo.png"
  "validation_dataset/matswap_baseline_compare/projected_row_8_albedo.png"
  "validation_dataset/matswap_baseline_compare/projected_row_9_albedo.png"
)

DEFAULT_PROJECTED_ALBEDOS=(
  "None"
  "None"
  "None"
  "None"
  "None"
  "None"
)

ADD_TEXTURE_SOURCE_IMAGES=(
  "validation_dataset/sample_8/coating_0.png"
  "validation_dataset/old_furniture/coating_0.png"
  "validation_dataset/fire_hydrant/coating_0.png"
  "validation_dataset/old_car/coating_0.png"
  "validation_dataset/room/coating_0.png"
  "validation_dataset/straw_basket/coating_0.png"
)

ADD_TEXTURE_COATING_MASKS=(
  "validation_dataset/sample_8/coating_mask.png"
  "validation_dataset/old_furniture/coating_mask.png"
  "validation_dataset/fire_hydrant/coating_mask.png"
  "validation_dataset/old_car/coating_mask.png"
  "validation_dataset/room/coating_mask_3.png"
  "validation_dataset/straw_basket/coating_mask_2.png"
)

ADD_TEXTURE_PROJECTED_ALBEDOS=(
  "validation_dataset/sample_8/coating_0_albedo.png"
  "validation_dataset/old_furniture/coating_0_albedo.png"
  "validation_dataset/fire_hydrant/coating_0_albedo.png"
  "validation_dataset/old_car/coating_0_albedo.png"
  "validation_dataset/room/coating_0_albedo.png"
  "validation_dataset/straw_basket/coating_0_albedo_2.png"
)

SANITY_SOURCE_IMAGES=(
  "coating_dataset_albedos_V2/sample_703/coating_0.png"
)

SANITY_COATING_MASKS=(
  "coating_dataset_albedos_V2/sample_703/coating_mask.png"
)

SANITY_PROJECTED_ALBEDOS=(
  "None"
  #"coating_dataset_albedos_V2/sample_703/coating_16_albedo.png"
)

declare -n SOURCE_IMAGES=ADD_TEXTURE_SOURCE_IMAGES #SANITY_SOURCE_IMAGES
declare -n COATING_MASKS=ADD_TEXTURE_COATING_MASKS #MAT_SWAP_COATING_MASKS #SANITY_COATING_MASKS
declare -n PROJECTED_ALBEDOS=ADD_TEXTURE_PROJECTED_ALBEDOS #MAT_SWAP_PROJECTED_ALBEDOS #DEFAULT_PROJECTED_ALBEDOS #SANITY_PROJECTED_ALBEDOS

## Sanity check with sample_703 coating 19.
#BASE_COLORS=("0.782,0.843,0.014")
#THICKNESSES=(0.453)
#METALLICS=(0.0)
#ROUGHNESSES=(0.0)
#TRANSMISSION_WEIGHTS=(1.0)
#TASK_OPTIONS=("AU")

# Material property definitions for coating variations
# Base color (RGB), thickness, alpha, metallic, roughness, transmission_weight
#BASE_COLORS=("0.014,0.933,0.428" "0.694,0.437,0.995" "0.125,0.928,0.453" "0.47,0.82,0.075" "0.122,0.476,0.827")
#THICKNESSES=(0.1 0.5 1.0 0.0 0.0)
#METALLICS=(0.0 1.0 1.0 0.0 0.0)
#ROUGHNESSES=(0.0 0.616 0.401 0.1 0.0)
#TRANSMISSION_WEIGHTS=(1.0 0.0 0.0 0.0 1.0)
#TASK_OPTIONS=("AU" "AU" "AU")

# Changes in thickness
#BASE_COLORS=("0.9,0.8,0.1" "0.9,0.8,0.1" "0.9,0.8,0.1" "0.9,0.8,0.1")
#THICKNESSES=(0.0 0.01 0.5 1.0)
#METALLICS=(1.0 1.0 1.0 1.0)
#ROUGHNESSES=(0.0 0.0 0.0 0.0)
#TRANSMISSION_WEIGHTS=(0.0 0.0 0.0 0.0)
#TASK_OPTIONS=("AT" "AT" "AT" "AT")

## Changes in roughness
#BASE_COLORS=("0.9,0.8,0.1" "0.9,0.8,0.1" "0.9,0.8,0.1" "0.9,0.8,0.1")
#THICKNESSES=(1.0 1.0 1.0 1.0)
#METALLICS=(1.0 1.0 1.0 1.0)
#ROUGHNESSES=(0.0 0.01 0.5 1.0)
#TRANSMISSION_WEIGHTS=(0.0 0.0 0.0 0.0)
#TASK_OPTIONS=("AU" "AU" "AU" "AU")

## Changes in color
#BASE_COLORS=("0.9,0.8,0.1" "0.2,0.8,0.3" "0.3,0.4,1.0")
#THICKNESSES=(0.5 0.5 0.5)
#METALLICS=(0.0 0.0 0.0)
#ROUGHNESSES=(1.0 1.0 1.0)
#TRANSMISSION_WEIGHTS=(0.0 0.0 0.0)
#TASK_OPTIONS=("AU" "AU" "AU")

## Transmission
BASE_COLORS=("1.0,1.0,1.0" "0.2,0.2,0.2" "1.0,2.0,0.8" "0.9,0.8,0.1"  "0.2,0.8,0.9")
THICKNESSES=(0.0 0.01 0.5 1.0 0.5)
METALLICS=(0.0 0.0 0.0 0.0 0.0)
ROUGHNESSES=(0.0 0.0 0.0 0.0 0.0)
TRANSMISSION_WEIGHTS=(1.0 1.0 1.0 1.0 1.0)
TASK_OPTIONS=("AU" "AU" "AU" "AU" "AU")

## Dielectric
#BASE_COLORS=("0.0,0.0,0.0" "0.0,0.0,0.0" "0.0,0.0,0.0" "0.0,0.0,0.0")
#THICKNESSES=(1.0 1.0 0.0 0.0)
#METALLICS=(0.0 0.0 0.0 0.0)
#ROUGHNESSES=(0.0 1.0 0.0 1.0)
#TRANSMISSION_WEIGHTS=(0.0 0.0 0.0 0.0)
#TASK_OPTIONS=("AT" "AT" "AT" "AT")

# Compare with papers
#BASE_COLORS=("0.0,0.0,0.0")
#THICKNESSES=(1.0)
#METALLICS=(0.0)
#ROUGHNESSES=(1.0)
#TRANSMISSION_WEIGHTS=(0.0)
#TASK_OPTIONS=("AT")

# Apply texture
#BASE_COLORS=("0.0,0.0,0.0" "0.0,0.0,0.0" "0.0,0.0,0.0" "0.0,0.0,0.0" "0.9,0.8,0.1" "0.3,0.4,1.0")
#THICKNESSES=(0.0 0.5 0.0 0.5 0.0 0.0)
#METALLICS=(1.0 1.0 0.0 0.0 0.0 0.0)
#ROUGHNESSES=(0.0 0.3 1.0 0.0 0.0 1.0)
#TRANSMISSION_WEIGHTS=(0.0 0.0 0.0 0.0 1.0 0.0)
#TASK_OPTIONS=("AT" "AT" "AT" "AT" "AU" "AU")

# Remove task
#BASE_COLORS=("0.9,0.8,0.1" "0.0,0.0,0.0")
#THICKNESSES=(0.0 0.0)
#METALLICS=(1.0 0.0)
#ROUGHNESSES=(0.0 0.6)
#TRANSMISSION_WEIGHTS=(0.0 0.0)
#TASK_OPTIONS=("RM" "RM")

# Replace task
#BASE_COLORS=("0.9,0.8,0.1" "0.9,0.8,0.1" "0.7,0.8,1.0")
#THICKNESSES=(0.0 0.0 0.0)
#METALLICS=(1.0 1.0 0.0)
#ROUGHNESSES=(0.0 0.8 0.0)
#TRANSMISSION_WEIGHTS=(0.0 0.0 1.0)
#TASK_OPTIONS=("RL" "RL" "RL")

# Convert arrays to space-separated strings for argument passing
SOURCE_IMAGES_STR="${SOURCE_IMAGES[*]}"
COATING_MASKS_STR="${COATING_MASKS[*]}"
PROJECTED_ALBEDOS_STR="${PROJECTED_ALBEDOS[*]}"
BASE_COLORS_STR="${BASE_COLORS[*]}"
THICKNESSES_STR="${THICKNESSES[*]}"
METALLICS_STR="${METALLICS[*]}"
ROUGHNESSES_STR="${ROUGHNESSES[*]}"
TRANSMISSION_WEIGHTS_STR="${TRANSMISSION_WEIGHTS[*]}"
TASK_OPTIONS_STR="${TASK_OPTIONS[*]}"

echo "Starting coating inpaint inference with:"
echo "  Source images: ${#SOURCE_IMAGES[@]}"
echo "  Coating variations: ${#BASE_COLORS[@]}"
echo "  Total combinations: $((${#SOURCE_IMAGES[@]} * ${#BASE_COLORS[@]}))"

# Activate virtual environment
source .venv/bin/activate

# Run inference
python material_coating_training/inference_inpaint_flux_lora.py \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-Fill-dev" \
    --lora_weights_path="$LORA_WEIGHTS_PATH" \
    --source_images $SOURCE_IMAGES_STR \
    --coat_masks $COATING_MASKS_STR \
    --projected_albedos $PROJECTED_ALBEDOS_STR \
    --base_colors $BASE_COLORS_STR \
    --thicknesses $THICKNESSES_STR \
    --roughnesses $ROUGHNESSES_STR \
    --transmission_weights $TRANSMISSION_WEIGHTS_STR \
    --metallics $METALLICS_STR \
    --task_options $TASK_OPTIONS_STR \
    --output_dir="$OUTPUT_DIR" \
    --resolution=512 \
    --guidance_scale=30 \
    --num_inference_steps=50 \
    --seed=43 \
    --mixed_precision="bf16" \
    --crop

echo "Inpaint inference completed! Results saved to: $OUTPUT_DIR"
echo "Check the results grid at: $OUTPUT_DIR/results_grid.png"