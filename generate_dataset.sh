#!/bin/bash
#SBATCH --job-name=blender_parallel
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gg:g0:1
#SBATCH --time=1-0
#SBATCH --array=0-160
#SBATCH --output=logs/blender_%A_%a.out
#SBATCH --error=logs/blender_%A_%a.err
#SBATCH --killable

# Uncomment and modify this line to use specific indices instead of range-based generation
#INDICES_LIST="1084,   1150,   1213,   1328,   1400,    310,    762,    884"

# Uncomment and modify this line to use specific filenames instead of range-based generation
#NAMES_LIST="Wooden Bowl,Vintage Lighter,Kapitel Romanski,Leather Purse,Popiersie,Owl,Cat Head"
# or read from a file:
#NAMES_LIST=$(cat coating_dataset_generator/generate_list_names.csv | cut -d',' -f1 | head -n 5 | paste -sd "," -)

if [[ -n "$SLURM_ARRAY_TASK_MAX" ]] && [[ "$SLURM_ARRAY_TASK_MAX" -gt 0 ]]; then
    if [[ -n "$INDICES_LIST" ]] || [[ -n "$NAMES_LIST" ]]; then
        echo "Error: Cannot run with SLURM_ARRAY_TASK_MAX > 0 ($SLURM_ARRAY_TASK_MAX) when INDICES_LIST or NAMES_LIST is provided."
        echo "These tasks would overlap between multiple GPUs."
        exit 1
    fi
fi

if [ -n "$INDICES_LIST" ]; then
    # Count the number of indices in the list
    TOTAL_SAMPLES=$(echo "$INDICES_LIST" | tr ',' '\n' | wc -l)
    echo "Job $SLURM_ARRAY_TASK_ID: Using specific indices: $INDICES_LIST (total: $TOTAL_SAMPLES)"
elif [ -n "$NAMES_LIST" ]; then
    TOTAL_SAMPLES=$(echo "$NAMES_LIST" | tr ',' '\n' | wc -l)
    echo "Job $SLURM_ARRAY_TASK_ID: Using specific names: $NAMES_LIST (total: $TOTAL_SAMPLES)"
else
    # Default range-based generation
    # Training: 103 * 15 * 3 + 13 * 15 = 4830 (some will be skipped)
    # Benchmark: 10 * 15 * 3 = 450 (some will be skipped)
    TOTAL_SAMPLES=480
    OFFSET=0
    NUM_JOBS=$((${SLURM_ARRAY_TASK_MAX} + 1))

    # Use ceiling division to distribute jobs more evenly
    SAMPLES_PER_JOB=$(( (TOTAL_SAMPLES + NUM_JOBS - 1) / NUM_JOBS ))

    # Calculate the base index for this job (0 to 830)
    BASE_INDEX=$((SLURM_ARRAY_TASK_ID * SAMPLES_PER_JOB))

    # Add the offset to get the actual starting index (4000 to 4830)
    START_INDEX=$((BASE_INDEX + OFFSET))

    # Handle the last job in case TOTAL_SAMPLES doesn't divide evenly
    if [ $SLURM_ARRAY_TASK_ID -eq $((NUM_JOBS - 1)) ]; then
        SAMPLES_THIS_JOB=$((TOTAL_SAMPLES - BASE_INDEX))
    else
        SAMPLES_THIS_JOB=$SAMPLES_PER_JOB
    fi

    echo "Job $SLURM_ARRAY_TASK_ID: Processing samples $START_INDEX to $((START_INDEX + SAMPLES_THIS_JOB - 1))"
fi

echo "Running on node: $SLURMD_NODENAME"
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "GPU assigned: $CUDA_VISIBLE_DEVICES ($GPU_TYPE)"

# Load necessary modules. Use profile for this.
source /etc/profile

# Use default because it seems not all nodes support the same module versions.
module load cuda/default # cuda/12.4.1
module load nvidia/default #nvidia/550.144.03
nvidia-smi
echo "Running on $(hostname) with $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) GPUs"

source .venv/bin/activate

mkdir -p logs

if [ -n "$INDICES_LIST" ]; then
    python -m coating_dataset_generator.dataset_generation_batch_manager 0 1 --indices_list "$INDICES_LIST"
elif [ -n "$NAMES_LIST" ]; then
    python -m coating_dataset_generator.dataset_generation_batch_manager 0 1 --names_list "$NAMES_LIST"
else
    python -m coating_dataset_generator.dataset_generation_batch_manager $START_INDEX $SAMPLES_THIS_JOB
fi

echo "Job $SLURM_ARRAY_TASK_ID completed successfully"