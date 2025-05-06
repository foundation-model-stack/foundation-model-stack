#!/bin/bash
#
# Slurm script to run Ring Attention inference using 2 GPUs

#SBATCH --account=edu
#SBATCH --job-name=ring_infer
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --time=00:30:00

echo "========================================================"
echo "Starting Ring Attention Inference Job on $(hostname) at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "========================================================"

# --- Setup Environment ---
# module load anaconda3/2023.09
# source activate your_env_name  # Replace with your actual conda env
# Define repo path for clarity
WORKING_DIR="$(pwd)"
REPO_DIR="$WORKING_DIR/foundation-model-stack"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8


# --- Run Inference ---
torchrun --nproc_per_node=2 \
  "${REPO_DIR}/scripts/inference.py" \
  --architecture llama \
  --variant 7b \
  --model_path $WORKING_DIR/llama-hf \
  --model_source hf \
  --tokenizer $WORKING_DIR/llama-hf/tokenizer.model \
  --device_type cuda \
  --default_dtype fp16 \
  --no_use_cache \
