#!/bin/bash
#
# Slurm script to run Ring Attention inference benchmark using 2 GPUs

#SBATCH --account=edu
#SBATCH --job-name=ring_benchmark
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=inference_insomnia_%j.out

echo "========================================================"
echo "Starting Ring Attention Benchmark on $(hostname) at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "========================================================"

# --- Setup Environment ---
# module load anaconda3/2023.09  # Uncomment if using modules
# source activate <your_env_name>  # Replace with your actual conda env name

WORKING_DIR="$(pwd)"
REPO_DIR="$WORKING_DIR/foundation-model-stack"

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8



# --- Run Benchmark Script ---
torchrun --nproc_per_node=2 \
  $REPO_DIR/scripts/benchmark_inference.py \
  --architecture llama \
  --variant 7b \
  --tokenizer $WORKING_DIR/llama-hf/tokenizer.model \
  --device_type cuda \
  --batch_size 1 \
  --skip_e2e_runs
