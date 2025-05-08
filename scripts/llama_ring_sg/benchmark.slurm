#!/bin/bash
#
# Slurm script to run Ring Attention inference benchmark using 2 GPUs

#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --job-name=ring_benchmark
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=70G
#SBATCH --time=00:10:00
#SBATCH --output=inference_insomnia_%j.out

echo "========================================================"
echo "Starting Ring Attention Benchmark on $(hostname) at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "========================================================"

# --- Setup Environment ---
# module load anaconda3/2023.09  # Uncomment if using modules
# source activate <your_env_name>  # Replace with your actual conda env name

export PYTHONPATH="/insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack:$PYTHONPATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

REPO_DIR="/insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack"

# --- Run Benchmark Script ---
torchrun --nproc_per_node=2 \
  "${REPO_DIR}/scripts/llama_ring_sg/benchmark_ring.py" \
  --architecture llama \
  --variant 7b \
  --model_path /insomnia001/depts/edu/COMSE6998/sg3790/llama-hf \
  --tokenizer /insomnia001/depts/edu/COMSE6998/sg3790/llama-hf/tokenizer.model \
  --device_type cuda \
  --num_tokens_to_benchmark 15 \
  --batch_size 1 \
  --run_ring_first
