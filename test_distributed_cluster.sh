#!/bin/bash
#SBATCH --account=edu
#SBATCH --job-name=tp_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=rg3681_logs/tp_test_%j.out
#SBATCH --error=rg3681_logs/tp_test_%j.err
#SBATCH --export=ALL

cd ~/foundation-model-stack

echo "== $(hostname) =="
echo "name, index, uuid"
nvidia-smi --query-gpu=name,index,uuid --format=csv,noheader

# Show which GPU this task is using
gpu_index=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$((SLURM_LOCALID+1)))
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $gpu_index)
gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $gpu_index)

echo "Task $SLURM_PROCID running on $(hostname), GPU: $gpu_index, $gpu_name, $gpu_uuid"

# Run the script
USE_SEQUENCE_PARALLELISM=true torchrun --nproc-per-node=2 test_distributed_cluster.py
