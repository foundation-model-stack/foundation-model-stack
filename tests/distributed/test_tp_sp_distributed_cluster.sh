#!/bin/bash
#SBATCH --account=edu                              # Account (REQUIRED on Insomnia)
#SBATCH --job-name=sp_test                         # Job name
#SBATCH --partition=gpu                            # Partition (queue)
#SBATCH --gres=gpu:2                               # Request 2 GPUs (generic)
#SBATCH --constraint='l40'                         # Specify GPU model (L40)
#SBATCH --ntasks=2                                 # One task per GPU
#SBATCH --cpus-per-task=24                         # CPU cores per task
#SBATCH --mem=192G                                 # Total memory per node
#SBATCH --time=00:20:00                            # Max run time
#SBATCH --output=distributed_test_logs/sp_test_%j.out  # Standard output log path
#SBATCH --error=distributed_test_logs/sp_test_%j.err   # Standard error log path
#SBATCH --export=ALL                               # Export current env vars

cd ~/tests/distributed

echo "== $(hostname) =="
echo "name, index, uuid"
nvidia-smi --query-gpu=name,index,uuid --format=csv,noheader

# Show which GPU this task is using
gpu_index=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$((SLURM_LOCALID+1)))
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $gpu_index)
gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $gpu_index)

echo "Task $SLURM_PROCID running on $(hostname), GPU: $gpu_index, $gpu_name, $gpu_uuid"

# Run the script with sequence parallelism enabled
USE_SEQUENCE_PARALLELISM=true torchrun --nproc-per-node=2 tests/distributed/test_tp_sp_distributed_cluster.py
