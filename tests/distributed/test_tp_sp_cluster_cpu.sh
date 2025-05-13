#!/bin/bash
#!/bin/bash
#SBATCH --account=edu                                 # Account
#SBATCH --job-name=sp_cpu_test                        # Job name
#SBATCH --partition=cpu                               # CPU partition
#SBATCH --ntasks=1                                    # Single task
#SBATCH --cpus-per-task=160                           # Request all logical threads
#SBATCH --mem=530G                                    # Request system RAM
#SBATCH --time=00:30:00                               # Max runtime
#SBATCH --output=distributed_test_logs/sp_cpu_test_%j.out  # Output log
#SBATCH --error=distributed_test_logs/sp_cpu_test_%j.err   # Error log
#SBATCH --export=ALL                                  # Export all env vars

# Navigate to distributed test folder (relative to home)
cd "$HOME/foundation-model-stack/tests/distributed" || {
  echo "[ERROR] Could not cd into tests/distributed directory"
  exit 1
}

echo "== Running on $(hostname) =="

# Print basic CPU details
echo "==== CPU Info ===="
lscpu | grep -E 'Model name|Socket|Thread|Core'

echo ""
echo "Task $SLURM_PROCID running on $(hostname)"
echo "SLURM_LOCALID  = $SLURM_LOCALID"
echo "SLURM_PROCID   = $SLURM_PROCID"
echo "CPUs per task  = $SLURM_CPUS_PER_TASK"

# Run benchmark with both TP+SP and TP only
torchrun --nproc-per-node=2 test_tp_sp_cluster_cpu.py