#!/usr/bin/env bash
# Ring Attention Inference submit script.
# Submits a Slurm job (Insomnia/GPU) or runs locally (CPU), then tails the output.

set -eo pipefail
IFS=$'\n\t'

# --- Script Selection ---
if [[ "$1" == "ring" ]]; then
  SCRIPT="ring"
  echo "[INFO] Running in Insomnia Ring environment. Will use Slurm on ring nodes for GPU execution."

elif [[ "$1" == "inference" ]]; then
  SCRIPT="inference"
  echo "[INFO] Running in Insomnia Default environment. Will use Slurm on default nodes for GPU execution."

else
  echo "[ERROR] Invalid or missing argument. Please specify 'ring', 'inference'"
  echo "Usage: $0 [ring|inference]"
  exit 1
fi


# --- Base Paths ---
CURR_DIR=$(pwd)
LOCAL_REPO_DIR="${CURR_DIR}/foundation-model-stack"
DEFAULT_MODEL_REL_PATH="${CURR_DIR}/llama-hf"
DEFAULT_TOKENIZER_REL_PATH="${CURR_DIR}/llama-hf/tokenizer.model"




if [[ "$SCRIPT" == "ring" ]]; then
  SLURM_SCRIPT_PATH="${LOCAL_REPO_DIR}/scripts/llama_ring/benchmark_ring.slurm"
else
  SLURM_SCRIPT_PATH="${LOCAL_REPO_DIR}/scripts/llama_ring/benchmark_inference.slurm"
fi


echo "[INFO] cd into $LOCAL_REPO_DIR"
cd "$LOCAL_REPO_DIR"

# if [[ "$RUN_LOCATION" == "insomnia" ]]; then
#   echo "[INFO] Fetching latest changes..."
#   git fetch origin || echo "[WARN] git fetch failed"
#   git reset --hard origin/$(git rev-parse --abbrev-ref HEAD) || echo "[WARN] git reset failed"
# fi

echo "[INFO] pip install -e ."
pip install -e . >/dev/null 2>&1 || echo "[WARN] pip install failed"

# --- Cleanup on exit ---
cleanup() {
  echo; echo "[INFO] Cleaning up…"
  if [[ "$RUN_LOCATION" == "insomnia" && -n "${job_id:-}" ]]; then
    scancel "$job_id" || true
  elif [[ "$RUN_LOCATION" == "local" && -n "${pid:-}" ]]; then
    kill "$pid" 2>/dev/null || true
  fi
  exit 130
}
trap cleanup SIGINT SIGTERM

# Build args
script_args=("@");
if ! printf '%s\n' "${script_args[@]}" | grep -q -- '--model_path'; then
  script_args+=(--model_path "$DEFAULT_MODEL_REL_PATH")
fi
if ! printf '%s\n' "${script_args[@]}" | grep -q -- '--tokenizer'; then
  script_args+=(--tokenizer "$DEFAULT_TOKENIZER_REL_PATH")
fi

echo "[INFO] Launching inference with args: ${script_args[*]}"

job_id=""; pid=""
cd "$CURR_DIR"
OUT_FILENAME="bench.out"
echo "[INFO] sbatch --output="$OUT_FILENAME" "$SLURM_SCRIPT_PATH" "${script_args[@]}" 2>&1"
sbatch_out=$(sbatch --output="$OUT_FILENAME" "$SLURM_SCRIPT_PATH" "${script_args[@]}" 2>&1) || {
  echo "[ERROR] sbatch failed: $sbatch_out"; exit 1
}
job_id=$(echo "$sbatch_out" | grep -oP 'Submitted batch job \K[0-9]+')
# Match the updated Slurm script's #SBATCH --output pattern

echo "[SUCCESS] Slurm job $job_id"
wait_cmd="squeue -u $USER"


echo "[INFO] Monitor with: $wait_cmd"
# echo "[INFO] Will tail: $output_file"

# # wait & tail
# for i in {1..12}; do
#   if [[ -s "$output_file" ]]; then
#     echo; echo "[INFO] Tailing…"; tail -n +1 -f "$output_file"; exit 0
#   fi
#   printf '.'
#   sleep 5
# done

# echo; echo "[ERROR] No output in $output_file"
# cleanup
