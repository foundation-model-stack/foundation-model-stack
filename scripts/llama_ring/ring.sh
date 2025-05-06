#!/usr/bin/env bash
# Ring Attention Inference submit script.
# Submits a single Slurm job (Insomnia/GPU) or runs a local process (CPU) in the background
# then tails the output.

set -eo pipefail
IFS=$'\n\t'

# --- Environment Selection ---
if [[ "$1" == "insomnia_ring" ]]; then
  RUN_LOCATION="insomnia"
  SCRIPT="ring"
  echo "[INFO] Running in Insomnia Ring environment. Will use Slurm on ring nodes for GPU execution."

elif [[ "$1" == "insomnia_default" ]]; then
  RUN_LOCATION="insomnia"
  SCRIPT="default"
  echo "[INFO] Running in Insomnia Default environment. Will use Slurm on default nodes for GPU execution."

else
  echo "[ERROR] Invalid or missing argument. Please specify 'insomnia_ring', 'insomnia_default', or 'local'."
  echo "Usage: $0 [insomnia_ring|insomnia_default|local]"
  exit 1
fi

# --- Base Paths ---
# LOCAL_REPO_DIR="/Users/sadigulcelik/Documents/CompSci/HPML-2025-Spring/FMSwrapper/foundation-model-stack"

# --- Defaults ---
DEFAULT_MODEL_REL_PATH="../llama-hf"
DEFAULT_TOKENIZER_REL_PATH="../llama-hf/tokenizer.model"

# --- Repo & Model Paths ---
INSOMNIA_BASE_DIR="$(pwd)"
CURRENT_REPO_DIR="${INSOMNIA_BASE_DIR}/foundation-model-stack"
if [[ "$SCRIPT" == "default" ]]; then
  SLURM_SCRIPT_PATH="${CURRENT_REPO_DIR}/scripts/llama_ring/run_default_inference.slurm"
else
  SLURM_SCRIPT_PATH="${CURRENT_REPO_DIR}/scripts/llama_ring/run_inference.slurm"
fi
DEFAULT_MODEL_ABS_PATH="${INSOMNIA_BASE_DIR}/llama-hf"
DEFAULT_TOKENIZER_ABS_PATH="${INSOMNIA_BASE_DIR}/llama-hf/tokenizer.model"



echo "[INFO] cd into $CURRENT_REPO_DIR"
cd "$CURRENT_REPO_DIR"

# if [[ "$RUN_LOCATION" == "insomnia" ]]; then
  # echo "[INFO] Fetching latest changes and resetting local branch..."
  # Fetch latest changes from origin
  # git fetch origin || echo "[WARN] git fetch failed"
  # Reset the current branch hard to its origin counterpart
  # git reset --hard origin/$(git rev-parse --abbrev-ref HEAD) || echo "[WARN] git reset --hard failed"
# fi

echo "[INFO] pip install -e ."
pip install -e . >/dev/null 2>&1 || echo "[WARN] pip install failed"
pip install sentencepiece

mkdir -p "${CURRENT_REPO_DIR}/testing"
cd "$HOME"

cleanup() {
  echo; echo "[INFO] Cleaning up…"
  if [[ "$RUN_LOCATION" == "insomnia" && -n "${job_id:-}" ]]; then
    scancel "$job_id" || true
  fi
  exit 130
}
trap cleanup SIGINT SIGTERM

# Build args
script_args=(); script_args+=("$@")
if ! printf '%s\n' "${script_args[@]}" | grep -q -- '--model_path'; then
  script_args+=(--model_path "$DEFAULT_MODEL_ABS_PATH")
fi
if ! printf '%s\n' "${script_args[@]}" | grep -q -- '--tokenizer'; then
  script_args+=(--tokenizer "$DEFAULT_TOKENIZER_ABS_PATH")
fi

echo "[INFO] Launching inference with args: ${script_args[*]}"

job_id=""; pid=""

OUT_FILENAME="llama_${SCRIPT}.out"
echo "[INFO] sbatch ${SLURM_SCRIPT_PATH} ${script_args[*]}"
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
