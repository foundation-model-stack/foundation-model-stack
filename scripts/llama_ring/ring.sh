#!/usr/bin/env bash
# Ring Attention Inference submit script.
# Submits a single Slurm job (Insomnia/GPU) or runs a local process (CPU) in the background
# then tails the output.

set -eo pipefail
IFS=$'\n\t'

# --- Environment Detection ---
INSOMNIA_REPO_DIR="/insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack"
if [[ -d "$INSOMNIA_REPO_DIR" ]]; then
  RUN_LOCATION="insomnia"
  echo "[INFO] Detected Insomnia environment. Will use Slurm for GPU execution."
else
  RUN_LOCATION="local"
  echo "[INFO] Detected Local environment. Will use Python directly for CPU execution."
fi

# --- Base Paths ---
LOCAL_REPO_DIR="/Users/sadigulcelik/Documents/CompSci/HPML-2025-Spring/FMSwrapper/foundation-model-stack"

# --- Defaults ---
DEFAULT_MODEL_REL_PATH="../llama-hf"
DEFAULT_TOKENIZER_REL_PATH="../llama-hf/tokenizer.model"

# --- Repo & Model Paths ---
if [[ "$RUN_LOCATION" == "insomnia" ]]; then
  INSOMNIA_BASE_DIR="/insomnia001/depts/edu/COMSE6998/sg3790"
  CURRENT_REPO_DIR="${INSOMNIA_BASE_DIR}/foundation-model-stack"
  SLURM_SCRIPT_PATH="${CURRENT_REPO_DIR}/scripts/llama_ring/run_inference.slurm"
  DEFAULT_MODEL_ABS_PATH="${INSOMNIA_BASE_DIR}/llama-hf"
  DEFAULT_TOKENIZER_ABS_PATH="${INSOMNIA_BASE_DIR}/llama-hf/tokenizer.model"
else
  CURRENT_REPO_DIR="$LOCAL_REPO_DIR"
  DEFAULT_MODEL_ABS_PATH="${CURRENT_REPO_DIR}/${DEFAULT_MODEL_REL_PATH}"
  DEFAULT_TOKENIZER_ABS_PATH="${CURRENT_REPO_DIR}/${DEFAULT_TOKENIZER_REL_PATH}"
fi


echo "[INFO] cd into $CURRENT_REPO_DIR"
cd "$CURRENT_REPO_DIR"

if [[ "$RUN_LOCATION" == "insomnia" ]]; then
  echo "[INFO] Fetching latest changes and resetting local branch..."
  # Fetch latest changes from origin
  git fetch origin || echo "[WARN] git fetch failed"
  # Reset the current branch hard to its origin counterpart
  git reset --hard origin/$(git rev-parse --abbrev-ref HEAD) || echo "[WARN] git reset --hard failed"
fi

echo "[INFO] pip install -e ."
pip install -e . >/dev/null 2>&1 || echo "[WARN] pip install failed"

mkdir -p "${CURRENT_REPO_DIR}/testing"
cd "$HOME"

echo "[INFO] Cleaning old outputs…"
rm -f "$HOME"/inference_insomnia_*.out "${CURRENT_REPO_DIR}/testing/inference_local_*.out"

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
script_args=(); script_args+=("$@")
if ! printf '%s\n' "${script_args[@]}" | grep -q -- '--model_path'; then
  script_args+=(--model_path "$DEFAULT_MODEL_ABS_PATH")
fi
if ! printf '%s\n' "${script_args[@]}" | grep -q -- '--tokenizer'; then
  script_args+=(--tokenizer "$DEFAULT_TOKENIZER_ABS_PATH")
fi

echo "[INFO] Launching inference with args: ${script_args[*]}"

job_id=""; pid=""

if [[ "$RUN_LOCATION" != "insomnia" ]]; then
  # local
  # Define output file with timestamp before running the command
  timestamp=$(date +%Y%m%d_%H%M%S)
  output_file="${CURRENT_REPO_DIR}/testing/inference_local_${timestamp}.out"
  echo "[INFO] torchrun (nproc=2) → $output_file"
  torchrun --nproc_per_node=2 \
    "$CURRENT_REPO_DIR/scripts/inference.py" \
    --architecture llama --variant 7b \
    --device_type cpu --default_dtype fp16 \
    --model_source hf --no_use_cache \
    --distributed --distributed_strategy ring \
    "${script_args[@]}" \
    >"$output_file" 2>&1 &
  pid=$!
  # Construct filename after getting PID
  echo "[SUCCESS] local PID=$pid"
  wait_cmd="ps -p $pid"
else
  # insomnia
  echo "[INFO] sbatch ${SLURM_SCRIPT_PATH} ${script_args[*]}"
  sbatch_out=$(sbatch "$SLURM_SCRIPT_PATH" "${script_args[@]}" 2>&1) || {
    echo "[ERROR] sbatch failed: $sbatch_out"; exit 1
  }
  job_id=$(echo "$sbatch_out" | grep -oP 'Submitted batch job \K[0-9]+')
  # Match the updated Slurm script's #SBATCH --output pattern
  output_file="${HOME}/inference_insomnia_${job_id}.out"
  echo "[SUCCESS] Slurm job $job_id"
  wait_cmd="squeue -u $USER"
fi

echo "[INFO] Monitor with: $wait_cmd"
echo "[INFO] Will tail: $output_file"

# wait & tail
for i in {1..12}; do
  if [[ -s "$output_file" ]]; then
    echo; echo "[INFO] Tailing…"; tail -n +1 -f "$output_file"; exit 0
  fi
  printf '.'
  sleep 5
done

echo; echo "[ERROR] No output in $output_file"
cleanup
