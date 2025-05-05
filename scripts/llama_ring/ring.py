#!/usr/bin/env python3

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from datetime import datetime

# --- Environment Detection ---
INSOMNIA_REPO_DIR = Path("/insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack")
if INSOMNIA_REPO_DIR.is_dir():
    RUN_LOCATION = "insomnia"
    print("[INFO] Detected Insomnia environment. Will use Slurm for GPU execution.")
else:
    RUN_LOCATION = "local"
    print("[INFO] Detected Local environment. Will use Python directly for CPU execution.")

# --- Base Paths ---
LOCAL_REPO_DIR = Path("/Users/sadigulcelik/Documents/CompSci/HPML-2025-Spring/FMSwrapper/foundation-model-stack")

# --- Defaults ---
DEFAULT_MODEL_REL_PATH = Path("../llama-hf")
DEFAULT_TOKENIZER_REL_PATH = Path("../llama-hf/tokenizer.model")

# --- Repo & Model Paths ---
if RUN_LOCATION == "insomnia":
    INSOMNIA_BASE_DIR = Path("/insomnia001/depts/edu/COMSE6998/sg3790")
    CURRENT_REPO_DIR = INSOMNIA_BASE_DIR / "foundation-model-stack"
    SLURM_SCRIPT_PATH = CURRENT_REPO_DIR / "scripts/llama_ring/run_inference.slurm"
    DEFAULT_MODEL_ABS_PATH = INSOMNIA_BASE_DIR / "llama-hf"
    DEFAULT_TOKENIZER_ABS_PATH = INSOMNIA_BASE_DIR / "llama-hf/tokenizer.model"
else:
    CURRENT_REPO_DIR = LOCAL_REPO_DIR
    DEFAULT_MODEL_ABS_PATH = CURRENT_REPO_DIR / DEFAULT_MODEL_REL_PATH
    DEFAULT_TOKENIZER_ABS_PATH = CURRENT_REPO_DIR / DEFAULT_TOKENIZER_REL_PATH

print(f"[INFO] cd into {CURRENT_REPO_DIR}")
os.chdir(CURRENT_REPO_DIR)

if RUN_LOCATION == "insomnia":
    print("[INFO] Fetching latest changes and resetting local branch...")
    try:
        subprocess.run(["git", "fetch", "origin"], check=False)
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        subprocess.run(["git", "reset", "--hard", f"origin/{branch}"], check=False)
    except Exception as e:
        print(f"[WARN] Git operation failed: {e}")

print("[INFO] pip install -e .")
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception:
    print("[WARN] pip install failed")

(CURRENT_REPO_DIR / "testing").mkdir(parents=True, exist_ok=True)
home = Path.home()

print("[INFO] Cleaning old outputs…")
for f in home.glob("inference_insomnia_*.out"):
    f.unlink(missing_ok=True)
for f in (CURRENT_REPO_DIR / "testing").glob("inference_local_*.out"):
    f.unlink(missing_ok=True)

# Cleanup handler
def cleanup(signum=None, frame=None):
    print("\n[INFO] Cleaning up…")
    if RUN_LOCATION == "insomnia" and job_id:
        subprocess.run(["scancel", job_id], check=False)
    elif RUN_LOCATION == "local" and pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    sys.exit(130)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# --- Build args ---
script_args = sys.argv[1:]
if "--model_path" not in script_args:
    script_args += ["--model_path", str(DEFAULT_MODEL_ABS_PATH)]
if "--tokenizer" not in script_args:
    script_args += ["--tokenizer", str(DEFAULT_TOKENIZER_ABS_PATH)]

print(f"[INFO] Launching inference with args: {' '.join(script_args)}")

pid = None
job_id = None

if RUN_LOCATION != "insomnia":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = CURRENT_REPO_DIR / "testing" / f"inference_local_{timestamp}.out"
    print(f"[INFO] torchrun (nproc=2) → {output_file}")
    with open(output_file, "w") as f:
        process = subprocess.Popen(
            ["torchrun", "--nproc_per_node=2",
             str(CURRENT_REPO_DIR / "scripts/inference.py"),
             "--architecture", "llama", "--variant", "7b",
             "--device_type", "cpu", "--default_dtype", "fp16",
             "--model_source", "hf", "--no_use_cache",
             "--distributed", "--distributed_strategy", "ring",
             *script_args],
            stdout=f,
            stderr=subprocess.STDOUT
        )
    pid = process.pid
    print(f"[SUCCESS] local PID={pid}")
    wait_cmd = f"ps -p {pid}"
else:
    print(f"[INFO] sbatch {SLURM_SCRIPT_PATH} {' '.join(script_args)}")
    try:
        result = subprocess.run(
            ["sbatch", str(SLURM_SCRIPT_PATH)] + script_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )
        sbatch_out = result.stdout
        job_id = next((line.split()[-1] for line in sbatch_out.splitlines() if "Submitted batch job" in line), "")
        output_file = home / f"inference_insomnia_{job_id}.out"
        print(f"[SUCCESS] Slurm job {job_id}")
        wait_cmd = f"squeue -u {os.environ['USER']}"
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] sbatch failed: {e.output}")
        sys.exit(1)

print(f"[INFO] Monitor with: {wait_cmd}")
print(f"[INFO] Will tail: {output_file}")

# wait & tail
for _ in range(12):
    if output_file.exists() and output_file.stat().st_size > 0:
        print("\n[INFO] Tailing…")
        subprocess.run(["tail", "-n", "+1", "-f", str(output_file)])
        sys.exit(0)
    print(".", end="", flush=True)
    time.sleep(5)

print(f"\n[ERROR] No output in {output_file}")
cleanup()
