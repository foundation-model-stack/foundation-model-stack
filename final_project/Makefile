PDF=report

all:
	latexmk -pdf $(PDF).tex

clean:
	latexmk -C

# --------------------------------------------------------------------
# Project helpers
# --------------------------------------------------------------------

# Create a local virtual‑environment and install all runtime deps
venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	# Install numpy first, then install the root repo (one level up) with extras
	.venv/bin/pip install numpy
	# The project’s `pyproject.toml` / `setup.py` live in the repo root, not here
	.venv/bin/pip install -e '..[inference,benchmark]'

# ---------------------------------------------------
# Benchmarks mirroring FlashAttention paper settings
# ---------------------------------------------------

# ▶ Quick single‑GPU sanity check on Llama‑2‑7B
llama-sanity: venv
	CUDA_VISIBLE_DEVICES=0 .venv/bin/python ../scripts/inference.py \
		--architecture=llama \
		--variant=7b \
		--tokenizer ~/llama_weights/tokenizer.model \
		--ckpt_dir ~/llama_weights/llama-2-7b/ \
		--seq_len 512 \
		--batch_size 1 \
		--use_flash \
		--repetitions 3

# ▶ Full FlashAttention sweep on Llama‑2‑7B
llama-benchmark: venv
	.venv/bin/torchrun --nnodes 1 --nproc-per-node 8 --standalone \
		../scripts/benchmark_inference.py \
		--architecture=llama \
		--variant=7b \
		--tokenizer ~/llama_weights/tokenizer.model \
		--ckpt_dir ~/llama_weights/llama-2-7b/ \
		--seq_lens 128 256 512 1024 2048 4096 8192 16384 32768 65536 \
		--batch_sizes 2 1 \
		--dtype fp16 \
		--compile_mode reduce-overhead \
		--distributed \
		--check_correctness \
		--repetitions 3

# ▶ FlashAttention sweep on Granite‑7B‑Instruct
granite-benchmark: venv
	.venv/bin/torchrun --nnodes 1 --nproc-per-node 8 --standalone \
		../scripts/benchmark_inference.py \
		--architecture=granite \
		--variant=7b-instruct \
		--tokenizer ~/granite_weights/tokenizer.model \
		--ckpt_dir ~/granite_weights/granite-7b-instruct/ \
		--seq_lens 128 256 512 1024 2048 4096 8192 16384 32768 65536 \
		--batch_sizes 2 1 \
		--dtype fp16 \
		--use_flash \
		--compile_mode reduce-overhead \
		--distributed \
		--repetitions 3

# Run all benchmarks
benchmarks: llama-benchmark granite-benchmark