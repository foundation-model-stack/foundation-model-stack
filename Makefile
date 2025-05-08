# --------------------------------------------------------------------
# Project helpers
# --------------------------------------------------------------------

# Show brief help for new contributors
help:
	@echo "  venv                 Create virtual environment and install runtime deps"
	@echo "  deps                 Install/refresh project & test dependencies (called automatically)"
	@echo "  test                 Run full test suite"
	@echo "  test-paged-attention Run paged attention tests"
	@echo "  test-embeddings      Run embedding‑specific tests"
	@echo "  check-torch          Print PyTorch & CUDA info"
	@echo "  download-tokenizer    Fetch the LLaMA SentencePiece tokenizer"
	@echo "  bench-llama          Benchmark LLaMA $(LLAMA_VARIANT) (default attention)"
	@echo "  bench-llama-paged    Benchmark LLaMA $(LLAMA_VARIANT) using paged attention"
	@echo "  bench-llama-t4       Memory‑friendly LLaMA 7B benchmark for 16 GB T4 GPUs"
	@echo "  bench-llama-paged-t4 Memory‑friendly paged LLaMA 7B benchmark for 16 GB T4 GPUs"
	@echo "  report           	  Compile final_project/report.tex → PDF"
	@echo "  report-clean         Remove LaTeX aux files & built PDF"
	@echo "  clean                Remove virtual environment, cache & stamp file"
	@echo "  help                 Show this message"
	@echo "  bench-attention-runtime  Benchmark attention runtime (default & paged) for various sequence lengths (CSV output)"
	@echo "  profile-memory         Profile peak memory usage for various sequence lengths (default & paged)"
	@echo "  profile-throughput     Profile throughput for various sequence lengths (default & paged)"

# --------------------------------------------------------------------
# Common variables & meta‑targets
# --------------------------------------------------------------------
PYTHON ?= python3
VENV_DIR ?= .venv
PIP := $(VENV_DIR)/bin/pip
ACTIVATE := . $(VENV_DIR)/bin/activate

# Dependency management
# If your project uses Poetry/PEP‑517 a pyproject.toml will exist; otherwise this
# expands to an empty string so Make does not choke on a missing file.
PYPROJECT := $(wildcard pyproject.toml)
# File listing pytest/CI extras; adjust if you renamed it
TEST_REQS := test-requirements.txt
REQS := requirements.txt
# Benchmark helpers
BENCH_SCRIPT := scripts/benchmark_inference.py
LLAMA_VARIANT ?= 7b                     # default; override e.g. 13b via `make LLAMA_VARIANT=13b bench-llama`

# Where we keep the tokenizer (file target below will download it if missing)
TOKENIZER_FILE := $(HOME)/llama_weights/tokenizer.model
TOKENIZER_URL  ?= https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model

# Path passed to benchmark script (override with `make TOKENIZER=/your/path`)
TOKENIZER ?= $(TOKENIZER_FILE)
# Extra args can be supplied at the CLI:  `make bench-llama EXTRA="--seq_len=1024"`
EXTRA ?=

# Preset for 16 GB T4: smaller prompt, no compile, and skip strict fp16 correctness check
T4_EXTRA := --batch_size=1 --seq_len=256 --max_new_tokens=128 \
            --skip_compile_runs --skip_correctness_check --skip_nokvcache_runs

# --------------------------------------------------------------------
# Automatic download of the SentencePiece tokenizer
# --------------------------------------------------------------------
$(TOKENIZER_FILE):
	@echo "Downloading LLaMA tokenizer to $@…"
	@mkdir -p $(dir $@)
	curl -L --fail -o $@ "$(TOKENIZER_URL)"
	@echo "✔  Tokenizer downloaded"

# Stamp file to track installed dependencies
DEPS_STAMP := $(VENV_DIR)/.deps_stamp

.PHONY: venv deps test test-embeddings check-torch report report-clean clean help bench-llama bench-llama-paged bench-llama-t4 bench-llama-paged-t4 download-tokenizer profile-memory profile-throughput

# Create virtual‑env if it doesn't exist
venv: $(VENV_DIR)/bin/python

$(VENV_DIR)/bin/python:
	$(PYTHON) -m venv $(VENV_DIR)

# Install/refresh project & test dependencies exactly once
deps: $(DEPS_STAMP)

$(DEPS_STAMP): $(PYPROJECT) $(TEST_REQS) $(REQS) | venv
	$(PIP) install --upgrade pip
	# install core runtime deps (nightly PyTorch pinned in requirements.txt)
	$(PIP) install -r $(REQS)
	# install project in editable mode
	$(PIP) install -e .
	# install test‑only extras
	$(PIP) install -r $(TEST_REQS)
	touch $@

# Check PyTorch version & CUDA availability
check-torch: deps
	$(VENV_DIR)/bin/python -c "import torch, sys; print(f'PyTorch version: {torch.__version__}\\nCUDA available: {torch.cuda.is_available()}'); print(f'GPU device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '')"

# Clean up
clean:
	rm -rf $(VENV_DIR) $(DEPS_STAMP) __pycache__ $(REPORT_PDF)

# --------------------------------------------------------------------
# Test targets
# --------------------------------------------------------------------

# Run unit tests
test: deps
	$(VENV_DIR)/bin/pytest tests

# Run paged attention tests
test-paged-attention: deps
	$(VENV_DIR)/bin/pytest tests/modules/test_paged_attention.py

# Run embeddings tests
test-embeddings: deps
	$(VENV_DIR)/bin/pytest tests/modules/test_embedding.py
	

# --------------------------------------------------------------------
# Benchmark targets  (LLaMA 7B single‑GPU inference)
# --------------------------------------------------------------------

## Eager + compiled benchmarks (default attention implementation)
bench-llama: deps $(TOKENIZER_FILE)
	@echo "Running benchmark on LLaMA $(LLAMA_VARIANT) (default attention)…"
	CUDA_VISIBLE_DEVICES=0 $(VENV_DIR)/bin/python $(BENCH_SCRIPT) \
	    --architecture=llama --variant=$(LLAMA_VARIANT) \
	    --tokenizer="$(TOKENIZER)" $(EXTRA)

## Same benchmarks but force paged‑attention via env var picked up in attention.py
bench-llama-paged: deps $(TOKENIZER_FILE)
	@echo "Running benchmark on LLaMA $(LLAMA_VARIANT) with paged‑attention…"
	CUDA_VISIBLE_DEVICES=0 FMS_ATTENTION_ALGO=paged \
	    $(VENV_DIR)/bin/python $(BENCH_SCRIPT) \
	    --architecture=llama --variant=$(LLAMA_VARIANT) \
	    --tokenizer="$(TOKENIZER)" $(EXTRA)

## Memory‑friendly benchmark for 16 GB GPUs like NVIDIA T4
bench-llama-t4: deps $(TOKENIZER_FILE)
	@echo "Running memory‑friendly benchmark (T4 preset)…"
	CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	    $(VENV_DIR)/bin/python $(BENCH_SCRIPT) \
	    --architecture=llama --variant=$(LLAMA_VARIANT) \
	    --tokenizer="$(TOKENIZER)" $(T4_EXTRA) $(EXTRA)

## Memory‑friendly paged‑attention benchmark for 16 GB GPUs like NVIDIA T4
bench-llama-paged-t4: deps $(TOKENIZER_FILE)
	@echo "Running memory‑friendly benchmark (T4 preset) with paged‑attention…"
	CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FMS_ATTENTION_ALGO=paged \
	    $(VENV_DIR)/bin/python $(BENCH_SCRIPT) \
	    --architecture=llama --variant=$(LLAMA_VARIANT) \
	    --tokenizer="$(TOKENIZER)" $(T4_EXTRA) $(EXTRA)

download-tokenizer: $(TOKENIZER_FILE)

# Run memory profiling for a single forward pass at a given sequence length and batch size.
# Usage:
#   make mem-profile [SEQ_LEN=4096] [BATCH_SIZE=1] [TOKENIZER=~/llama_weights/tokenizer.model]
mem-profile:
	$(VENV_DIR)/bin/python scripts/benchmark_inference.py \
		--profile_memory \
		--seq_len=$(SEQ_LEN) \
		--batch_size=$(BATCH_SIZE) \
		--architecture=llama \
		--variant=7b \
		--tokenizer=$(TOKENIZER)

# Run throughput profiling for multiple concurrent generation requests.
# Usage:
#   make throughput-profile [NUM_REQUESTS=4] [SEQ_LEN=4096] [BATCH_SIZE=1] [TOKENIZER=~/llama_weights/tokenizer.model]
throughput-profile:
	$(VENV_DIR)/bin/python scripts/benchmark_inference.py \
		--profile_throughput \
		--num_requests=$(NUM_REQUESTS) \
		--seq_len=$(SEQ_LEN) \
		--batch_size=$(BATCH_SIZE) \
		--architecture=llama \
		--variant=7b \
		--tokenizer=$(TOKENIZER)

# Default values (can be overridden on the command line)
NUM_REQUESTS ?= 4
SEQ_LEN ?= 4096
BATCH_SIZE ?= 1
TOKENIZER ?= $(HOME)/llama_weights/tokenizer.model

# --------------------------------------------------------------------
# LaTeX targets  (final_project/report.tex → final_project/report.pdf)
# --------------------------------------------------------------------
REPORT_DIR := final_project
REPORT_TEX := $(REPORT_DIR)/report.tex
REPORT_PDF := $(REPORT_DIR)/report.pdf

.PHONY: report report-clean

# Build the PDF (prefers latexmk; falls back to two pdflatex passes)
report: $(REPORT_PDF)

$(REPORT_PDF): $(REPORT_TEX)
ifeq ($(shell command -v latexmk 2>/dev/null),)
	@echo "latexmk not found – falling back to pdflatex (running twice)…"
	cd $(REPORT_DIR) && pdflatex -interaction=nonstopmode $(notdir $(REPORT_TEX)) >/dev/null
	cd $(REPORT_DIR) && pdflatex -interaction=nonstopmode $(notdir $(REPORT_TEX)) >/dev/null
else
	cd $(REPORT_DIR) && latexmk -pdf -interaction=nonstopmode $(notdir $(REPORT_TEX))
endif
	@echo "✔  PDF generated at $(REPORT_PDF)"

# Remove auxiliary files (and the PDF when doing a full clean)
report-clean:
ifeq ($(shell command -v latexmk 2>/dev/null),)
	cd $(REPORT_DIR) && rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk
else
	cd $(REPORT_DIR) && latexmk -C
endif
	rm -f $(REPORT_PDF)

bench-attention-runtime: deps $(TOKENIZER_FILE)
	@echo "Running attention runtime benchmark (default attention)…"
	CUDA_VISIBLE_DEVICES=0 $(VENV_DIR)/bin/python scripts/benchmark_attention_runtime.py \
	    --architecture=llama --variant=$(LLAMA_VARIANT) \
	    --tokenizer="$(TOKENIZER)" --output_csv=attention_runtime_default.csv
	@echo "Running attention runtime benchmark (paged attention)…"
	CUDA_VISIBLE_DEVICES=0 FMS_ATTENTION_ALGO=paged \
	    $(VENV_DIR)/bin/python scripts/benchmark_attention_runtime.py \
	    --architecture=llama --variant=$(LLAMA_VARIANT) \
	    --tokenizer="$(TOKENIZER)" --paged --output_csv=attention_runtime_paged.csv
	@echo "✔  Results written to attention_runtime_default.csv and attention_runtime_paged.csv"

profile-memory: deps $(TOKENIZER_FILE)
	@echo "Profiling peak memory usage (default attention)…"
	CUDA_VISIBLE_DEVICES=0 $(VENV_DIR)/bin/python scripts/benchmark_profile_memory.py \
	    --architecture=llama --variant=$(LLAMA_VARIANT) \
	    --tokenizer="$(TOKENIZER)" > profile_memory_default.tsv
	@echo "Profiling peak memory usage (paged attention)…"
	CUDA_VISIBLE_DEVICES=0 FMS_ATTENTION_ALGO=paged \
	    $(VENV_DIR)/bin/python scripts/benchmark_profile_memory.py \
	    --architecture=llama --variant=$(LLAMA_VARIANT) \
	    --tokenizer="$(TOKENIZER)" --paged > profile_memory_paged.tsv
	@echo "✔  Results written to profile_memory_default.tsv and profile_memory_paged.tsv"

profile-throughput: deps $(TOKENIZER_FILE)
	@echo "Profiling throughput (default attention)…"
	CUDA_VISIBLE_DEVICES=0 $(VENV_DIR)/bin/python scripts/benchmark_profile_throughput.py \
	    --architecture=llama --variant=$(LLAMA_VARIANT) \
	    --tokenizer="$(TOKENIZER)" > profile_throughput_default.tsv
	@echo "Profiling throughput (paged attention)…"
	CUDA_VISIBLE_DEVICES=0 FMS_ATTENTION_ALGO=paged \
	    $(VENV_DIR)/bin/python scripts/benchmark_profile_throughput.py \
	    --architecture=llama --variant=$(LLAMA_VARIANT) \
	    --tokenizer="$(TOKENIZER)" --paged > profile_throughput_paged.tsv
	@echo "✔  Results written to profile_throughput_default.tsv and profile_throughput_paged.tsv"

