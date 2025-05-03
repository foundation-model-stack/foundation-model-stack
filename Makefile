# --------------------------------------------------------------------
# Project helpers
# --------------------------------------------------------------------

# Show brief help for new contributors
help:
	@echo "Available targets:"
	@echo "  venv              Create virtual environment and install runtime deps"
	@echo "  deps              Install/refresh project & test dependencies (called automatically)"
	@echo "  test              Run full test suite"
	@echo "  test-embeddings   Run embedding‑specific tests"
	@echo "  check-torch       Print PyTorch & CUDA info"
	@echo "  report            Compile final_project/report.tex → PDF"
	@echo "  report-clean      Remove LaTeX aux files & built PDF"
	@echo "  clean             Remove virtual environment, cache & stamp file"
	@echo "  help              Show this message"

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
# Stamp file to track installed dependencies
DEPS_STAMP := $(VENV_DIR)/.deps_stamp

.PHONY: venv deps test test-embeddings check-torch report report-clean clean help

# Create virtual‑env if it doesn't exist
venv: $(VENV_DIR)/bin/python

$(VENV_DIR)/bin/python:
	$(PYTHON) -m venv $(VENV_DIR)

# Install/refresh project & test dependencies exactly once
deps: $(DEPS_STAMP)

$(DEPS_STAMP): $(PYPROJECT) $(TEST_REQS) | venv
	$(PIP) install --upgrade pip
	$(PIP) install -e .
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
	$(VENV_DIR)/bin/python -m unittest discover -s tests -p "test_*.py"
	
# Run embeddings tests
test-embeddings: deps
	$(VENV_DIR)/bin/python -m unittest tests.modules.test_embedding
	

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