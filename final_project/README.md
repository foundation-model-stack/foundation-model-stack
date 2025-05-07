# COMS E6998 - HPML Final Project (Spring 2025) - Group 3

## Description

Our project aims to integrate PyTorch's Paged Attention into the Foundation Model Stack (FMS) using Flex Attention. We intend to enhance memory efficiency and inference speed for long-context language models without sacrificing model accuracy. Specifically, we will implement a dynamic, paged key-value (KV) cache that minimizes memory fragmentation, benchmark its performance against standard attention mechanisms, and evaluate the impact of various paging strategies on overall model performance.

## Outline of Code Repository

```text
foundation-model-stack/
├── .github/
│   └── workflows/
├── final_project/
│   ├── README.md   # This file!
│   └── report.tex  # Final project report
├── fms/
│   ├── datasets/
│   ├── models/
│   │   ├── gpt_bigcode/
│   │   ├── llama/
│   │   ├── roberta/
│   │   └── hf/
│   ├── modules/
│   │   ├── attention.py  # This is the core implementation of paged attention
│   │   └── # other files
│   ├── training/
│   └── utils/
├── notebooks/
├── scripts/
│   ├── benchmark_inference.py
│   ├── train_causal.py
│   └── # other helper & benchmark scripts
├── static/
├── tests/
│   ├── modules/
│   │   ├── test_paged_attention.py  # Unit test that we added to validate behavior of paged attention implementation
│   │   └── # other files
├── .gitignore
├── .isort.cfg
├── LICENSE
├── README.md
├── code-of-conduct.md
├── hf-requirements.txt
├── requirements.txt  # Includes using a Python wheel for installing the latest/nightly PyTorch dev build
├── test-requirements.txt
├── Makefile  # Includes all of the setup and build/test targets that we add/use in the repo
└── setup.py
```

## Example Commands

### Dev Env Setup: Ensure PyTorch 2.8+dev is installed and GPU with CUDA is available

```bash
make check-torch
```

Example output:
```bash
ndhillon@instance-20250303-021938:~/foundation-model-stack$ make check-torch
.venv/bin/python -c "import torch, sys; print(f'PyTorch version: {torch.__version__}\\nCUDA available: {torch.cuda.is_available()}'); print(f'GPU device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '')"
PyTorch version: 2.8.0.dev20250503+cu126
CUDA available: True
GPU device: Tesla T4
```

### Download Llama Tokenizer (needed for Llama inference)

```bash
make download-tokenizer
```

### Run Llama Inference Benchmarks with Regular Attention

```bash
make bench-llama

# If you are using a machine with < 16GB of GPU memory, recommend running a lighter benchmark
make bench-llama-t4
```


### Run Llama Inference Benchmarks with Paged Attention

```bash
bench-llama-paged

# If you are using a machine with < 16GB of GPU memory, recommend running a lighter benchmark
make bench-llama-paged-t4
```

### Run Paged Attention Unit Tests

```bash
make test-paged-attention
```

## Results

- TODO: Results (including charts/tables) and your observations  

## Wandb Project Board

https://wandb.ai/nsd2147-columbia-university/HPML%20Final%20Project/overview
