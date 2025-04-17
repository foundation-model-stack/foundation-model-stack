# COMS E6998 - High Performance Machine Learning Final Project (Spring 2025)

## Dev Env Setup

### Convert .tex to .pdf

```bash
sudo apt-get update
sudo apt-get install texlive-full

make all
make clean
```

### Setup benchmarks

```bash
git clone https://github.com/thomasjoshi/foundation-model-stack.git
cd foundation-model-stack

python3 -m venv .venv
source .venv/bin/activate
pip install -e .[inference,benchmark]


```


## GPU Memory Hierarchy on NVIDIA T4

| Level | Size | Bandwidth / Latency | Notes |
|-------|------|---------------------|-------|
| Registers | 64 k × 32‑bit per SM | ~20 TB/s aggregate | fastest, not addressable for tiling |
| Shared Mem / L1 | Configurable 64 KB : 32 KB or 32 KB : 64 KB per SM | ~7–8 TB/s¹ | we’ll request 64 KB shared / 32 KB L1 |
| L2 cache | 4 MB total | ~1 TB/s (est.) | — |
| HBM (GDDR6) | 16 GB, 320 GB/s | 300–400 ns | quadratic memory growth lives here |

¹Measured by Jia *et al.* “Dissecting the Nvidia Turing T4 GPU via Micro‑benchmarking” (2019).

## Choosing the right tile (block) size

FlashAttention shows that performance is maximized when one tile fits *exactly* shared memory and avoids register spilling. 
The NVIDIA T4 has 64KB of SRAM, therefore all benchmarks on this GPU should be using FLASH_ATTENTION_BLOCK=64.

Note: This is exactly the block size the FlashAttention authors selected for T4 in Fig. 8 (page 29) where they note that “T4 SRAM is smaller...so we need to make the block sizes smaller in FlashAttention”

## Benchmark Test Suite

| Model | Context lengths | Batch sizes | Dtype |
|-------|-----------------|------------|-------|
| Llama‑2‑7B‑HF | 128, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K | 1 (≤ 8 K) / 2 (≤ 2 K) | fp16 |
| Granite‑7B‑Instruct | same | same | fp16 |

We’ll run **3 repetitions** per point to smooth noise.

### Llama

```bash
// TODO: Implement this.
```

### IBM granite

```bash
// TODO: Implement this.
```
