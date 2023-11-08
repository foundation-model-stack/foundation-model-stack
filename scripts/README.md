# Example scripts for the repo

## Inference script

This example script validates the LLaMA implementation by running inference on a couple of prompts.

## Benchmark script

This script lets you run a battery of timing tests on our LLaMa implementation. The specific test matrix is the following:
- Batch size (`--batch_size`), input length (`--seq_len`), and generation length (`--max_new_tokens`) can be controlled. Defaults: 2, 512, 256
- By default, we run single token generation, sequence generation, eager, compile, no kv-cache, and kv-cache test combinations, which ends up being 2 x 2 x 2 = 8 tests. Any of the combinations can be disabled with the following flags: `--skip_eager_runs`, `--skip_compile_runs`, `--skip_kvcache_runs`, `--skip_nokvcache_runs`, `--skip_single_token_runs`, `--skip_e2e_runs`.
- If you want to check the code is producing the same output tokens, you can add the `--check_correctness` flag.
- For best compilation performance (with BIG caveats), you can use `--compile_mode="reduce-overhead"`. In PyTorch 2.1, there are several memory management issues with this mode, which will cause memory leaks, memory fragmentation, and eventually OOMs, as well as possible race conditions. Therefore, we recommend running only the test you want to get performance numbers for, as the program will eventually crash from lack of memory.
- If you're running 13B or 70B tests, use the `--distributed` flag to let the program know it should initialize all the distributed machinery.

An example command to run a 70B test with expected maximum performance would be:
```
torchrun --nnodes=1 --nproc-per-node=8 --standalone scripts/benchmark_inference.py --architecture=llama --variant=70b --tokenizer="~/llama_weights/tokenizer.model" --distributed --compile_mode="reduce-overhead"
```

And to get the end to end generation numbers, you would do:

```
torchrun --nnodes=1 --nproc-per-node=8 --standalone scripts/benchmark_inference.py --architecture=llama --variant=70b --tokenizer="~/llama_weights/tokenizer.model" --distributed --compile_mode="reduce-overhead" --skip_eager_runs --skip_single_token_runs --skip_nokvcache_runs --skip_correctness_check
```

While running a single GPU 7B test would be:
```
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_inference.py --architecture=llama --variant=7b --tokenizer="~/llama_weights/tokenizer.model"
```
