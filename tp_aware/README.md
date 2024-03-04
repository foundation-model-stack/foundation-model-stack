#TP-Aware GPTQ Method

In our paper we describe a method to avoid all-gather while keeping the weight matrix data locality optimization required by exllama kernel. This folder contains an implementation of the method in vLLM by offline preshuffling the weight matrix of the Llama MLP layers.

There are 3 layers: `up_proj`, `gate_proj` and `down_proj` in LlamaMLP. In TP mode, the `up_proj` and `gate_proj` are column-sharded, and the down_proj is row-sharded. To avoid all-gather inbetween column-parallel and row-parallel layers, the following steps are taken to preshuffle the data of these three layers:

- For each MLP block, compute `perm` from `down_proj.g_idx`. `perm` is then used to shuffle the columns of `qweight`, `scales`, `zeros` of both `up_proj` and `gate_proj`.
  - Note that the weight rows of `up_proj` and `gate_proj` are not shuffled using their `g_idx` offline. vLLM will take care of those during model loading.
- Use `perm` computed in the previous step to shuffle the rows of `down_proj.qweight`
  - This is necessary because with the previous step, the combined activations of the `up_proj` and `gate_proj` will be in an order that corresponds to shuffled `down_proj.qweight`. Also, vLLM doesn't have logic that performs global shuffling of rows **before** row-sharding.

## Performance (short prompt)
### 2 GPUS (4, 5 on netsres118)

```
Preshuffled checkpoint act=True, hacked vLLM
INFO 01-29 20:11:59 llm_engine.py:711] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 33.7 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.1%, CPU KV cache usage: 0.0%
Normal GPTQ checkpoint act=True, Vanilla vLLM
INFO 01-29 20:18:09 llm_engine.py:706] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 20.8 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.1%, CPU KV cache usage: 0.0%
```

### 2 GPUS (0, 1 on L40S)
```
Preshuffled checkpoint act=True, hacked vLLM
INFO 01-30 19:18:01 llm_engine.py:711] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 29.6 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.4%, CPU KV cache usage: 0.0%
Normal GPTQ checkpoint act=True, Vanilla vLLM
INFO 01-30 19:30:18 llm_engine.py:706] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 21.5 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.2%, CPU KV cache usage: 0.0%
```
