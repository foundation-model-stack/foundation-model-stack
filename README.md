# Foundation Model Stack

Foundation Model Stack is a collection of components for development, inference, training, and tuning of foundation models leveraging PyTorch native components. For inference optimizations we aim to support PyTorch compile, accelerated transformers, and tensor parallelism. At training time we aim to support FSDP, accelerated transformers, and PyTorch compile.

## Models Supported
| Model family | Inference | Tuning | Training |
| ----------- | ---------- | -------- | ----- |
| Llama | :heavy_check_mark: | :x: | :x: |

## Installation

### Pypi

```
pip install ibm-fms
```

### Local

Requires [PyTorch >= 2.1](https://pytorch.org/get-started/locally/).

```
pip install -e .
```
or
```
python setup.py install
```


## Inference

#### Approach
Our approach for inference optimization is to use PyTorch compile, accelerated transformers, and tensor parallelism. PyTorch compile compiles the code into optimized kernels, accelerated transformers leverages `scaled_dot_product_attention` (SDPA) for accelerating attention computation while saving memory, and tensor parallelism is necessary for larger models.

We provide a re-implementation of the Llama architecture. To enable the model to compile, we reimplement `RoPE` encodings without complex numbers. We have verified that the `forward` pass compiles (there is work that needs to be done for `backward` to work with FSDP).

#### Inference latency
We measured inference latencies with 1024 token prompt and generation of 256 tokens on AWS P4de instance nodes with 8 80G A100 GPUs and report the median latency in the below table.
| Model | # GPUs | Median latency (ms) |
| ----- | ----------- | ----- |
| 7B | 1 | 14ms |
| 13B | 1 | 22ms |
| 70B | 8 | 30ms |

If you would like to reproduce the latencies, you can run the `scripts/benchmark_inference.py` and the details are described in [inference](./scripts).

## HF Model Support

You can experience the models from Hugging Face APIs using our HF model adapter. The speedups obtained were measured using the Meta checkpoints for the models. One can also obtain similar speedups using the HF adapter with our implementation of the Llama architecture as follows:

```python
# fms model
llama: LLaMA = LLaMA(config)

# huggingface model backed by fms internals
llama_hf = LLaMAHFForCausalLM.from_fms_model(llama)

# compile the model -- in HF, the decoder only
model.decoder = torch.compile(model.decoder)

# generate some text -- the first time will be slow since the model needs to be compiled, but subsequent generations should be faster.
llama_generator = pipeline(task="text-generation", model=llama_hf, tokenizer=tokenizer)
llama_generator("""q: how are you? a: I am good. How about you? q: What is the weather like today? a:""")
```

A detailed example is provided [here](./notebooks/hf_llama_generation_example.ipynb).

## Tuning (Coming Soon!!)

## Training (Coming Soon!!)

## Open Issues
There are open issues that we are tracking to improve the stability and memory footprint of the inference latencies.

* https://github.com/pytorch/pytorch/issues/107824 prevents training/finetuning from working

## References

* Huggingface TGI: https://github.com/huggingface/text-generation-inference
* IBM TGIS: https://github.com/IBM/text-generation-inference
