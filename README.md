# Foundation Model Stack

Foundation Model Stack is a collection of components for development, training,
and tuning of foundation models leveraging PyTorch native components only. The current (09/08/2023) state is that we support inference optimizations on LLaMa models. For inference optimizations, the main approach we take is to leverage PyTorch compile, accelerated transformers, and tensor parallelism. For training optimizations, we use FSDP and the various sharding strategies, accelerated transformers, and PyTorch compile.

## Installation

```
pip install -e .
```
or
```
python setup.py install
```

For an example inference, one can run the script in `./examples`.


## Inference

Our approach for inference optimization as we stated earlier is to use PyTorch compile, accelerated transformers, and tensor parallelism. PyTorch compile compiles the code into optimized kernels, accelerated transformers leverages `scaled_dot_product_attention` (SDPA) for accelerating attention computation while saving memory, and tensor parallelism is necessary for larger models like LLaMa 70B. In our experiments with various models, `compile` has given 2-2.5x speedups for inference, with `SDPA` providing 30-40% improvements.

Given the popularity of the LLaMa2 models, we target the optimization of inference of the LLaMa2 family of models. This particular repository re-implements the LLaMa architecture so that the `RoPE` encodings will compile, we have verified that `forward` pass compiles (there is work that needs to be done for the `backward` to work with FSDP).

TODO: Get graph from Mudhakar and remove 70B from it.

Tensor parallel inference numbers are **coming soon**!

## Training (Coming Soon!!)

## Open Issues
* https://github.com/pytorch/pytorch/issues/108780 requires adding graph breaks to preserve accuracy.
* https://github.com/pytorch/pytorch/issues/107824 prevents training/finetuning from working

## References

Huggingface TGI: https://github.com/huggingface/text-generation-inference
IBM TGIS: https://github.com/IBM/text-generation-inference
