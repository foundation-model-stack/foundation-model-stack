# Foundation Model Stack

Foundation Model Stack is a collection of components for development of training, tuning, and inference of foundation models leveraging PyTorch native components.  For inference optimizations, the main approach we take is to leverage PyTorch compile, accelerated transformers, and tensor parallelism. For training optimizations, we use FSDP and the various sharding strategies, accelerated transformers, and PyTorch compile.

## Status
* 09/08/2023: Inference on 7B, 13B LLaMa models

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

The figure below shows the latency improvements as we move from eager mode execution to adding SDPA, compile, and SDPA+Compile. The measurements are for 7 and 13B models.
![image (21)](https://github.com/ibm-pytorch/foundation-model-stack/assets/8322403/3d9c6a0f-c3ef-454b-806c-271f352afa4d)


Tensor parallel inference numbers for 13B and 70B models are **coming soon**!

## Training (Coming Soon!!)

## Open Issues
* https://github.com/pytorch/pytorch/issues/108780 requires adding graph breaks to preserve accuracy.
* https://github.com/pytorch/pytorch/issues/107824 prevents training/finetuning from working

## References

* Huggingface TGI: https://github.com/huggingface/text-generation-inference
* IBM TGIS: https://github.com/IBM/text-generation-inference
