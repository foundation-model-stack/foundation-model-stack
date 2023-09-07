# Foundation Model Stack

Foundation Model Stack is a collection of components for development, training,
and tuning of foundation models.

## Installation

```
pip install -e .
```
or
```
python setup.py install
```

There's an example inference script under `./examples`.


## Llama

The first model we are releasing here is our reimplementation of Llama. Our reasoning behind reimplementing Llama is to support the combination of torch.compile(), PyTorch's scaled_dot_product_attention, and Tensor Parallel. Currently, no other open-source implementation we are aware of supports the three at the same time, for a variety of reasons.

PSA: As of 09/07/2023, there's still two pending issues in PyTorch preventing Tensor Parallel from fully working with torch.compile(): https://github.com/pytorch/pytorch/issues/107824 prevents training/finetuning from working, and https://github.com/pytorch/pytorch/issues/108780 requires adding graph breaks to preserve accuracy. Once these are fixed, the performance numbers reported below can be achieved without any loss of accuracy. Without these fixes, the combination compile+TP+SDPA will be fast but generate incorrect results.

Without further ado, here be some numbers/plots:

![image](https://github.com/ibm-pytorch/foundation-model-stack/assets/919977/16ea178f-1c50-4f26-b549-dd21f73f51f8)

First, let's explain this plot: here, we're comparing the implementation of LlaMA-7B in this repo (llama-fms) to the implementation hosted in HuggingFace's TGI (llama), which uses Flash Attention v2 and manually fused kernels. We observe similar performance between our compiled llama (compile+SDPA) and the manually tuned one in TGI. VUS stands for Virtual UserS, and has a direct equivalency to the batch size in this benchmark. This is running on a single A100 80GB GPU using the watsonx.ai infrastructure and benchmarking code, which lets us easily compare to the HF TGI implementation on a similar environment (IBM's TGIS).

Similar results for 13B on single GPU will be posted shortly.

As for TP+compile+SDPA results, we have observed the following (with incorrect results due to issues in PSA above and with the caveat that once the issues are fixed there might be a performance change):

4 A100 80GB GPUs: 34.5 ms/token for bs=1 (vs 50-55ms/token in the TGI implementation running on the same stack w/ Flash V2 and a custom layer norm kernel)

8 A100 80GB GPUs: 29 ms/token for bs=1

Further performance studies will be posted here as they come out, as well as open-source benchmarking scripts to reproduce the results seen here.

## References

Huggingface TGI: https://github.com/huggingface/text-generation-inference
IBM TGIS: https://github.com/IBM/text-generation-inference