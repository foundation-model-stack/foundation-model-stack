import dataclasses
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn


@dataclasses.dataclass
class ModelSignatureParams:
    """Model Signature params dataclass for readability"""

    model: nn.Module
    params: Union[int, List[str]]
    other_params: Optional[Dict] = None
    logits_getter_fn: Optional[Callable] = None
    inp: Optional[torch.LongTensor] = None


@dataclasses.dataclass
class HFModelSignatureParams(ModelSignatureParams):
    """Specific form of model Signature params which defaults the other params and logits getter to take what hf requires"""

    other_params: Optional[Dict] = dataclasses.field(
        default_factory=lambda: {"return_dict": True}
    )
    logits_getter_fn: Optional[Callable] = lambda o: o.logits


def get_signature(
    model,
    params: Union[int, List[str]] = 1,
    inp: Optional[torch.LongTensor] = None,
    optional_params: Optional[dict] = None,
    logits_getter_fn: Optional[Callable] = None,
    device: Union[int, str] = "cpu",
) -> List[float]:
    """Takes a model, and the number of inputs / named parameters it expects in a forward pass and returns a compressed
    signature that acts as an effective tool for output correctness checking within some tolerance

    Note: signatures will always be created with fp32 precision

    Parameters
    ----------
    model: nn.Module
        the model to use to produce a signature
    params: int or list(str), optional
        the params to set to the default tensor value (inp). If an integer, will use *args, if a list, will use **kwargs (default is 1)
    inp: torch.LongTensor, optional
        the input to use for params. If not given, torch.arange(0, 16).unsqueeze(0) will be used. (default is None)
    optional_params: dict, optional
        optional params to pass to the model forward. If model forward does not contain one of the other_params, it will be
        ignored. (default is None)
    logits_getter_fn: Callable, optional
        function which given the output of forward, will return the logits as a torch.Tensor
    device: int or str, optional
        the device to use (default is cpu)

    Returns
    -------
    list(float)
        list of floats denoting the signature of the model given the input
    """
    model.eval()

    cuda_available = torch.cuda.is_available()

    def run_forward(inp, optional_params):
        if inp is None:
            inp = torch.arange(16).unsqueeze(0).to(device)
        else:
            inp = inp.to(device)

        if not optional_params:
            optional_params = {}

        if isinstance(params, list):
            inps = {p: inp for p in params}
            p = model(**inps, **optional_params)
        else:
            inps = [inp] * params
            p = model(*inps, **optional_params)

        if logits_getter_fn:
            p = logits_getter_fn(p)

        return p

    # If cuda is available, we want to always create a signature using fp32, so this will ensure that.
    # This fix was added because signatures could not be verified on cpu vs gpu for fp32 vs tf32 with 1e-3 allowance.
    if cuda_available:
        original_matmul_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        # running in a try here in case something fails, we want to make sure to reset the precision
        try:
            p = run_forward(inp, optional_params)
        finally:
            torch.set_float32_matmul_precision(original_matmul_precision)
    else:
        p = run_forward(inp, optional_params)

    s = p.max(2)[0] - p.min(2)[0] if p.dim() >= 3 else p
    return (s.squeeze() - s.min()).tolist()


def compare_model_signatures(
    model_params_1: ModelSignatureParams,
    model_params_2: ModelSignatureParams,
    atol: float = 1e-3,
    rtol: float = 1e-5,
):
    """This utility function will compare the signature between 2 models using np.allclose

    Parameters
    ----------

    model_params_1: ModelSignatureParam
        set of params to generate first signature

    model_params_2: ModelSignatureParam
        set of params to generate second signature

    atol: float, optional
        The absolute tolerance (default is 1e-3)
    """
    model_params_1.model.eval()
    model_params_2.model.eval()
    signature = get_signature(
        model_params_1.model,
        params=model_params_1.params,
        optional_params=model_params_1.other_params,
        logits_getter_fn=model_params_1.logits_getter_fn,
        inp=model_params_1.inp,
    )
    signature2 = get_signature(
        model_params_2.model,
        params=model_params_2.params,
        optional_params=model_params_2.other_params,
        logits_getter_fn=model_params_2.logits_getter_fn,
        inp=model_params_2.inp,
    )

    signature = np.array(signature)
    signature2 = np.array(signature2)
    result_text = f"\nabs mean: {np.mean(np.abs(signature2 - signature))}\nsignature 1: {signature}\nsignature 2: {signature2}"
    assert np.allclose(signature, signature2, atol=atol, rtol=rtol), result_text
