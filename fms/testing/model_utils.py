import inspect
from typing import Union, List, Optional, Callable
import torch


def get_signature(
    model,
    params: Union[int, List[str]] = 1,
    inp: Optional[torch.LongTensor] = None,
    optional_params: Optional[dict] = None,
    logits_getter_fn: Optional[Callable] = None,
    device: Union[int, str] = "cpu",
) -> List[float]:
    """Takes a model, and the number of inputs it expects in a forward pass. Returns a compressed signature
    that acts as an effective hash for the model, allowing for correctness checking

    Note: signatures will always be created with fp32 precision

    Parameters
    ----------
    model: nn.Module
        the model to use to produce a signature
    params: int or list(str), optional
        the params to set to the default tensor value (inp). If an integer, will use *args, if a dict, will use **kwargs (default is 1)
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

    # If cuda is available, we want to always create a signature using fp32, so this will ensure that.
    # This fix was added because signatures could not be verified on cpu vs gpu for fp32 vs tf32 with 1e-3 allowance.
    if cuda_available:
        original_matmul_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")

    if inp is None:
        inp = torch.arange(16).unsqueeze(0).to(device)
    else:
        inp = inp.to(device)

    if not optional_params:
        optional_params = {}

    all_forward_params = inspect.signature(model.forward).parameters
    params_to_ignore = []
    for k, v in optional_params.items():
        if k in all_forward_params:
            optional_params[k] = v
        else:
            params_to_ignore.append(k)

    if len(params_to_ignore) != 0:
        print(
            f"the following params were ignored as they did not exist in the forward function: {params_to_ignore}"
        )

    if isinstance(params, list):
        inps = {p: inp for p in params}
        p = model(**inps, **optional_params)
    else:
        inps = [inp] * params
        p = model(*inps, **optional_params)

    if logits_getter_fn:
        p = logits_getter_fn(p)

    # Temporary dummy backward pass to avoid checkpointing problems (see issue #591)
    p.abs().mean().mul(0).backward()

    # make sure to set the allow_tf32 to its original value
    if cuda_available:
        torch.set_float32_matmul_precision(original_matmul_precision)

    s = p.max(2)[0] - p.min(2)[0]
    return (s.squeeze() - s.min()).tolist()
