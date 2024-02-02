import contextlib
import functools
import math
import os
import re
from typing import Callable, Dict, Type, Union

import torch.distributed as dist
import torch.nn as nn
from torch import LongTensor
from torch.profiler import profile


@functools.lru_cache(maxsize=None)
def has_package(name):
    try:
        __import__(name)
    except ImportError:
        return False
    else:
        return True


def human_readable(x, n_digits=1):
    """
    Parse given int or float into human-readable string interpretation (e.g. -3480331.7 -> "-3.5 million")
    """
    assert n_digits >= 0, "n_digits must be a non-negative integer"
    assert n_digits <= 3, "n_digits > 3 is not very human-readable!"
    if x == 0:
        return "0"
    is_neg = abs(x) != x
    if is_neg:
        x *= -1
    digits = int(math.log10(x))
    suf_idx = max(0, min(4, digits // 3))
    suf = ["", " thousand", " million", " billion", " trillion"][suf_idx]
    val = [1, 1e3, 1e6, 1e9, 1e12][suf_idx]

    out = "{:." + str(n_digits) + "f}"
    if is_neg:
        x *= -1
    return out.format(x / val) + suf


def human_readable_time(inp):
    """
    Parse given second count into human-readable string interpretation (e.g. 35586.4 -> "9 hours 53 minutes")
    """
    assert inp > 0, "Second count must be positive nonzero value"
    x = round(inp)
    factor = [1, 60, 60 * 60, 60 * 60 * 24]
    i = 1
    while i < 4 and x // factor[i] != 0:
        i += 1
    i -= 1
    pref = int(x // factor[i])
    suff = round((x % factor[i]) / factor[i - 1])
    if suff == factor[i]:
        pref += 1
        suff = 0
    pref_plural = "s" if pref != 1 else ""
    suff_plural = "s" if suff != 1 else ""
    if i == 0:
        return "{:.2f} seconds".format(inp)
    elif i == 1:
        return f"{pref} minute{pref_plural} {suff} second{suff_plural}"
    elif i == 2:
        return f"{pref} hour{pref_plural} {suff} minute{suff_plural}"
    else:
        return f"{pref} day{pref_plural} {suff} hour{suff_plural}"


def get_latest(targdir, qualifier=lambda x: True):
    """Fetch the latest file or folder written to target directory, subject to name passing the qualifier fn.
    If directory is empty or nonexistent or no items qualify, return None."""
    if os.path.exists(targdir) and len(os.listdir(targdir)) > 0:
        latest = max(
            [os.path.join(targdir, x) for x in os.listdir(targdir) if qualifier(os.path.join(targdir, x))],
            key=os.path.getctime,
        )
        return os.path.join(targdir, latest)
    return None


def get_oldest(targdir, qualifier=lambda x: True):
    """Fetch the oldest file or folder written to target directory, subject to name passing the qualifier fn.
    If directory is empty or nonexistent or no items qualify, return None."""
    if os.path.exists(targdir) and len(os.listdir(targdir)) > 0:
        oldest = min(
            [os.path.join(targdir, x) for x in os.listdir(targdir) if qualifier(os.path.join(targdir, x))],
            key=os.path.getctime,
        )
        return os.path.join(targdir, oldest)
    return None


def pcount(model):
    """Count parameters of a PyTorch nn.Module model"""
    return sum([p.numel() for p in model.parameters()])


def pyarrow_idx_to_torch(data, offset=0):
    """Convert a PyArrow index array into a PyTorch tensor, with specified index offset."""
    return LongTensor([x + offset for x in data.to_pylist()])


def read_contents(data_src):
    t_str = ""
    if data_src is not None and os.path.isfile(data_src):
        with open(data_src, "r") as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                t_str += line.strip()
        fp.close()
    else:
        t_str = data_src
    return t_str


def create_list_from_css(src_str, r, fn, pre_str):
    match = re.match(r, src_str)
    assert match is not None and match.span()[1] == len(src_str), f"{src_str} not a valid comma separated string."

    out = [pre_str + fn(x.strip()) if pre_str != "" else fn(x.strip()) for x in src_str.split(",")]
    return out


def get_datasets_and_weights(args_datasets, args_weights):
    if args_datasets == "":
        args_datasets = None
    if args_weights == "":
        args_weights = None

    datasets_str = read_contents(args_datasets)
    weights_str = read_contents(args_weights)

    if datasets_str != None:
        # Regex: letters, followed by any number of [/letters], followed by a possible /, (i.e. a filepath),
        # with any number of repetitions of the same, separated by a comma and any amount of white space.
        # ("letters" here can include digits, "_" and "=")
        datasets = create_list_from_css(
            datasets_str, "^([=\w]+(\/[=\w]+)*\/?)(\s*,\s*[=\w]+(\/[=\w]+)*\/?)*", lambda x: x, "dataset="
        )
    else:
        datasets = datasets_str

    if weights_str != None:
        weights = create_list_from_css(weights_str, "^([0-9]+)(\s*,\s*[0-9]+)*$", lambda x: int(x), "")
    else:
        weights = weights_str

    if datasets != None and weights != None:
        assert len(datasets) == len(
            weights
        ), f"datasets: {datasets_str} and their sampling weights {weights_str} should have same number of entries"

    return datasets, weights


class GELUTanh(nn.GELU):
    def __init__(self):
        super().__init__(approximate="tanh")


"""
Simple dict which given an activation string, return an activation function class
"""
__ACT_2_CLS: Dict[str, Type[nn.Module]] = {
    "gelu": nn.GELU,
    "gelu-tanh": GELUTanh,
    "mish": nn.Mish,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}

"""
Simple dict which given an activation class, returns the activation string

Note: SiLU will always return swish when using this dict
"""
__CLS_2_ACT: Dict[Type[nn.Module], str] = {v: k for k, v in __ACT_2_CLS.items()}


def str_to_activation(activation_str: str) -> nn.Module:
    """Convert an activation string to an instantiated activation function

    Parameters
    ----------
    activation_str: str
        the activation key to convert

    Returns
    -------
    nn.Module
        one of nn.GELU, nn.Mish, nn.ReLU, nn.Sigmoid, nn.SiLU, nn.Tanh depending on the key given
    """
    activation_str = activation_str.lower()
    if activation_str not in __ACT_2_CLS.keys():
        raise ValueError(f"activation string must be one of {__ACT_2_CLS.keys()}")
    return __ACT_2_CLS[activation_str.lower()]()


def activation_to_str(activation: Union[Type[nn.Module], nn.Module]) -> str:
    """Convert an activation function or activation class to its string representation

    Parameters
    ----------
    activation: type(nn.Module) or nn.Module
        the activation key to convert

    Returns
    -------
    str
        one of "gelu", "gelu-tanh", "mish", "relu", "sigmoid", "silu", "swish", "tanh" depending on the key given
    """
    if not isinstance(activation, type):
        activation = type(activation)

    if activation not in (a for a in __CLS_2_ACT.keys()):
        raise TypeError(f"activation module or activation module type must be one of {__CLS_2_ACT.keys()}")
    return __CLS_2_ACT[activation]



_RANK = os.getenv("RANK")
if _RANK is not None:
    _RANK = int(_RANK)

_WORLD_SIZE = os.getenv("WORLD_SIZE")
if _WORLD_SIZE is not None:
    _WORLD_SIZE = int(_WORLD_SIZE)

_LOCAL_RANK = os.getenv("LOCAL_RANK")
if _LOCAL_RANK is not None:
    _LOCAL_RANK = int(_LOCAL_RANK)


def get_world_size() -> int:
    return _WORLD_SIZE


def get_local_rank() -> int:
    return _LOCAL_RANK


def get_rank() -> int:
    return _RANK


def setup_distributed():
    """Abstracts away a bunch of the underlying PyTorch distributed training setup"""
    options = dist.ProcessGroupNCCL.Options()
    options.is_high_priority_stream = True
    dist.init_process_group("nccl", pg_options=options)
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


def run_rank_n(func: Callable, rank: int = 0, barrier: bool = False) -> None:
    """
    wraps a method so that it only runs on a specific rank
    returns a dummy method on other ranks
    """

    # wrapper function for the rank to execute on
    def func_rank_n(*args, **kwargs):
        output = func(*args, **kwargs)
        if barrier:
            dist.barrier()
        return output

    # a dummy method that doesn't do anything
    def func_rank_other(*args, **kwargs):
        if barrier:
            dist.barrier()

    if get_rank() == rank:
        return func_rank_n
    elif get_rank() == None:
        # distributed is not initialized
        return func
    else:
        return func_rank_other


def maybe_profile(use_profiler: bool = False, **kwargs):
    if use_profiler == True:
        cm = profile(**kwargs)
    else:
        cm = contextlib.nullcontext()
    return cm


def trace_handler(p, output_path, extra_name=""):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace(f"{output_path}/trace_step{str(p.step_num)}_{extra_name}.json")
