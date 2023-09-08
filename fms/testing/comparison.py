import dataclasses
import inspect
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from fms.testing.model_utils import get_signature


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


def compare_model_signatures(
    model_params_1: ModelSignatureParams,
    model_params_2: ModelSignatureParams,
    atol: float = 1e-3,
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
    )
    signature2 = get_signature(
        model_params_2.model,
        params=model_params_2.params,
        optional_params=model_params_2.other_params,
        logits_getter_fn=model_params_2.logits_getter_fn,
    )

    signature = np.array(signature)
    signature2 = np.array(signature2)
    assert np.allclose(signature, signature2, atol=atol)
