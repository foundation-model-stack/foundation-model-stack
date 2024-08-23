import sys

import torch


def test_triton_import():
    [sys.modules.pop(k) for k in sys.modules.keys() if "fms.triton" in k]
    from fms.models import get_model

    fms_triton_modules = [k for k in sys.modules.keys() if "fms.triton." in k]

    # The only fms.triton module allowed to be loaded in general should be pytorch ops
    assert "fms.triton.pytorch_ops" in fms_triton_modules
    assert len(fms_triton_modules) == 1
