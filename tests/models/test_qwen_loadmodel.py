# -*- coding: utf-8 -*-

"""
Test case for Qwen3 models support.
"""

# pylint: disable=protected-access,unused-argument,abstract-method
# pylint: disable=missing-module-docstring,disable=missing-class-docstring
# pylint: disable=missing-function-docstring,line-too-long,invalid-name
# pylint: disable=unused-import,too-few-public-methods
# pylint: disable=unknown-option-value,arguments-differ
# type: ignore

import torch
from fms.models import get_model


BEGIN_WARNING_MSG = "[WARNING] Keys from checkpoint (adapted to FMS) not copied into model: "
MODEL_PATH="/home/kurtis/tmp/models/qwen3-1.7B-old"

class Test_Qwen3:
    def test_model_load(self, capsys):
        _ = get_model(
        architecture='hf_pretrained',
        variant=None,
        model_path=MODEL_PATH,
        device_type="cpu",
        # data_type=torch.float16,
        data_type=torch.bfloat16,
        source=None,
        distributed_strategy=None,
        group=None,
        linear_config={'linear_type': 'torch_linear'},
        fused_weights=False)

        captured = capsys.readouterr()
        warnings = captured.out.startswith(BEGIN_WARNING_MSG)
        assert_msg = ""
        if warnings is True:
            warn_msg = captured.out[len(BEGIN_WARNING_MSG):]
            eol_index = 0
            try:
                eol_index = warn_msg.index("}")
                missing_str = warn_msg[1:eol_index]
                missing_keys = missing_str.replace("'", "").split(", ")
                missing_str = missing_str.replace(',', '\n')
                assert_msg = f"Found {len(missing_keys)} missing keys:\n" + (
                        missing_str)
            except ValueError:
                assert_msg = f"Expected no warnings, got: {captured.out}"
        assert warnings is False, assert_msg
