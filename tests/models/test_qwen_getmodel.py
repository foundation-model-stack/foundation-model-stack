# -*- coding: utf-8 -*-

"""
Test fms get_model() function and check no weights are not loaded
    export KWR_SKIP=1 temporarily to get this test to succeed.
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
MODEL_PATH="/home/kurtis/tmp/models/qwen3-1.7B"

class Test_Qwen3:
    """_summary_
    """
    def test_get_model(self, capsys):
        _ = get_model(
        architecture="qwen3",
        variant="1.7b",
        model_path=MODEL_PATH,
        device_type="cpu",
        # data_type=torch.float16,
        data_type=torch.bfloat16,
        distributed_strategy=None,
        group=None,
        linear_config={'linear_type': 'torch_linear'},
        fused_weights=False)

        captured = capsys.readouterr()
        outf = "./captured.out"
        with open(outf, "w", encoding='utf-8') as file:
            file.write(captured.out)
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
                # assert_msg = f"Found {len(missing_keys)} missing keys:\n" + missing_str
                assert_msg = captured.out + f"Found {len(missing_keys)} missing keys:\n" + missing_str
            except ValueError:
                assert_msg = f"Expected no warnings, got: {captured.out}"
        assert warnings is False, assert_msg
