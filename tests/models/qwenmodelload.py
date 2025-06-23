#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright IBM Corp.
# SPDX-License-Identifier: Apache-2.0


"""
<description of this script>
Avoids pylint: missing-module-docstring message
"""

# pylint: disable=protected-access,unused-argument,abstract-method
# pylint: disable=missing-module-docstring,disable=missing-class-docstring
# pylint: disable=missing-function-docstring,line-too-long,invalid-name
# pylint: disable=unused-import,too-few-public-methods
# pyline: disable=unknown-option-value
# pyline: arguments-differ
# type: ignore

import pprint
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchinfo import summary
from fms.models import get_model
from fms.utils import generation, tokenizers
from fms.utils.generation import generate, pad_input_ids


MODEL_PATH="/home/kurtis/tmp/models/qwen3-1.7B-old"
SEP = '=' * 100


model = get_model(
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
    fused_weights=False,
)

print(f"{SEP}\n= Simple print of var \"model\":\n{SEP}")
print(model)
print(f"{SEP}\n= summary(..., depth=5\n{SEP}")
# summary(model, input_data=(model_inputs["input_ids"],
#                model_inputs["attention_mask"]),
#                depth=5)
print(f"{SEP}\n= for name, module in model.named_modules():\n{SEP}")
for name, module in model.named_modules():
    if len(list(module.children())) == 0:
        print(name, module)
