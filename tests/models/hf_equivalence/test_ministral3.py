from datetime import datetime, timedelta
import os
import pytest
import torch
import requests
import warnings

from fms.models import get_model
from fms.utils.generation import generate, pad_input_ids

from packaging.version import Version

device = "cpu"

def _get_inputs(processor, model_path):
    from PIL import Image

    # Load system prompt else, error out to make sure we test with right system prompt
    url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/europe.png"
    images = [Image.open(requests.get(url, stream=True).raw)]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is this? Answer in one sentence.",
                },
                {"type": "image"},
            ],
        },
    ]
    # Apply chat template and process inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text, images=images, return_tensors="pt").to(
        device
    )
    return inputs


def _get_hf_model_output(model_path, inputs, max_new_tokens=6):
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, use_cache=True, do_sample=False
        )
    return output


def _get_fms_model_output(model_path, inputs, max_new_tokens=6):
    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.bfloat16,
        device_type=device,
    )
    model.eval()
    torch.set_grad_enabled(False)

    inputs["only_last_token"] = True
    inputs["attn_name"] = "sdpa_causal"
    input_ids = inputs.pop("input_ids")
    input_ids, padding_kwargs = pad_input_ids(input_ids, min_pad_length=0)
    inputs["mask"] = padding_kwargs["mask"].to(device)
    inputs["position_ids"] = padding_kwargs["position_ids"].to(device)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output = generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            max_seq_len=model.config.text_config.max_expected_seq_len,
            extra_kwargs=inputs,
            prepare_model_inputs_hook=model.prepare_inputs_for_generation,
        )

    return output


@pytest.mark.slow
def test_ministral3_8b_equivalence():
    from transformers import __version__ as tf_version
    from transformers import AutoProcessor

    if Version(tf_version) < Version("5.0.0"):
        warnings.warn(f"This test requires transformers version > 5.0.0. Installed version {tf_version}. Skipping this test!")
        return

    # for now, this test won't be run, but it has been verified
    # if you would like to try this, set model_path to the HF model path
    # for ministral-3

    model_path = "/path/to/mistralai/Ministral-3-14B-Reasoning-2512/"
    # NOTE: Ministral-3-8B-Instruct-2512-BF16 model doesn't come with its own processor
    # You can use mistralai/Ministral-3-14B-Reasoning-2512 in that case

    # model_path = ""
    processor = AutoProcessor.from_pretrained(model_path)

    # Get inputs with the model path for system prompt loading
    inputs = _get_inputs(processor, model_path)

    hf_model_output = _get_hf_model_output(model_path, inputs)
    fms_model_output = _get_fms_model_output(model_path, inputs)

    # Expected result: `This is a map of Europe`
    torch.testing.assert_close(fms_model_output, hf_model_output)


if __name__ == "__main__":
    test_ministral3_8b_equivalence()
