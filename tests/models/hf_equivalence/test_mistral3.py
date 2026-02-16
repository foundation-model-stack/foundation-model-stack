from datetime import datetime, timedelta
import os
import pytest
import torch
import requests

from fms.models import get_model
from fms.utils.generation import generate, pad_input_ids

device = "cuda"


def load_system_prompt(repo_id: str, filename: str) -> str:
    """Load system prompt from model directory

    Ref: https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506#transformers
    """
    # Try to grab the system prompt if it's local, otherwise download it
    maybe_local_file = os.path.join(repo_id, filename)
    if os.path.isfile(maybe_local_file):
        file_path = maybe_local_file
    else:
        from huggingface_hub import hf_hub_download

        file_path = hf_hub_download(repo_id=repo_id, filename=filename)

    with open(file_path, "r") as file:
        system_prompt = file.read()
    today = datetime.today().strftime("%Y-%m-%d")
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    model_name = repo_id.split("/")[-1]
    return system_prompt.format(name=model_name, today=today, yesterday=yesterday)


def _get_inputs(processor, model_path):
    from PIL import Image

    # Load system prompt else, error out to make sure we test with right system prompt
    system_prompt = load_system_prompt(model_path, "SYSTEM_PROMPT.txt")
    url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/europe.png"
    images = [Image.open(requests.get(url, stream=True).raw)]

    messages = [
        {"role": "system", "content": system_prompt},
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
        device, dtype=torch.bfloat16
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
def test_mistral3_24b_equivalence():
    from transformers import AutoProcessor

    # for now, this test won't be run, but it has been verified
    # if you would like to try this, set model_path to the HF model path
    # for mistral 3.1
    #
    # Further, we use bf16 to allow this test to run on a single (large) GPU,
    # e.g., h100. As such, we only check a very short sequence for equivalence
    # in this test.
    model_path = "/path/to/Mistral-Small-3.1-24B-Instruct-2503"

    # NOTE: Mistral 3.2 doesn't have the HF processor config in the checkpoint,
    # so we use the processor from Mistral 3.1 which is compatible
    processor = AutoProcessor.from_pretrained(model_path)

    # Get inputs with the model path for system prompt loading
    inputs = _get_inputs(processor, model_path)

    hf_model_output = _get_hf_model_output(model_path, inputs)
    fms_model_output = _get_fms_model_output(model_path, inputs)

    # Expected result: `This is a map of Europe`
    torch.testing.assert_close(fms_model_output, hf_model_output)


if __name__ == "__main__":
    test_mistral3_24b_equivalence()
