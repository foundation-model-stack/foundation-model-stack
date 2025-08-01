import pytest
import torch
import requests

from fms.models import get_model
from fms.utils.generation import generate, pad_input_ids

device = "cuda"
torch.set_default_dtype(torch.float32)


def _get_inputs(processor):
    from PIL import Image

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n<|user|>\n<image>\nWhat animal is shown in this image?\n<|assistant|>\n"
    inputs = processor(text=inputs, images=image, return_tensors="pt").to(device)
    return inputs


def _get_hf_model_output(model_path, inputs):
    from transformers import LlavaNextForConditionalGeneration

    model = LlavaNextForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=20, use_cache=True, do_sample=False
        )
    return output


def _get_fms_model_output(model_path, inputs):
    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.float32,
        device_type=device,
    )
    model.eval()

    input_ids = inputs["input_ids"]
    inputs.pop("input_ids")
    input_ids, padding_kwargs = pad_input_ids(input_ids, min_pad_length=0)
    inputs["mask"] = padding_kwargs["mask"].to(device)
    inputs["position_ids"] = padding_kwargs["position_ids"].to(device)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output = generate(
            model,
            input_ids,
            max_new_tokens=20,
            use_cache=True,
            do_sample=False,
            max_seq_len=model.config.text_config.max_expected_seq_len,
            extra_kwargs=inputs,
            prepare_model_inputs_hook=model.prepare_inputs_for_generation,
        )

    return output


@pytest.mark.slow
def test_granite_vision_3_2_2b_equivalence():
    from transformers import LlavaNextProcessor
    # for now, this test won't be run, but it has been verified
    # if you would like to try this, set model_path to the actual model checkpoint

    model_path = "/path/to/granite-vision-3.2-2b"
    processor = LlavaNextProcessor.from_pretrained(model_path)
    inputs = _get_inputs(processor)

    hf_model_output = _get_hf_model_output(model_path, inputs)
    fms_model_output = _get_fms_model_output(model_path, inputs)

    torch.testing.assert_close(fms_model_output, hf_model_output)
    print(processor.decode(hf_model_output[0], skip_special_tokens=True))


if __name__ == "__main__":
    test_granite_vision_3_2_2b_equivalence()
