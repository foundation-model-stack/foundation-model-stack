from PIL import Image
import pytest
import torch
import requests

from fms.models import get_model
from fms.utils.generation import encode, pad_input_ids

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

def _get_image_input(processor):
    from PIL import Image
    # NOTE - this way of invoking the processor is pretty irregular,
    # since this model bundles its own code
    tiger_image_url = "https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg"
    image = Image.open(requests.get(tiger_image_url, stream=True).raw)
    image_inputs = processor.process_images([image])
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    return image_inputs

def _get_text_input(processor):
    text = "A photo of a tiger"
    text_inputs = processor.process_queries([text])
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    return text_inputs

def _get_hf_model_output(model_path, inputs):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        _attn_implementation="sdpa",
    )

    with torch.no_grad():
        return model(**inputs)


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
        output = encode(
            model,
            input_ids,
            prepare_model_inputs_hook=model.prepare_inputs_for_generation,
            extra_kwargs=inputs,
        )

    return output

def test_granite_vision_embed_3_3_2b_equivalence_text_only():
    from transformers import AutoProcessor

    model_path = "ibm-granite/granite-vision-3.3-2b-embedding"
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    inputs = _get_text_input(processor)

    hf_model_output = _get_hf_model_output(model_path, inputs)
    fms_model_output = _get_fms_model_output(model_path, inputs)

    torch.testing.assert_close(fms_model_output, hf_model_output)

def test_granite_vision_embed_3_3_2b_image_only():
    from transformers import AutoProcessor
    # for now, this test won't be run, but it has been verified
    # if you would like to try this, set model_path to the actual model checkpoint

    model_path = "ibm-granite/granite-vision-3.3-2b-embedding"
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    inputs = _get_image_input(processor)

    hf_model_output = _get_hf_model_output(model_path, inputs)
    fms_model_output = _get_fms_model_output(model_path, inputs)
    torch.testing.assert_close(hf_model_output, fms_model_output)
