from transformers import AutoModel, AutoTokenizer
import pytest
import torch
import random

from fms.models import get_model
from fms.utils import tokenizers
from fms.utils.generation import pad_input_ids

device = "cpu"
SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)  # pytorch random seed
torch.use_deterministic_algorithms(True)

model_path = "sentence-transformers/all-mpnet-base-v2"
model_hf = AutoModel.from_pretrained(model_path).to(device)
model_fms = get_model(
    architecture="hf_pretrained",
    variant=model_path,
)


def _get_inputs():
    sentences = "This is an example sentence"
    tokenizer = tokenizers.get_tokenizer("sentence-transformers/all-mpnet-base-v2")
    encoded_input = tokenizer.tokenize(sentences)
    ids = tokenizer.convert_tokens_to_ids(encoded_input)
    ids = torch.tensor([ids], dtype=torch.long, device="cpu")
    return ids


def _get_inputs_hf():
    sentences = [
        "This is an example sentence",
        "Each sentence is converted",
        "This is random sentence",
        "This is a mpnet test",
    ]
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    encoded_input = tokenizer(sentences, return_tensors="pt", padding=True)

    input_ids, padding_kwargs = pad_input_ids(
        encoded_input["input_ids"],
        is_causal_mask=False,
        padding_side="right",
        position_ids_offset=model_hf.config.pad_token_id + 1,
    )
    pos_ids = padding_kwargs["position_ids"]
    return encoded_input, pos_ids


def _get_inputs_fms():
    sentences = [
        "This is an example sentence",
        "Each sentence is converted",
        "This is random sentence",
        "This is a mpnet test",
    ]
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    ids = []
    for s in sentences:
        ids.append(tokenizer.encode(s, return_tensors="pt").squeeze(0))
    inputs, extra_generation_kwargs = pad_input_ids(
        ids,
        is_causal_mask=False,
        padding_side="right",
        position_ids_offset=model_fms.config.pad_id + 1,
    )
    return inputs, extra_generation_kwargs


def _get_hf_model_output(inputs):
    model_hf.eval()
    with torch.no_grad():
        output = model_hf(inputs)
    return output


def _get_fms_model_output(inputs):
    model_fms.eval()
    with torch.no_grad():
        output = model_fms(inputs)

    return output


def _get_hf_model_output_multi_input(inputs, position_ids):
    model_hf.eval()
    with torch.no_grad():
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        output = model_hf(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
    return output


def _get_fms_model_output_multi_input(inputs, **kwargs):
    model_fms.eval()
    with torch.no_grad():
        output = model_fms(inputs, **kwargs)

    return output


@pytest.mark.slow
def test_mpnet_v2_equivalence():
    inputs = _get_inputs()

    hf_model_output = _get_hf_model_output(inputs)
    fms_model_output = _get_fms_model_output(inputs)
    torch.testing.assert_close(fms_model_output[0], hf_model_output.last_hidden_state)

    inputs_hf, position_ids_hf = _get_inputs_hf()
    inputs_fms, kwargs_fms = _get_inputs_fms()
    hf_model_output = _get_hf_model_output_multi_input(inputs_hf, position_ids_hf)
    fms_model_output = _get_fms_model_output_multi_input(inputs_fms, **kwargs_fms)
    torch.testing.assert_close(
        fms_model_output[0][1, :-2], hf_model_output.last_hidden_state[1, :-2]
    )


if __name__ == "__main__":
    test_mpnet_v2_equivalence()
