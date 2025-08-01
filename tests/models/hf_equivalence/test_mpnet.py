from transformers import AutoModel
import pytest
import torch
import random

from fms.models import get_model
from fms.utils import tokenizers

device = "cpu"
SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)  # pytorch random seed
torch.use_deterministic_algorithms(True)

def _get_inputs():
    sentences = 'This is an example sentence'
    tokenizer = tokenizers.get_tokenizer(
                "sentence-transformers/all-mpnet-base-v2")
    encoded_input = tokenizer.tokenize(sentences)
    ids = tokenizer.convert_tokens_to_ids(encoded_input)
    ids = torch.tensor([ids], dtype=torch.long, device="cpu")
    return ids


def _get_hf_model_output(model_path, inputs):
    model = AutoModel.from_pretrained(model_path).to(device)

    with torch.no_grad():
        output = model(inputs)
    return output


def _get_fms_model_output(model_path, inputs):
    model = get_model(
            architecture="hf_pretrained",
            variant=model_path,
    ) 
    with torch.no_grad():
        output = model(inputs)

    return output


@pytest.mark.slow
def test_mpnet_v2_equivalence():
    # for now, this test won't be run, but it has been verified
    # set model_path to the actual model checkpoint
    model_path = "sentence-transformers/all-mpnet-base-v2"
    inputs = _get_inputs()

    hf_model_output = _get_hf_model_output(model_path, inputs)
    fms_model_output = _get_fms_model_output(model_path, inputs)
    torch.testing.assert_close(fms_model_output[0], 
                               hf_model_output.last_hidden_state[0], 
                               atol=0.77, 
                               rtol=1e-6)


if __name__ == "__main__":
    test_mpnet_v2_equivalence()

