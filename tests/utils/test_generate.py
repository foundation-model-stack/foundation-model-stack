import torch
import torch.nn.functional as F
from torch import nn

from fms.utils.generation import generate, truncate_after_eos
from fms.utils.tokenizers import get_tokenizer


class ModelMock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask=None, **kwargs):
        results = []
        for i in range(inputs.shape[0]):
            local_inp = inputs[i]
            local_mask = mask[i] if mask is not None else mask
            results.append(self.forward_one(local_inp, mask=local_mask, **kwargs))
        return torch.stack(results, 0)

    def forward_one(self, inputs, mask=None, **kwargs):
        if mask is not None:
            # just make an assumption that the last dim is the values we want
            # this is a very simplistic version of masking that is just checking that we are properly adding pads in generation
            masked_inputs = torch.zeros((torch.count_nonzero(~mask[-1]), 256)).float()
            inputs = torch.masked_select(inputs, mask[-1])
        inputs = inputs.view(inputs.numel())
        inputs = torch.cat((inputs[1:], torch.tensor([inputs[-1] + 1])), -1)
        inputs[inputs > 255] = 0
        inputs = F.one_hot(inputs, 256).float()
        if mask is not None:
            inputs = torch.cat((masked_inputs, inputs))
        return inputs


def test_generate():
    _model_mock = ModelMock()
    tokenizer = get_tokenizer("char_tokenizer")
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("ABCDE"))
    ids = torch.tensor(ids)
    result = generate(_model_mock, ids, max_new_tokens=5, do_sample=False)
    result = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result))
    assert result == "ABCDEFGHIJ"

    result = generate(
        _model_mock, ids, max_new_tokens=5, do_sample=True, temperature=0.01
    )
    result = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result))
    assert result == "ABCDEFGHIJ"

    result = generate(_model_mock, ids, max_new_tokens=5, do_sample=True, temperature=5)
    result = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result))
    assert result != "ABCDEFGHIJ"


def test_batched():
    _model_mock = ModelMock()
    tokenizer = get_tokenizer("char_tokenizer")
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("ABCDE"))
    first = torch.tensor(ids)
    second = torch.tensor(ids)
    ids = torch.stack((first, second), dim=0)
    result = generate(_model_mock, ids, max_new_tokens=5, do_sample=False)
    assert torch.allclose(result[0], result[1])
    result = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(result[0])
    )
    assert result == "ABCDEFGHIJ"

    result = generate(
        _model_mock, ids, max_new_tokens=5, do_sample=True, temperature=0.01
    )
    assert torch.allclose(result[0], result[1])
    result = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(result[0])
    )
    assert result == "ABCDEFGHIJ"


def test_batched_jagged():
    _model_mock = ModelMock()
    tokenizer = get_tokenizer("char_tokenizer")
    prompts = ["ABCDE", "ABCDEFGH"]

    ids_batch = [
        torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt)))
        for prompt in prompts
    ]
    batch_result = generate(_model_mock, ids_batch, max_new_tokens=5, do_sample=False)
    prompt1_batch_result_str = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(batch_result[0])
    )
    prompt2_batch_result_str = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(batch_result[1])
    )
    assert prompt1_batch_result_str == "\x00\x00\x00ABCDEFGHIJ"
    assert prompt2_batch_result_str == "ABCDEFGHIJKLM"

    prompt1_standalone_result = generate(
        _model_mock, ids_batch[0], max_new_tokens=5, do_sample=False
    )
    prompt2_standalone_result = generate(
        _model_mock, ids_batch[1], max_new_tokens=5, do_sample=False
    )
    prompt1_standalone_result_str = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(prompt1_standalone_result)
    )
    prompt2_standalone_result_str = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(prompt2_standalone_result)
    )

    assert prompt1_standalone_result_str == prompt1_batch_result_str.replace("\x00", "")
    assert prompt2_standalone_result_str == prompt2_batch_result_str


def test_truncate():
    result = torch.ones(20)
    result[10] = 5
    result = truncate_after_eos(result, 5)
    expected = torch.ones(11)
    expected[10] = 5
    torch.testing.assert_close(result, expected)
