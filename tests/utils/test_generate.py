import pytest
import torch
import torch.nn.functional as F
from torch import nn

from fms.models import get_model
from fms.utils.generation import generate, truncate_after_eos
from fms.utils.tokenizers import get_tokenizer


class ModelMock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, **kwargs):
        results = []
        for i in range(inputs.shape[0]):
            results.append(self.forward_one(inputs[i], **kwargs))
        return torch.stack(results, 0)

    def forward_one(self, inputs, **kwargs):
        inputs = inputs.view(inputs.numel())
        inputs = torch.cat((inputs[1:], torch.tensor([inputs[-1] + 1])), -1)
        inputs[inputs > 255] = 0
        inputs = F.one_hot(inputs, 256).float()
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


def test_batched_homogeneous():
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


def test_batched_heterogeneous():
    torch.set_grad_enabled(False)
    _model_mock = get_model("gpt_bigcode", "micro")
    _model_mock.reset_parameters()
    _model_mock.eval()
    tokenizer = get_tokenizer("char_tokenizer")
    first = torch.tensor(
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize("ABCDE")), dtype=torch.long
    )
    second = torch.tensor(
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize("CDEFGHIJKL")),
        dtype=torch.long,
    )
    ids = [first, second]
    # use_cache=False
    result = generate(_model_mock, ids, max_new_tokens=5, do_sample=False)
    result1_batched = result[0]
    result2_batched = result[1]

    result1 = generate(_model_mock, first, max_new_tokens=5, do_sample=False)
    torch.testing.assert_close(
        result1, result1_batched[second.size(0) - first.size(0) :]
    )

    result2 = generate(_model_mock, second, max_new_tokens=5, do_sample=False)
    torch.testing.assert_close(result2, result2_batched)
    # use_cache=True
    result = generate(
        _model_mock, ids, max_new_tokens=5, do_sample=False, use_cache=True
    )
    result1_batched = result[0]
    result2_batched = result[1]

    result1 = generate(
        _model_mock, first, max_new_tokens=5, do_sample=False, use_cache=True
    )
    torch.testing.assert_close(
        result1, result1_batched[second.size(0) - first.size(0) :]
    )

    result2 = generate(
        _model_mock, second, max_new_tokens=5, do_sample=False, use_cache=True
    )
    torch.testing.assert_close(result2, result2_batched)


def test_truncate():
    result = torch.ones(20)
    result[10] = 5
    result = truncate_after_eos(result, 5)
    expected = torch.ones(11)
    expected[10] = 5
    torch.testing.assert_close(result, expected)
