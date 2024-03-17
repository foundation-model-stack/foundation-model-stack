import torch
import torch.nn.functional as F
from torch import nn

from fms.utils.generation import generate, truncate_after_eos, _repetition_penalty
from fms.utils.tokenizers import get_tokenizer


class ModelMock(nn.Module):
    def __init__(self, repeat=False):
        super().__init__()
        self.repeat = repeat

    def forward(self, inputs, **kwargs):
        results = []
        for i in range(inputs.shape[0]):
            results.append(self.forward_one(inputs[i], **kwargs))
        return torch.stack(results, 0)

    def forward_one(self, inputs, **kwargs):
        inputs = inputs.view(inputs.numel())
        increment = int(not self.repeat)
        inputs = torch.cat((inputs[1:], torch.tensor([inputs[-1] + increment])), -1)
        inputs[inputs > 255] = 0
        inputs = F.one_hot(inputs, 256).float() * 10 + 5
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


def test_truncate():
    result = torch.ones(20)
    result[10] = 5
    result = truncate_after_eos(result, 5)
    expected = torch.ones(11)
    expected[10] = 5
    torch.testing.assert_close(result, expected)


def test_repeat_penalty():
    input_ids = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 7, 7], [2, 3, 3, 3, 3, 3, 3, 3, 3, 3]],
        dtype=torch.long,
    )
    scores = torch.randn((2, 10))

    # make sure we have positive and negative logits to test.
    torch.abs_(scores[0][7])
    torch.abs_(scores[1][3])
    scores[1][3] = -scores[1][3]

    penalty = 1.5
    penalized = _repetition_penalty(input_ids, scores, penalty=penalty)
    # make sure scores that had no examples weren't penalized
    assert penalized[0][8] == scores[0][8]
    assert penalized[1][0] == scores[1][0]

    assert penalized[0][7] == scores[0][7] / penalty
    assert penalized[1][3] == scores[1][3] * penalty

    penalized = _repetition_penalty(input_ids, scores, penalty=penalty, compound=True)
    assert penalized[0][8] == scores[0][8]
    assert penalized[1][0] == scores[1][0]

    assert penalized[0][7] == scores[0][7] / (penalty**3)
    assert penalized[1][3] == scores[1][3] * (penalty**9)


def test_generate_penalty():
    _model_mock = ModelMock()
    tokenizer = get_tokenizer("char_tokenizer")
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("ABCDE"))
    ids = torch.tensor(ids)

    # a very low repetition penalty should force repetitions. I.e. instead
    # of producing the expected 'F', produce something from the original input
    # string.
    result = generate(
        _model_mock, ids, max_new_tokens=10, do_sample=False, repetition_penalty=0.1
    )
    result = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result))
    for c in result:
        assert c in "ABCDE"

    _model_mock = ModelMock(repeat=True)

    # with repeat=True, mock model will generate `ABCDEEEEEE...`. repetition
    # penalty can stop from producing E's.
    # logits are 15 vs 5 in mock model so penalty < 3 won't do anything, > 3 will.
    result = generate(
        _model_mock, ids, max_new_tokens=10, do_sample=False, repetition_penalty=3.1
    )
    result = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result))
    assert result != "ABCDEEEEEEEEEEE"
    result = generate(
        _model_mock, ids, max_new_tokens=10, do_sample=False, repetition_penalty=2.9
    )
    result = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result))
    assert result == "ABCDEEEEEEEEEEE"
