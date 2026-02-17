import torch
import torch.nn.functional as F
from torch import nn

from fms.models import get_model
from fms.utils.generation import (
    generate,
    pad_input_ids,
    trim_prefix,
    truncate_after_eos,
)
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
    torch.manual_seed(0)
    with torch.no_grad():
        _model_mock = get_model("gpt_bigcode", "micro")
        _model_mock.reset_parameters()
        _model_mock.eval()
        tokenizer = get_tokenizer("char_tokenizer")
        first = torch.tensor(
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize("ABCDE")),
            dtype=torch.long,
        )
        second = torch.tensor(
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize("CDEFGHIJKL")),
            dtype=torch.long,
        )

        # use_cache=False
        ids, padding_kwargs = pad_input_ids([first, second])
        result = generate(
            _model_mock,
            ids,
            max_new_tokens=5,
            do_sample=False,
            extra_kwargs=padding_kwargs,
        )
        result1_batched = result[0]
        result2_batched = result[1]

        result1 = generate(_model_mock, first, max_new_tokens=5, do_sample=False)
        torch.testing.assert_close(
            result1, result1_batched[second.size(0) - first.size(0) :]
        )

        result2 = generate(_model_mock, second, max_new_tokens=5, do_sample=False)
        torch.testing.assert_close(result2, result2_batched)

        # use_cache=True
        ids, padding_kwargs = pad_input_ids([first, second])
        result = generate(
            _model_mock,
            ids,
            max_new_tokens=5,
            do_sample=False,
            use_cache=True,
            extra_kwargs=padding_kwargs,
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


def test_pad_input_ids():
    input_ids = [
        torch.arange(1, 5, dtype=torch.long),
        torch.arange(1, 10, dtype=torch.long),
    ]

    padded_input_ids, padding_kwargs = pad_input_ids(input_ids)

    expected_input_ids = torch.tensor(
        [([0] * 5) + [i for i in range(1, 5)], [i for i in range(1, 10)]],
        dtype=torch.long,
    )

    expected_position_ids = torch.tensor(
        [([0] * 5) + [i for i in range(0, 4)], [i for i in range(0, 9)]],
        dtype=torch.long,
    )

    expected_mask = torch.tensor(
        [([1] * 5) + [0 for _ in range(0, 4)], [0 for _ in range(0, 9)]],
        dtype=torch.bool,
    )
    expected_mask = (expected_mask.unsqueeze(-1) == expected_mask.unsqueeze(-2)).tril()
    expected_mask = torch.where(expected_mask.logical_not(), -torch.inf, 0.0)

    torch.testing.assert_close(padded_input_ids, expected_input_ids)
    torch.testing.assert_close(padding_kwargs["position_ids"], expected_position_ids)
    torch.testing.assert_close(padding_kwargs["mask"], expected_mask)

    padded_input_ids, padding_kwargs = pad_input_ids(input_ids, min_pad_length=64)

    expected_input_ids = torch.tensor(
        [([0] * 60) + [i for i in range(1, 5)], ([0] * 55) + [i for i in range(1, 10)]],
        dtype=torch.long,
    )

    expected_position_ids = torch.tensor(
        [([0] * 60) + [i for i in range(0, 4)], ([0] * 55) + [i for i in range(0, 9)]],
        dtype=torch.long,
    )

    expected_mask = torch.tensor(
        [([1] * 60) + [0 for _ in range(0, 4)], ([1] * 55) + [0 for _ in range(0, 9)]],
        dtype=torch.bool,
    )
    expected_mask = (expected_mask.unsqueeze(-1) == expected_mask.unsqueeze(-2)).tril()
    expected_mask = torch.where(expected_mask.logical_not(), -torch.inf, 0.0)

    torch.testing.assert_close(padded_input_ids, expected_input_ids)
    torch.testing.assert_close(padding_kwargs["position_ids"], expected_position_ids)
    torch.testing.assert_close(padding_kwargs["mask"], expected_mask)


def test_padding_right_1d():
    input_ids = [torch.arange(1, 4, dtype=torch.long)]
    min_pad_length = 10

    left_padded_input_ids, left_padding_kwargs = pad_input_ids(
        input_ids, min_pad_length
    )
    right_padded_input_ids, right_padding_kwargs = pad_input_ids(
        input_ids, min_pad_length, padding_side="right"
    )
    assert left_padded_input_ids.shape[1] == min_pad_length
    assert right_padded_input_ids.shape[1] == min_pad_length
    assert torch.equal(
        left_padded_input_ids[:, min_pad_length - len(input_ids[0]) :],
        right_padded_input_ids[:, : len(input_ids[0])],
    )
    assert torch.equal(
        left_padding_kwargs["position_ids"][:, min_pad_length - len(input_ids[0]) :],
        right_padding_kwargs["position_ids"][:, : len(input_ids[0])],
    )
    assert torch.equal(
        left_padded_input_ids[:, min_pad_length - len(input_ids[0]) :] - 1,
        left_padding_kwargs["position_ids"][:, min_pad_length - len(input_ids[0]) :],
    )
    assert torch.equal(
        right_padded_input_ids[:, : len(input_ids[0])] - 1,
        right_padding_kwargs["position_ids"][:, : len(input_ids[0])],
    )


def test_padding_right_2d():
    input_ids = [
        torch.arange(1, 4, dtype=torch.long),
        torch.arange(1, 10, dtype=torch.long),
        torch.arange(1, 5, dtype=torch.long),
        torch.arange(1, 2, dtype=torch.long),
    ]
    min_pad_length = 12

    left_padded_input_ids, left_padding_kwargs = pad_input_ids(
        input_ids, min_pad_length
    )
    right_padded_input_ids, right_padding_kwargs = pad_input_ids(
        input_ids, min_pad_length, padding_side="right"
    )
    assert left_padded_input_ids.shape[1] == min_pad_length
    assert right_padded_input_ids.shape[1] == min_pad_length
    assert torch.sum(left_padded_input_ids == 0).int().item() == min_pad_length * len(
        input_ids
    ) - sum([len(_) for _ in input_ids])

    for batch in range(len(input_ids)):
        assert torch.sum(
            right_padded_input_ids[batch] == 0
        ).int().item() == min_pad_length - len(input_ids[batch])


def test_forward_pad_right():
    _model_mock = get_model("llama", "micro")
    tokenizer = get_tokenizer("char_tokenizer")
    prompt = "Hello"
    min_pad_length = 12
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
    ids = torch.tensor(ids)
    prompt_id_length = ids.shape[0]
    input_ids_left_padded, left_padded_kwargs = pad_input_ids([ids], min_pad_length=12)
    input_ids_right_padded, right_padded_kwargs = pad_input_ids(
        [ids], min_pad_length=min_pad_length, padding_side="right"
    )
    left_padded_output = _model_mock(input_ids_left_padded, **left_padded_kwargs)
    right_padded_output = _model_mock(input_ids_right_padded, **right_padded_kwargs)
    for seq in range(min_pad_length):
        torch.testing.assert_close(
            right_padded_output[:, seq, :],
            left_padded_output[
                :, (min_pad_length - prompt_id_length + seq) % min_pad_length, :
            ],
        )


def test_trimming():
    sentence = torch.cat((torch.zeros((10,)), torch.ones((20,))), dim=0)
    result = trim_prefix(sentence)
    torch.testing.assert_close(result, torch.ones((20,)))
    result = trim_prefix(sentence, pad_token_id=2)
    torch.testing.assert_close(result, sentence)
