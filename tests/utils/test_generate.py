import torch
import torch.nn.functional as F

from fms.utils.generation import generate
from fms.utils.tokenizers import get_tokenizer


class ModelMock:
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

    result = generate(_model_mock, ids, max_new_tokens=5, do_sample=True, temperature=0.01)
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
    result = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result[0]))
    assert result == "ABCDEFGHIJ"

    result = generate(_model_mock, ids, max_new_tokens=5, do_sample=True, temperature=0.01)
    assert torch.allclose(result[0], result[1])
    result = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result[0]))
    assert result == "ABCDEFGHIJ"
