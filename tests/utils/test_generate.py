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


@pytest.mark.slow
def test_cache_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("llama", "7b", device_type=device)
    ids = torch.arange(1, 16, dtype=torch.long, device=device).unsqueeze(0)
    expandable_results = generate(model, ids, do_sample=False, use_cache=True)
    no_cache_results = generate(model, ids, do_sample=False, use_cache=False)
    torch.testing.assert_allclose(expandable_results, no_cache_results)

    if torch.cuda.is_available():
        from fms.utils.cache.paged import PagedKVCacheManager

        paged_kv_cache = PagedKVCacheManager(
            model.config.nlayers,
            model.config.nheads,
            model.config.emb_dim,
            model.config.kvheads,
            total_num_gpu_blocks=100,
        )
        paged_results = generate(
            model, ids, do_sample=False, use_cache=True, kv_cache_manager=paged_kv_cache
        )
        torch.testing.assert_allclose(paged_results, no_cache_results)
