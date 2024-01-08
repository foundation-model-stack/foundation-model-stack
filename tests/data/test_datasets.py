import tempfile

import torch

from fms import datasets
from fms.datasets.instructions import JsonInstructions
from fms.datasets.text import CausalTextDatasetFromString
from fms.utils import tokenizers


sample_json = """[{
"instruction": "a question",
"output": "an answer"
}]"""


def test_instructions_dataset():
    tokenizer = tokenizers.get_tokenizer(tokenizers.char_tokenizer)
    with tempfile.NamedTemporaryFile(mode="w+t") as file:
        file.writelines(sample_json)
        file.seek(0)
        instructions = JsonInstructions(file.name, tokenizer)
        input, label = instructions[0]
        assert input[0] == 2
        assert label[len(label) - 1] == 3
        assert label[0] == -100


def test_text_dataset():
    text = "a" * 1000
    tokenizer = tokenizers.get_tokenizer(tokenizers.char_tokenizer)
    ds = CausalTextDatasetFromString(text, tokenizer, seq_len=99, pad_token="b")
    assert len(ds) == 11
    first_input, _ = ds[0]
    last_input, last_label = ds[10]
    assert last_input[0] == tokenizer.convert_tokens_to_ids("b")[0]
    expected = (["b"] * 90) + (["a"] * 9)
    torch.testing.assert_close(
        last_input, torch.tensor(tokenizer.convert_tokens_to_ids(expected))
    )
    assert last_label[0] == -100
    torch.testing.assert_close(
        first_input, torch.tensor(tokenizer.convert_tokens_to_ids(["a"] * 99))
    )

    ds = CausalTextDatasetFromString(text, tokenizer, seq_len=99)
    assert len(ds) == 10


def test_dataset_getter():
    text = "a" * 10
    with tempfile.NamedTemporaryFile(mode="w+t") as file:
        file.writelines(text)
        file.seek(0)
        result = datasets.get_dataset(
            "text", tokenizers.get_tokenizer(tokenizers.char_tokenizer), file.name
        )
        # for input, output in result:
        #     print(input, output)
        input, _ = result[0]
        assert input[0].item() == ord("a")
        assert input.shape[0] == 9
