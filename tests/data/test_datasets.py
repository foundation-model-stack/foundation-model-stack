from fms.datasets.text import CausalTextDatasetFromString
from fms.datasets.instructions import JsonInstructions
from fms.utils import tokenizers
import tempfile
import torch


sample_json = """[{
"instruction": "a question",
"output": "an answer"
}]"""


def test_instructions_dataset():
    tokenizer = tokenizers.get_tokenizer(tokenizers.char_tokenizer)
    with tempfile.NamedTemporaryFile(mode="w+t") as file:
        file.writelines(sample_json)
        file.seek(0)
        instructions = JsonInstructions(
            file.name, tokenizer, bos_tok_id=1, eos_tok_id=2
        )
        input, label = instructions[0]
        assert input[0] == 1
        assert label[len(label) - 1] == 2
        assert label[0] == -100


def test_text_dataset():
    text = "a" * 1000
    tokenizer = tokenizers.get_tokenizer(tokenizers.char_tokenizer)
    ds = CausalTextDatasetFromString(text, tokenizer, seq_len=99, pad_token="b")
    assert len(ds) == 11
    first_input, _ = ds[0]
    last_input, last_label = ds[10]
    assert last_input[0] == tokenizer.convert_tokens_to_ids("b")[0]
    assert last_label[0] == -100
    torch.testing.assert_close(
        first_input, torch.tensor(tokenizer.convert_tokens_to_ids(["a"] * 99))
    )

    ds = CausalTextDatasetFromString(text, tokenizer, seq_len=99)
    assert len(ds) == 10
