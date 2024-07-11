import random
import tempfile

import torch
from torch.utils.data import Dataset

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
    assert last_input[0] == tokenizer.convert_tokens_to_ids("b")
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


class _MockDS(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def test_restartable():
    data = [1, 2, 3, 4, 5]
    ds = _MockDS(data)
    rds = datasets.RestartableFromMapDataset(ds)
    assert len(rds) == 5
    i = iter(rds)
    assert next(i) == 1
    assert next(i) == 2
    sd = rds.state_dict()

    assert next(i) == 3
    assert next(i) == 4
    assert next(i) == 5

    rds = datasets.RestartableFromMapDataset(_MockDS(data))
    rds.load_state_dict(sd)
    assert rds.state_dict() == sd

    i = iter(rds)
    assert next(i) == 3
    assert next(i) == 4
    assert next(i) == 5


class _MockNested(Dataset, datasets.SavableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            yield item


def test_nested_restartable():
    data = [1, 2, 3, 4, 5]

    ds = _MockDS(data)
    ds = datasets.RestartableFromMapDataset(ds)
    ds = _MockNested(ds)

    i = iter(ds)
    assert next(i) == 1
    assert next(i) == 2
    sd = ds.state_dict()

    assert next(i) == 3
    assert next(i) == 4

    ds = _MockDS(data)
    ds = datasets.RestartableFromMapDataset(ds)
    ds = _MockNested(ds)
    ds.load_state_dict(sd)
    assert ds.state_dict() == sd

    i = iter(ds)
    assert next(i) == 3
    assert next(i) == 4
    assert next(i) == 5


def test_packing_ds():
    data = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    ds = _MockDS(data)
    ds = datasets.RestartableFromMapDataset(ds)
    pds = datasets.PackedSequenceDataset(ds, 2)

    i = iter(pds)
    assert next(i) == [1, 2]

    sd = pds.state_dict()

    assert next(i) == [3, 4]
    assert next(i) == [5, 6]
    assert next(i) == [7, 8]

    ds = _MockDS(data)
    ds = datasets.RestartableFromMapDataset(ds)
    pds = datasets.PackedSequenceDataset(ds, 2)

    pds.load_state_dict(sd)

    assert pds.state_dict() == sd

    i = iter(pds)
    assert next(i) == [3, 4]
    assert next(i) == [5, 6]
    assert next(i) == [7, 8]


def test_eos_bos():
    bos = 11
    eos = 12

    data = list(
        [list(range(random.randint(5, 10), random.randint(12, 15))) for x in range(10)]
    )
    ds = _MockDS(data)
    ds = datasets.WithSeparatorDataset(ds, bos_token_id=bos, eos_token_id=eos)
    i = iter(ds)
    assert next(i) == [bos] + data[0] + [eos]
    assert next(i) == [bos] + data[1] + [eos]


def test_composed_eos_bos():
    bos = 11
    eos = 12

    data = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    ds = _MockDS(data)
    ds = datasets.RestartableFromMapDataset(ds)
    separated = datasets.WithSeparatorDataset(ds, bos_token_id=bos, eos_token_id=eos)
    pds = datasets.PackedSequenceDataset(separated, 5)

    i = iter(pds)
    assert next(i) == [bos, 1, 2, 3, eos]

    sd = pds.state_dict()

    assert next(i) == [bos, 4, 5, 6, 7]
    assert next(i) == [eos, bos, 8, 9, eos]

    ds = _MockDS(data)
    ds = datasets.RestartableFromMapDataset(ds)
    separated = datasets.WithSeparatorDataset(ds, bos_token_id=bos, eos_token_id=eos)
    pds = datasets.PackedSequenceDataset(separated, 5)

    pds.load_state_dict(sd)

    assert pds.state_dict() == sd
    i = iter(pds)
    assert next(i) == [bos, 4, 5, 6, 7]
    assert next(i) == [eos, bos, 8, 9, eos]
