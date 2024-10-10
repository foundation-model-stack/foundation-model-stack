from typing import Any, Callable, List, Mapping, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

try:
    from fms.datasets import arrow
except:
    print('warning: arrow is not available')
from fms.datasets import text
from fms.datasets.aml import AMLDataset
from fms.datasets.instructions import JsonInstructions
from fms.datasets.sentiment import JsonCausal, JsonSentiment
from fms.utils.tokenizers import BaseTokenizer


def _arrow_ds_generator(data, tokenizer, **kwargs):
    return arrow.ArrowFilesDataset(data, **kwargs)


class MockDataset(IterableDataset):
    def __init__(self, data, tokenizer: BaseTokenizer, max_seq_len=4096):
        self.tokenizer = tokenizer
        self.data = data
        self.max_seq_len = max_seq_len
        self.last_val = 0

    def nextval(self):
        self.last_val += 1
        self.last_val = self.last_val % self.tokenizer.vocab_size()
        return self.last_val

    def __iter__(self):
        while True:
            t = torch.tensor([self.nextval() for _ in range(self.max_seq_len)])
            yield t, t


__dataset_factory: Mapping[str, Callable[[str, BaseTokenizer], Dataset]] = {
    "instruction": JsonInstructions,
    "sentiment": JsonSentiment,
    "causal": JsonCausal,
    "aml": AMLDataset,
    "text": text.causaltext,
    "arrow": _arrow_ds_generator,
    "mock": MockDataset,
}


def get_dataset(name: str, tokenizer: BaseTokenizer, data: str = "", **kwargs):
    """
    Get a dataset by type.

    Args:
    name: Which style of dataset to use. E.g. instruction, text.
    data: The data this dataset should load. E.g. a url or file system path.
    """
    name = name.lower()
    if name not in __dataset_factory:
        raise NameError(f"Dataset name should be one of {__dataset_factory.keys()}")
    ctr = __dataset_factory[name]
    return ctr(data, tokenizer, **kwargs)


# avoid circular dependency
from fms.datasets.util import (
    PackedSequenceDataset,
    RestartableFromMapDataset,
    SavableDataset,
    WithSeparatorDataset,
)
