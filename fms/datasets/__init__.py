from typing import Any, Callable, List, Mapping, Optional

from torch.utils.data import Dataset, IterableDataset

from fms.datasets import arrow, text
from fms.datasets.instructions import JsonInstructions
from fms.utils.tokenizers import BaseTokenizer


def _arrow_ds_generator(data, tokenizer, **kwargs):
    return arrow.ArrowFilesDataset(data, **kwargs)


__dataset_factory: Mapping[str, Callable[[str, BaseTokenizer], Dataset] | type] = {
    "instruction": JsonInstructions,
    "text": text.causaltext,
    "arrow": _arrow_ds_generator,
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
