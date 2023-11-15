from torch.utils.data import Dataset
from typing import Callable, Mapping, Optional
from fms.datasets.instructions import JsonInstructions
from fms.datasets import text
from fms.utils.tokenizers import BaseTokenizer

__dataset_factory: Mapping[str, Callable[[str, BaseTokenizer], Dataset] | type] = {
    "instruction": JsonInstructions,
    "text": text.causaltext,
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
