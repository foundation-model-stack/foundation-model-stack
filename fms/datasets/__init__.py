from torch.utils.data import Dataset, IterableDataset
from typing import Callable, Mapping
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


def _state_dict_save_helper(target):
    if isinstance(target, dict):
        dict_attrs = target
    else:
        dict_attrs = target.__dict__
    result = {}
    for attr in dict_attrs:
        if isinstance(attr, str) and (not len(attr) or attr[0] == "_"):
            continue

        if isinstance(dict_attrs[attr], DatasetStateDictMixin):
            result[attr] = dict_attrs[attr].state_dict()
        # We don't serialize any dataset that doesn't have this mixin.
        elif isinstance(dict_attrs[attr], Dataset):
            raise TypeError(
                "Attempting to serialize an unsupported dataset",
                attr,
                type(dict_attrs[attr]),
            )
        elif isinstance(dict_attrs[attr], dict):
            sub_dict = _state_dict_save_helper(dict_attrs[attr])
            # format like pytorch nn.Module state dicts instead of nesting
            for sub_attr in sub_dict:
                result[f"{attr}.{sub_attr}"] = sub_dict[sub_attr]
        else:
            result[attr] = dict_attrs[attr]
    return result


def _state_dict_load_helper(target, state_dict):
    if isinstance(target, dict):
        dict_attrs = target
    else:
        dict_attrs = target.__dict__
    for attr in state_dict:
        if "." in attr:
            attr_name, sub_attr = attr.split(".", 1)
            sub_dict = {sub_attr: state_dict[attr]}
            if isinstance(dict_attrs[attr], DatasetStateDictMixin):
                # Make sure we use any overriden load_state_dict in nested datasets
                dict_attrs[attr].load_state_dict(sub_dict)
            else:
                _state_dict_load_helper(dict_attrs[attr_name], sub_dict)
        elif attr not in dict_attrs:
            raise KeyError(f"Unexpected key {attr} in state dict")
        elif isinstance(dict_attrs[attr], DatasetStateDictMixin):
            dict_attrs[attr].load_state_dict(state_dict[attr])
        elif isinstance(dict_attrs[attr], dict):
            _state_dict_load_helper(dict_attrs[attr], state_dict[attr])
        else:
            dict_attrs[attr] = state_dict[attr]


class DatasetStateDictMixin:
    """
    In large-scale pre-training, because we typically only train for only a
    single epoch, we often need to be able to retain the state of the dataset
    across restarts.
    This mixin indicates a dataset that can be serialized and deserialized.
    IterableDatasets that implement this interface are expected to be
    re-startable (pick up where they left off).

    If you need a restartable MapDataPipe, wrap it in a
    RestartableFromMapDataset.

    The default implementation may not work for your dataset, so override
    `state_dict` and/or `load_state_dict` as-needed.
    """

    def state_dict(self):
        return _state_dict_save_helper(self)

    # In cases where the instance of DataPipeStateDictMixin composes another
    # DataPipe, an explicit implementation of this function will be needed.
    # This default implementation doesn't know the type of the serialized
    # datapipe, so can't construct it.
    def load_state_dict(self, state_dict):
        _state_dict_load_helper(self, state_dict)


class RestartableFromMapDataset(DatasetStateDictMixin, IterableDataset):
    def __init__(self, map_ds):
        super().__init__()
        self._map_ds = map_ds
        self.current_index = 0

    def __iter__(self):
        for index in range(self.current_index, len(self._map_ds)):
            self.current_index = index + 1
            yield self._map_ds[index]

    def __len__(self):
        return len(self._map_ds)
