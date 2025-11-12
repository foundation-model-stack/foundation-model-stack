import copy
import inspect
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="ModelConfig")


@dataclass
class ModelConfig:
    @classmethod
    def load(cls, json_file: Union[str, os.PathLike]) -> "ModelConfig":
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        json_dict = json.loads(text)

        return cls(
            **{
                k: v
                for k, v in json_dict.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def as_dict(self) -> dict:
        return asdict(self)

    def save(self, file_path: Union[str, os.PathLike]):
        with open(file_path, "w") as f:
            json.dump(self.as_dict(), f)

    def updated(self: T, **kwargs) -> T:
        """Clone this ModelConfig and override the parameters of the ModelConfig specified by kwargs

        Note: This will always return a deep copy

        Parameters
        ----------
        kwargs
            all possibly ModelConfig dataclass named parameters to override

        Note: To override parameters of nested ModelConfig a dict can be passed, e.g. the following kwargs
        text_config = {'head_dim': 128}
        would update the `head_dim` parameter of `text_config`

        Returns
        -------
        ModelConfig
            a new instance of ModelConfig with the parameters overridden
        """
        # create a deep copy as we don't want to modify this reference
        copied_config = copy.deepcopy(self)
        unknown_params = []
        for k, v in kwargs.items():
            if hasattr(copied_config, k):
                # For multi-model hierarchical configs, ModelConfig are nested
                # i.e. ModelConfig may contain sub ModelConfig(s).
                # A dict passed as kwarg can update parameter(s) in the nested hierarchy
                # for this the key must be an existing sub config of ModelConfig and the value must be a dict
                # Since we are recursively calling `updated` method it is possible to have sub-configs in sub-configs
                if isinstance(getattr(copied_config, k), ModelConfig) and isinstance(
                    v, dict
                ):
                    sub_config = getattr(copied_config, k)
                    setattr(copied_config, k, sub_config.updated(**v))
                else:
                    setattr(copied_config, k, v)
            else:
                unknown_params.append(k)
        if len(unknown_params) > 0:
            logger.info(
                f"""Found the following unknown parameters while cloning and updating the configuration: {unknown_params}"""
            )
        return copied_config
