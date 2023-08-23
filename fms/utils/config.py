import inspect
import json
import os
from dataclasses import asdict, dataclass
from typing import Union


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

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: unknown parameter {k}")
