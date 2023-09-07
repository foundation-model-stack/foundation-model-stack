import abc
import os
from typing import List

import pytest


def _case_paths(model_name: str) -> List[str]:
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "resources/models", model_name
    )

    return [os.path.join(model_path, x) for x in os.listdir(model_path)]


def _test_ids(path):
    return f"(model: {os.path.basename(os.path.normpath(path))})"


class ModelBase(metaclass=abc.ABCMeta):
    @pytest.fixture
    @abc.abstractmethod
    def cases(self, request):
        pass
