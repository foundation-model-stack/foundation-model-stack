import abc
import os
from typing import List

import pytest


def _case_paths(test_name: str, common_tests_path: str) -> List[str]:
    abs_path = os.path.join(os.path.dirname(__file__), common_tests_path, test_name)

    return [os.path.join(abs_path, x) for x in os.listdir(abs_path)]


def resource_path_fixture(
    test_name: str,
    prefix: str,
    common_tests_path: str = "../../../tests/resources/models",
):
    """
    Pytest Fixture which will find all files under thr resources/models directory within the folder directory_name

    Parameters
    ----------
    test_name: str
        name of directory to produce test cases for
    prefix: str
        output prefix id for pytest fixture
    common_tests_path: str
        name of directory containing <test_name> (default is "../../../tests/resources/models")

    Returns
    -------
    pytest.fixture
        a pytest fixture which will find all resources under resources/model/<directory_name>
    """
    return pytest.fixture(
        params=_case_paths(test_name, common_tests_path),
        ids=lambda path: _test_ids(path, prefix),
    )


def _test_ids(path, prefix):
    return f"({prefix}: {os.path.basename(os.path.normpath(path))})"


class AbstractResourcePath(metaclass=abc.ABCMeta):
    @pytest.fixture
    @abc.abstractmethod
    def cases(self, request):
        pass
