import abc
import os
from typing import List

import pytest
from _pytest.fixtures import FixtureFunction, FixtureRequest


def _case_paths(test_name: str, common_tests_path: str) -> List[str]:
    abs_path = os.path.join(common_tests_path, test_name)

    return [os.path.join(abs_path, x) for x in os.listdir(abs_path)]


RESOURCE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "tests", "resources"
)


def resource_path_fixture(
    test_name: str,
    prefix: str,
    common_tests_path: str = f"{RESOURCE_PATH}/models",
) -> FixtureFunction:
    """
    Pytest Fixture which will find all files under the common_tests_path directory within the folder test_name. If
    using this fixture, the number of tests created is exactly equal to the number of files directly within the
    test_name folder

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
    FixtureFunction
        a pytest fixture which will find all test case resource folders under <common_tests_path>/<test_name>

    Examples
    --------

    >>> @resource_path_fixture(test_name="my_test_case", prefix="my file test case", common_tests_path="/path/to/tests/resources/some_tests")
    >>> def resource_path(self, request):
    >>>     return request.param

    >>> @pytest.fixture
    >>> def my_resource(resource_path):
    >>>     # this path is now referring to a file in /path/to/tests/resources/some_tests/<folder_i>/my_file.json where there exists multiple folders in some_tests
    >>>     my_file_path = os.path.join(resource_path, "my_file.json")
    """
    return pytest.fixture(
        params=_case_paths(test_name, common_tests_path),
        ids=lambda path: _test_ids(path, prefix),
        scope="class",
        autouse=True,
    )


def _test_ids(path, prefix):
    return f"({prefix}: {os.path.basename(os.path.normpath(path))})"


class AbstractResourcePath(metaclass=abc.ABCMeta):
    """A simple abstract class which provides a contract to a specific test case resource path to its extending classes"""

    @pytest.fixture
    @abc.abstractmethod
    def resource_path(self, request: FixtureRequest) -> str:
        """get the directory containing the files to be tested

        Parameters
        ----------
        request: FixtureRequest
            a special fixture providing information of the requesting test function

        Returns
        -------
        str
            the absolute path to the directory containing the files to be tested
        Examples
        --------

        >>> @resource_path_fixture(test_name="llama", prefix="model")
        >>> def resource_path(self, request: FixtureRequest):
        >>>     return request.param
        """
        pass
