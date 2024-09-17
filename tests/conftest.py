# content of conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--capture_expectation",
        action="store_true",
        default=False,
        help="capture the output expectation for a given test",
    )
    parser.addoption(
        "--autogptq",
        action="store_true",
        default=False,
        help="run tests requiring AutoGPTQ package (with GPU support)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "capture expectation: expectation was captured")
    config.addinivalue_line("markers", "autogptq: mark test requiring AutoGPTQ")


def pytest_generate_tests(metafunc):
    option_value = metafunc.config.option.capture_expectation
    if "capture_expectation" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("capture_expectation", [option_value])


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--autogptq"):
        skip_autogptq = pytest.mark.skip(reason="need --autogptq option to run")
        for item in items:
            if "autogptq" in item.keywords:
                item.add_marker(skip_autogptq)
