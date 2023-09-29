# content of conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--capture_expectation",
        action="store_true",
        default=False,
        help="capture the output expectation for a given test",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "capture expectation: expectation was captured")


def pytest_generate_tests(metafunc):
    option_value = metafunc.config.option.capture_expectation
    if "capture_expectation" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("capture_expectation", [option_value])


def pytest_collection_modifyitems(config, items):
    if config.getoption("--capture_expectation"):
        return
