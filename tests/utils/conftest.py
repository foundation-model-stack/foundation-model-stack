# content of conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--autogptq",
        action="store_true",
        default=False,
        help="run tests requiring AutoGPTQ package (with GPU support)"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "autogptq: mark test requiring AutoGPTQ")


def pytest_collection_modifyitems(config, items):
    skip_autogptq = pytest.mark.skip(reason="need --autogptq option to run")
    if not config.getoption("--autogptq"):
        for item in items:
            if "autogptq" in item.keywords:
                item.add_marker(skip_autogptq)
