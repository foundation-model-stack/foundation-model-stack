import abc
import json
import os
import tempfile
from typing import Type

import pytest

from fms.utils.config import ModelConfig
from tests.models.base import ModelBase


_FAILED_CONFIG_LOAD_MSG = """
Failed to load the configuration that was stored in the test case resources for the following reason:

1. configuration parameters have changed 

If (1), then please re-run fm_nlp.tests.architecture.generate_small_model_tests with --generate_config

Please provide a justification for re-running the generate_small_model_tests in a PR
"""


class AbstractConfigTest(ModelBase):
    """General config testing class for future use with other models"""

    @property
    @abc.abstractmethod
    def _config_class(self) -> Type[ModelConfig]:
        pass

    @pytest.fixture
    def config(self, cases) -> ModelConfig:
        config_path = os.path.join(cases, "config.json")
        with open(config_path, "r", encoding="utf-8") as reader:
            text = reader.read()
        json_dict = json.loads(text)
        try:
            config = self._config_class(**json_dict)
        except RuntimeError:
            raise RuntimeError(_FAILED_CONFIG_LOAD_MSG)
        return config

    def test_config_round_trip(self, config):
        """Test that the config can save and load properly"""

        with tempfile.TemporaryDirectory() as workdir:
            config_path = f"{workdir}/config.json"
            config.save(config_path)
            try:
                config_loaded = type(config).load(config_path)
            except RuntimeError:
                raise RuntimeError(_FAILED_CONFIG_LOAD_MSG)
            assert config.as_dict() == config_loaded.as_dict()

    def test_as_dict(self, config, cases):
        config_path = os.path.join(cases, "config.json")
        with open(config_path, "r", encoding="utf-8") as reader:
            text = reader.read()
        json_dict = json.loads(text)
        assert config.as_dict() == json_dict
