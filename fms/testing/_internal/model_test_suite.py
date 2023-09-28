import abc
import json
import os
import tempfile
from typing import Type, List

import numpy as np
import pytest
import torch
import torch.nn as nn

from fms.testing._internal.test_resource_utils import AbstractResourcePath
from fms.testing.comparison import get_signature
from fms.utils.config import ModelConfig

_FAILED_MODEL_WEIGHTS_LOAD_MSG = """
Failed to load the state dict of the model that was stored in the test case resources for the following reason:

1. named parameter change in the underlying architecture. 

If (1), then please re-run fms.tests.models.generate_small_model_tests with --generate_weights --generate_signature

Please provide a justification for re-running the generate_small_model_tests in a PR
"""

_FAILED_CONFIG_LOAD_MSG = """
Failed to load the configuration that was stored in the test case resources for the following reason:

1. configuration parameters have changed 

If (1), then please re-run fms.tests.models.generate_small_model_tests with --generate_config

Please provide a justification for re-running the generate_small_model_tests in a PR
"""

_FAILED_MODEL_SIGNATURE_OUTPUT_MSG = """
Failed consistency of signature. This could fail for one of 2 reasons:

1. either there was a change in the model architecture which caused a difference in model output
2. a bug was fixed which is causing a different model output and that is expected

If (2) then please re-run fms.tests.models.generate_small_model_tests with --generate_weights --generate_signature

Please provide a justification for re-running generate_small_model_tests in a PR
"""


class ConfigFixtureMixin(metaclass=abc.ABCMeta):
    """Mix this in with another AbstractResourcePath testing class to include the config and config_class fixtures"""

    @pytest.fixture(autouse=True, scope="class")
    def config(
        self, resource_path: str, config_class: Type[ModelConfig]
    ) -> ModelConfig:
        """
        get the config stored in the test case directory

        Parameters
        ----------
        resource_path: str
            path to the specific test case directory specified in resource_path fixture
        config_class: Type[ModelConfig]
            the config class type

        Returns
        -------
        ModelConfig
            the config from the test case directory to be tested
        """
        config_path = os.path.join(resource_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as reader:
            text = reader.read()
        json_dict = json.loads(text)
        try:
            config = config_class(**json_dict)
        except RuntimeError:
            raise RuntimeError(_FAILED_CONFIG_LOAD_MSG)
        return config

    @pytest.fixture(scope="class", autouse=True)
    def config_class(self) -> Type[ModelConfig]:
        """
        Returns
        -------
        Type[ModelConfig]
            the config class type which will be tested
        """
        return self._config_class

    @property
    @abc.abstractmethod
    def _config_class(self) -> Type[ModelConfig]:
        """
        Returns
        -------
        Type[ModelConfig]
            the config class type which will be tested
        """
        pass


class ModelFixtureMixin(metaclass=abc.ABCMeta):
    """Mix this in with another AbstractResourcePath testing class to include the model and model_class fixtures"""

    @pytest.fixture(scope="class", autouse=True)
    def model(
        self, resource_path: str, config: ModelConfig, model_class: Type[nn.Module]
    ) -> nn.Module:
        """
        get the model stored in the test case directory

        Parameters
        ----------
        resource_path: str
            path to the specific test case directory specified in resource_path fixture
        config: ModelConfig
            the model config associated with this test case
        model_class: Type[nn.Module]
            the model class type

        Returns
        -------
        nn.Module
            the model from the test case directory to be tested
        """
        model = model_class(config)
        try:
            model.load_state_dict(
                torch.load(os.path.join(resource_path, "model_state.pth"))
            )
        except RuntimeError:
            raise RuntimeError(_FAILED_MODEL_WEIGHTS_LOAD_MSG)
        return model

    @pytest.fixture(scope="class", autouse=True)
    def model_class(self) -> Type[nn.Module]:
        """
        Returns
        -------
        Type[nn.Module]
            the model class type which will be tested
        """
        return self._model_class

    @property
    @abc.abstractmethod
    def _model_class(self) -> Type[nn.Module]:
        """
        Returns
        -------
        Type[nn.Module]
            the model class type which will be tested
        """
        pass

    @property
    @abc.abstractmethod
    def _forward_parameters(self) -> int:
        """get the number of parameters required to run a forward pass

        Note: In most cases with FMS models:
        decoder-only - 1
        encoder-only - 1
        encoder-decoder - 2

        Returns
        -------
        int
            the number of parameters required to run a forward pass.
        """
        pass


class ModelConfigTestSuite(AbstractResourcePath, ConfigFixtureMixin, ModelFixtureMixin):
    """
    This is a test suite which will test ModelConfigs and how they interact with the specific Model Architecture

    This suite will specifically test:

    - test failure when a config value isn’t serializable
    - test model construction with just arguments constructs the proper configuration
    - test model construction via a configuration
    """

    # common tests

    def test_config_round_trip(self, config):
        """Test that the config can save and load properly without serialization/deserialization issues"""

        with tempfile.TemporaryDirectory() as workdir:
            config_path = f"{workdir}/config.json"
            config.save(config_path)
            try:
                config_loaded = type(config).load(config_path)
            except RuntimeError:
                raise RuntimeError(_FAILED_CONFIG_LOAD_MSG)
            assert config.as_dict() == config_loaded.as_dict()

    def test_config_params_passed_as_kwargs_to_model(self, model, config):
        """test model construction with just arguments constructs the proper configuration"""
        params = config.as_dict()
        config_from_params = type(config)(**params)
        model_from_params = type(model)(**params)
        assert model_from_params.get_config().as_dict() == config_from_params.as_dict()

    def test_config_passed_to_model(self, model, config):
        """test model construction via a configuration"""
        model = type(model)(config)
        assert model.get_config().as_dict() == config.as_dict()


class ModelConsistencyTestSuite(
    AbstractResourcePath, ConfigFixtureMixin, ModelFixtureMixin
):
    """All tests related to model consistency will be part of this mixin"""

    @pytest.fixture(scope="class", autouse=True)
    def signature(self, resource_path) -> List[float]:
        """retrieve the signature from the test case directory

        Parameters
        ----------
        resource_path: str
            path to the specific test case directory specified in resource_path fixture

        Returns
        -------
        List[float]
            the signature stored in the test case directory that was created when generate_small_model_tests was called
            for this specific test model
        """
        return torch.load(os.path.join(resource_path, "signature.pth"))

    def test_model_output(self, model, signature):
        """test consistency of model output with signature"""

        actual = get_signature(model, params=self._forward_parameters)
        assert np.allclose(
            np.array(actual), np.array(signature)
        ), _FAILED_MODEL_SIGNATURE_OUTPUT_MSG
