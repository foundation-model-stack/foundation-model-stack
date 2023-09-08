import abc
import os
import tempfile
from typing import Type, List

import numpy as np
import pytest
import torch
import torch.nn as nn

from fm.utils import utils

from fms.testing.comparison import ModelSignatureParams, compare_model_signatures
from fms.testing._internal.common_config import AbstractConfigTest
from fms.utils.config import ModelConfig

_FAILED_MODEL_WEIGHTS_LOAD_MSG = """
Failed to load the state dict of the model that was stored in the test case resources for the following reason:

1. named parameter change in the underlying architecture. 

If (1), then please re-run fms.tests.models.generate_small_model_tests with --generate_weights --generate_signature

Please provide a justification for re-running the generate_small_model_tests in a PR
"""

_FAILED_MODEL_SIGNATURE_OUTPUT_MSG = """
Failed consistency of signature. This could fail for one of 2 reasons:

1. either there was a change in the model architecture which caused a difference in model output
2. a bug was fixed which is causing a different model output and that is expected

If (2) then please re-run fms.tests.models.generate_small_model_tests with --generate_weights --generate_signature

Please provide a justification for re-running generate_small_model_tests in a PR
"""


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


class AbstractModelTest(AbstractConfigTest, ModelFixtureMixin):
    """General model testing class for future use with other models"""

    # abstract methods
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

    # class specific fixtures
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

    # common tests
    def test_model_round_trip(self, model, config):
        """Test that the model can save and load properly (config and model)"""
        model_from_config = type(model)(config)
        assert model_from_config.config.as_dict() == config.as_dict()

        with tempfile.TemporaryDirectory() as workdir:
            config_path = f"{workdir}/config.json"
            model_from_config.config.save(config_path)
            config_loaded = type(config).load(config_path)
            assert config.as_dict() == config_loaded.as_dict()

        model2 = type(model)(config_loaded)
        # ensure params are same
        model2.load_state_dict(model.state_dict())
        compare_model_signatures(
            ModelSignatureParams(model, self._forward_parameters),
            ModelSignatureParams(model2, self._forward_parameters),
        )

    def test_model_output(self, model, signature):
        """test consistency of model output with signature"""
        actual = utils.get_signature(model, params=self._forward_parameters)
        assert np.allclose(
            np.array(actual), np.array(signature)
        ), _FAILED_MODEL_SIGNATURE_OUTPUT_MSG

    def test_get_config(self, model, config):
        """test get_config method works as intended and returns the right config"""
        assert model.get_config().as_dict() == config.as_dict()
