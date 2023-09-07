import abc
import os
import tempfile
from typing import Type

import numpy as np
import pytest
import torch
import torch.nn as nn

from fm.utils import utils

from .test_config import AbstractConfigTest
from .utils import ModelSignatureParams, compare_model_signatures


_FAILED_MODEL_WEIGHTS_LOAD_MSG = """
Failed to load the state dict of the model that was stored in the test case resources for the following reason:
            
1. named parameter change in the underlying architecture. 

If (1), then please re-run fm_nlp.tests.architecture.generate_small_model_tests with --generate_weights --generate_signature

Please provide a justifaction for re-running the generate_small_model_tests in a PR
"""

_FAILED_MODEL_SIGNATURE_OUTPUT_MSG = """
Failed consistency of signature. This could fail for one of 2 reasons:
        
1. either there was a change in the model architecture which caused a difference in model output
2. a bug was fixed which is causing a different model output and that is expected

If (2) then please re-run fm_nlp.tests.architecture.generate_small_model_tests with --generate_weights --generate_signature

Please provide a justification for re-running generate_small_model_tests in a PR
"""


class AbstractModelTest(AbstractConfigTest):
    """General model testing class for future use with other models"""

    @property
    @abc.abstractmethod
    def _model_class(self) -> Type[nn.Module]:
        pass

    @property
    @abc.abstractmethod
    def _forward_parameters(self) -> int:
        pass

    @pytest.fixture
    def model(self, cases, config) -> nn.Module:
        model = self._model_class(config)
        try:
            model.load_state_dict(torch.load(os.path.join(cases, "model_state.pth")))
        except RuntimeError:
            raise RuntimeError(_FAILED_MODEL_WEIGHTS_LOAD_MSG)
        return model

    @pytest.fixture
    def signature(self, cases) -> nn.Module:
        return torch.load(os.path.join(cases, "signature.pth"))

    def test_model_round_trip(self, model, config):
        """Test that the config can save and load properly (config and model)"""
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
        actual = utils.get_signature(model, params=self._forward_parameters)
        assert np.allclose(
            np.array(actual), np.array(signature)
        ), _FAILED_MODEL_SIGNATURE_OUTPUT_MSG

    def test_get_config(self, model, config):
        assert model.get_config().as_dict() == config.as_dict()
