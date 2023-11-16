import abc
import os
import platform
import tempfile
from typing import List, Union

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch._dynamo.exc import TorchDynamoException
from torch._dynamo.testing import CompileCounterWithBackend

from fms.testing.comparison import get_signature
from fms.utils.config import ModelConfig

_FAILED_CONFIG_LOAD_MSG = """
Failed to load the configuration. This could occur if there was a change in the configuration and the implementation of 
the ModelConfig is not accounting for it.
"""

_FAILED_MODEL_SIGNATURE_OUTPUT_MSG = """
Failed consistency of signature. This could fail for one of 2 reasons:

1. either there was a change in the model architecture which caused a difference in model output
2. a bug was fixed which is causing a different model output and that is expected

If (2) then please re-run this test with --capture_expectation
"""

_FAILED_MODEL_WEIGHTS_KEYS_MSG = """
Failed consistency of model weights. This is most likely due to: 

1. a new weight being introduced in the model
2. a weight's name changing in the model 

If either (1) or (2) was done purposely, please re-run this test with --capture_expectation
"""


class ConfigFixtureMixin(metaclass=abc.ABCMeta):
    """Include this mixin if you would like to have the config fixture"""

    @abc.abstractmethod
    @pytest.fixture(scope="class", autouse=True)
    def config(self, **kwargs) -> ModelConfig:
        """include this fixture to get a models config"""
        pass


class ModelFixtureMixin(metaclass=abc.ABCMeta):
    """Include this mixin if you would like to have the model fixture"""

    @abc.abstractmethod
    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, **kwargs) -> nn.Module:
        pass

    @pytest.fixture(scope="class", autouse=True)
    def model(self, uninitialized_model: nn.Module):
        """include this fixture to get a model that is fully initialized"""
        torch.random.manual_seed(5)
        sd = uninitialized_model.state_dict()
        params = sorted(sd.keys())
        for key in params:
            parameter = sd[key]
            values = torch.randn_like(parameter)
            values -= 0.5
            values /= 20.0
            parameter.copy_(values)
        return uninitialized_model


class SignatureFixtureMixin:
    """Include this mixin if you would like to get the signature fixture for a given model test"""

    @pytest.fixture(scope="class", autouse=True)
    def signature(self, **kwargs) -> List[float]:
        """include this fixture to get a models signature (defaults to what is in tests/resources/expectations)"""
        return self._signature()

    def _signature(self) -> List[float]:
        import inspect

        try:
            config_file = open(
                os.path.join(
                    os.path.dirname(inspect.getfile(self.__class__)),
                    "..",
                    "resources",
                    "expectations",
                    f"{self.__class__.__module__}.{self.__class__.__name__}.test_model_output",
                )
            )
            line = config_file.readline()
            return [float(v) for v in line.split(",")]
        except FileNotFoundError:
            print(
                "Signature failed to load, please re-run the tests with --capture_expectation"
            )


class ModelConfigTestSuite(ConfigFixtureMixin, ModelFixtureMixin):
    """
    This is a test suite which will test ModelConfigs and how they interact with the specific Model Architecture

    This suite will specifically test:

    - test failure when a config value isn’t serializable
    - test model construction with just arguments constructs the proper configuration
    - test model construction via a configuration
    """

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


class ModelCompileTestSuite(ModelFixtureMixin):
    """A set of tests associated with compilation of fms models"""

    @property
    @abc.abstractmethod
    def _get_signature_params(self) -> Union[int, List[str]]:
        """the value to pass into params in get_signature function for this model

        Returns
        -------
        Union[int, List[str]]
            the params to set to the default tensor value (inp) in get_signature. If an integer, will use *args, if a
            list, will use **kwargs
        """
        pass

    @pytest.mark.skipif(
        platform.system() != "Linux",
        reason=f"pytorch compile is more stable on Linux, skipping as current platform is {platform.platform()}",
    )
    def test_model_compile_no_graph_breaks(self, model):
        """Test that an FMS model is compilable without graph breaks"""
        try:
            torch._dynamo.reset()
            cnt = CompileCounterWithBackend("inductor")
            compiled_model = torch.compile(model=model, backend=cnt, fullgraph=True)
            assert cnt.frame_count == 0
            get_signature(
                compiled_model,
                params=self._get_signature_params,
                # default attn_algorithm won't compile on CPU
                # TODO: add non-mmath attn_algorithm when we have GPUs to run unit tests
                optional_params={"attn_algorithm": "math"},
            )
            assert cnt.frame_count == 1
        except TorchDynamoException as e:
            pytest.fail(f"Failed to get signature of full-graph compiled model:\n{e}")


class ModelConsistencyTestSuite(ModelFixtureMixin, SignatureFixtureMixin):
    """All tests related to model consistency will be part of this test suite"""

    @property
    @abc.abstractmethod
    def _get_signature_params(self) -> Union[int, List[str]]:
        """the value to pass into params in get_signature function for this model

        Returns
        -------
        Union[int, List[str]]
            the params to set to the default tensor value (inp) in get_signature. If an integer, will use *args, if a
            list, will use **kwargs
        """
        pass

    def test_model_output(self, model, signature, capture_expectation):
        """test consistency of model output with signature"""

        actual = get_signature(model, params=self._get_signature_params)

        if capture_expectation:
            import inspect

            to_write = os.path.join(
                os.path.dirname(inspect.getfile(self.__class__)),
                "..",
                "resources",
                "expectations",
                f"{self.__class__.__module__}.{self.__class__.__name__}.test_model_output",
            )
            with open(to_write, "w") as signature_file:
                signature_file.write(",".join(map(str, actual)))
            signature_file.close()
            pytest.fail(
                "Signature file has been saved, please re-run the tests without --capture_expectation"
            )

        assertion_msg = f"""
        difference: {np.mean(np.abs(np.array(actual) - np.array(signature)))}
        
        {_FAILED_MODEL_SIGNATURE_OUTPUT_MSG}
        """

        torch.testing.assert_close(
            torch.tensor(actual), torch.tensor(signature)
        ), assertion_msg

    def test_model_weight_keys(self, model, capture_expectation):
        import inspect

        actual_keys = list(sorted(model.state_dict().keys()))

        weight_keys_path = os.path.join(
            os.path.dirname(inspect.getfile(self.__class__)),
            "..",
            "resources",
            "expectations",
            f"{self.__class__.__module__}.{self.__class__.__name__}.test_model_weight_keys",
        )

        if capture_expectation:
            with open(weight_keys_path, "w") as weight_keys_file:
                weight_keys_file.write(",".join(map(str, actual_keys)))
            weight_keys_file.close()
            pytest.fail(
                "Weights Key file has been saved, please re-run the tests without --capture_expectation"
            )

        try:
            weight_keys_file = open(weight_keys_path)
            expected_keys = [k for k in weight_keys_file.readline().split(",")]
            assert actual_keys == expected_keys, _FAILED_MODEL_WEIGHTS_KEYS_MSG
        except:
            pytest.fail(
                "Weights Key file failed to load, please re-run the tests with --capture_expectation"
            )
