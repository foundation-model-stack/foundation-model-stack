import abc
import os
import platform
import tempfile
from typing import Optional, Union

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch._dynamo.exc import TorchDynamoException
from torch._dynamo.testing import CompileCounterWithBackend

from fms.testing.comparison import get_signature
from fms.utils.config import ModelConfig
from fms.utils.fusion import apply_unfuse_weights


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
            opt_parameter_initialized = self._maybe_get_initialized_parameter(
                key, parameter
            )
            if opt_parameter_initialized is not None:
                parameter.copy_(opt_parameter_initialized)
            else:
                values = torch.randn_like(parameter)
                values -= 0.5
                values /= 20.0
                parameter.copy_(values)
        return uninitialized_model

    def _maybe_get_initialized_parameter(
        self, key: str, parameter: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Optionally return an initialized tensor parameter. If None is returned, the default randn_like will be used
        for the given parameter

        Parameters
        ----------
        key: str
            state dict key for tensor
        parameter: torch.Tensor
            the parameter tensor associated with a key in a state dict

        Returns
        -------
        Optional[torch.Tensor]
            torch.Tensor if randomly initialized tensor for this model
            None if default initialization to be used
        """
        return None


class SignatureFixtureMixin:
    """Include this mixin if you would like to get the signature fixture for a given model test"""

    @pytest.fixture(scope="class", autouse=True)
    def model_id(self) -> Optional[str]:
        return None

    @pytest.fixture(scope="class", autouse=True)
    def signature(self, model_id: Optional[str], **kwargs) -> list[float]:
        """include this fixture to get a models signature (defaults to what is in tests/resources/expectations)"""
        return self._signature(model_id)

    def _get_expectation_path(self, test_name, model_id):
        """
        get the path to the expectation file for a given test case. The path will be formatted as follows:

        <current_dir>/../resources/expectations/<ModuleName>.<TestClass>.<model_id_if_exists>.<test_name>

        Example: module=models.test_model, test_class=MyTest, test_name=test_model_output, model_id=llama-7b

        models.test_model.MyTest.llama-7b.test_model_output
        """
        import inspect

        file_name = f"{self.__class__.__module__}.{self.__class__.__name__}"

        # adding model_id to the file name if a class using this mixin includes it
        if model_id is not None:
            file_name = f"{file_name}.{os.path.basename(model_id)}"

        expectation_file_path = os.path.join(
            os.path.dirname(inspect.getfile(self.__class__)),
            "..",
            "resources",
            "expectations",
            f"{file_name}.{test_name}",
        )
        return expectation_file_path

    def _signature(self, model_id: Optional[str]) -> list[float]:
        try:
            expectation_file = open(
                self._get_expectation_path("test_model_output", model_id)
            )
            line = expectation_file.readline()
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

    @staticmethod
    def _get_signature_logits_getter_fn(f_out) -> torch.Tensor:
        """function which given the output of forward, will return the logits as a torch.Tensor

        Returns
        -------
        Optional[torch.Tensor]
            the output logits
        """
        return f_out

    @property
    @abc.abstractmethod
    def _get_signature_params(self) -> Union[int, list[str]]:
        """the value to pass into params in get_signature function for this model

        Returns
        -------
        Union[int, list[str]]
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

            optional_params = self._get_signature_optional_params
            # default attn_algorithm won't compile on CPU for older pytorch versions
            # TODO: add non-math attn_algorithm when we have GPUs to run unit tests
            optional_params.update({"attn_algorithm": "math"})

            get_signature(
                compiled_model,
                params=self._get_signature_params,
                optional_params=optional_params,
                logits_getter_fn=self._get_signature_logits_getter_fn,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            assert cnt.frame_count == 1
        except TorchDynamoException as e:
            pytest.fail(f"Failed to get signature of full-graph compiled model:\n{e}")


class ModelConsistencyTestSuite(ModelFixtureMixin, SignatureFixtureMixin):
    """All tests related to model consistency will be part of this test suite"""

    @property
    def _get_signature_input_ids(self) -> Optional[torch.Tensor]:
        """the value to pass into inp in get_signature function for this model

        Returns
        -------
        Optional[torch.Tensor]
            the input ids
        """
        return None

    @staticmethod
    def _get_signature_logits_getter_fn(f_out) -> torch.Tensor:
        """function which given the output of forward, will return the logits as a torch.Tensor

        Returns
        -------
        Optional[torch.Tensor]
            the output logits
        """
        return f_out

    @property
    @abc.abstractmethod
    def _get_signature_params(self) -> Union[int, list[str]]:
        """the value to pass into params in get_signature function for this model

        Returns
        -------
        Union[int, list[str]]
            the params to set to the default tensor value (inp) in get_signature. If an integer, will use *args, if a
            list, will use **kwargs
        """
        pass

    @property
    def _get_signature_optional_params(self) -> Optional[dict[str, torch.Tensor]]:
        """the value to pass into optional_params in get_signature function for this model

        Returns
        -------
        Optional[dict[str, torch.Tensor]]
            the dictionary of optional params to pass to the model
        """
        return {}

    def test_model_output(self, model, signature, model_id, capture_expectation):
        """test consistency of model output with signature"""

        actual = get_signature(
            model,
            inp=self._get_signature_input_ids,
            params=self._get_signature_params,
            optional_params=self._get_signature_optional_params,
            logits_getter_fn=self._get_signature_logits_getter_fn,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        if capture_expectation:
            expectation_file_path = self._get_expectation_path(
                "test_model_output", model_id
            )
            with open(expectation_file_path, "w") as signature_file:
                signature_file.write(",".join(map(str, actual)))
            signature_file.close()
            pytest.fail(
                "Signature file has been saved, please re-run the tests without --capture_expectation"
            )
        print(actual, signature)
        assertion_msg = f"""
        difference: {np.mean(np.abs(np.array(actual) - np.array(signature)))}

        {_FAILED_MODEL_SIGNATURE_OUTPUT_MSG}
        """

        (
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(signature)),
            assertion_msg,
        )

    def test_model_weight_keys(self, model, model_id, capture_expectation):
        actual_keys = list(sorted(model.state_dict().keys()))

        expectation_file_path = self._get_expectation_path(
            "test_model_weight_keys", model_id
        )

        if capture_expectation:
            with open(expectation_file_path, "w") as weight_keys_file:
                weight_keys_file.write(",".join(map(str, actual_keys)))
            weight_keys_file.close()
            pytest.fail(
                "Weights Key file has been saved, please re-run the tests without --capture_expectation"
            )

        try:
            weight_keys_file = open(expectation_file_path)
            expected_keys = [k for k in weight_keys_file.readline().split(",")]
            assert actual_keys == expected_keys, _FAILED_MODEL_WEIGHTS_KEYS_MSG
        except OSError:
            pytest.fail(
                "Weights Key file failed to load, please re-run the tests with --capture_expectation"
            )

    def test_model_unfused(self, model, signature):
        """test unfused model output against signature"""

        unfused_model = apply_unfuse_weights(model)
        unfused_signature = get_signature(
            unfused_model,
            inp=self._get_signature_input_ids,
            params=self._get_signature_params,
            optional_params=self._get_signature_optional_params,
            logits_getter_fn=self._get_signature_logits_getter_fn,
        )

        assertion_msg = f"""
        difference: {np.mean(np.abs(np.array(unfused_signature) - np.array(signature)))}

        Output for signature of unfused model is incorrect
        """

        (
            torch.testing.assert_close(
                torch.tensor(unfused_signature), torch.tensor(signature)
            ),
            assertion_msg,
        )
