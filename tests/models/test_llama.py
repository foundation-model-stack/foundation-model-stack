import pytest
import torch
from importlib.util import find_spec

from fms.models.llama import LLaMA, LLaMAConfig
from fms.modules import UninitializedModule
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig


class LLaMA2Fixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base LLaMA 2 Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: LLaMAConfig):
        return LLaMA(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        return LLaMAConfig(
            src_vocab_size=381,
            emb_dim=16,
            norm_eps=1e-05,
            nheads=2,
            kvheads=0,
            nlayers=2,
            pad_id=0,
            hidden_grow_factor=8 / 3,
            multiple_of=2,
            activation_fn="swish",
            p_dropout=0.0,
            max_expected_seq_len=4096,
            rope_scaling={},
            linear_config={"linear_type": "torch_linear"},
        )

    @pytest.fixture(scope="class", autouse=True)
    def model(self, uninitialized_model: torch.nn.Module):
        """include this fixture to get a model that is fully initialized"""

        # Special random seed for Llama to ensure tests pass
        torch.random.manual_seed(0)
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


class TestLlama2(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    LLaMA2Fixtures,
):
    """
    Model Test Suite for llama

    This suite will include tests for:
    - model configuration
    - basic load/save model
    - consistency of model output
    """

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config without mutating the original config"""
        model = type(model)(config=config, pad_id=config.pad_id + 1)
        # check not same reference
        assert model.get_config() is not config

        # modify pad_id to the new value expected and check equivalence
        config.pad_id = config.pad_id + 1
        assert model.get_config().as_dict() == config.as_dict()


class LLaMA2GQAFixtures(ModelFixtureMixin):
    """
    Base LLaMA 2 Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self):
        return LLaMA(
            src_vocab_size=381,
            emb_dim=16,
            norm_eps=1e-05,
            nheads=4,
            kvheads=2,
            nlayers=2,
            pad_id=0,
            hidden_grow_factor=8 / 3,
            multiple_of=2,
            activation_fn="swish",
            p_dropout=0.0,
            max_expected_seq_len=4096,
            rope_scaling={},
        )


class TestLlama2GQA(
    ModelConsistencyTestSuite, ModelCompileTestSuite, LLaMA2GQAFixtures
):
    """
    Test LLaMA2-GQA model consistency
    """

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]


class LLaMA2GPTQFixtures(ModelFixtureMixin):
    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self):
        return LLaMA(
            src_vocab_size=381,
            emb_dim=32,
            norm_eps=1e-05,
            nheads=2,
            kvheads=0,
            nlayers=2,
            pad_id=0,
            hidden_grow_factor=2.0,
            multiple_of=2,
            activation_fn="swish",
            p_dropout=0.0,
            max_expected_seq_len=4096,
            rope_scaling={},
            linear_config={"linear_type": "gptq_cpu"},
        )

    def _maybe_get_initialized_parameter(self, key, parameter):
        if "qweight" in key:
            return torch.randint(
                low=0,
                high=torch.iinfo(torch.int32).max,
                size=parameter.shape,
                dtype=torch.int32,
            )
        elif "qzeros" in key:
            return torch.ones(parameter.shape, dtype=torch.int32) * 8
        elif "g_idx" in key:
            return parameter
        else:
            return None


@pytest.mark.autogptq
class TestLlama2GPTQ(
    ModelConsistencyTestSuite, ModelCompileTestSuite, LLaMA2GPTQFixtures
):
    """
    Test LLaMA2-GPTQ model consistency
    """

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    def test_model_unfused(self, model, signature):
        pytest.skip("weight unfuse is not implemented for GPTQ")


class LLaMA3Fixtures(ModelFixtureMixin):
    """
    Base LLaMA 3 Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self):
        return LLaMA(
            src_vocab_size=381,
            emb_dim=16,
            norm_eps=1e-05,
            nheads=4,
            kvheads=2,
            nlayers=2,
            pad_id=0,
            hidden_grow_factor=3.5,
            multiple_of=2,
            activation_fn="swish",
            p_dropout=0.0,
            max_expected_seq_len=8192,
            rope_theta=500000.0,
            rope_scaling={},
        )


class TestLlama3(ModelConsistencyTestSuite, ModelCompileTestSuite, LLaMA3Fixtures):
    """
    Test LLaMA 3 model consistency
    """

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]


class LLaMA31Fixtures(ModelFixtureMixin):
    """
    Base LLaMA 3.1-3.3 Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self):
        return LLaMA(
            src_vocab_size=381,
            emb_dim=16,
            norm_eps=1e-05,
            nheads=4,
            kvheads=2,
            nlayers=2,
            pad_id=0,
            hidden_grow_factor=3.5,
            multiple_of=2,
            activation_fn="swish",
            p_dropout=0.0,
            max_expected_seq_len=131072,
            rope_theta=500000.0,
            rope_scaling={
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        )


class TestLlama31(ModelConsistencyTestSuite, ModelCompileTestSuite, LLaMA31Fixtures):
    """
    Test LLaMA 3.1 model consistency
    """

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]


def fp8_linear_type(name: str) -> str:
    if "head" in name:
        return "torch_linear"
    return "fp8"


class LLaMA31FP8Fixtures(ModelFixtureMixin):
    """
    Base LLaMA 3.1-3.3 FP8 Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self):
        if find_spec("fms_mo"):
            import fms_mo.aiu_addons
            import fms_mo.aiu_addons.fp8.fp8_linear  # noqa: F401
        else:
            raise ImportError("fms-model-optimizer needed to run FP8 tests")
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            model = LLaMA(
                src_vocab_size=381,
                emb_dim=16,
                norm_eps=1e-05,
                nheads=4,
                kvheads=2,
                nlayers=2,
                pad_id=0,
                hidden_grow_factor=4,
                multiple_of=2,
                activation_fn="swish",
                p_dropout=0.0,
                max_expected_seq_len=131072,
                rope_theta=500000.0,
                rope_scaling={
                    "factor": 8.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192,
                    "rope_type": "llama3",
                },
                linear_config={
                    "linear_type": fp8_linear_type,
                    "input_activations": {
                        "actorder": None,
                        "block_structure": None,
                        "dynamic": True,
                        "group_size": None,
                        "num_bits": 8,
                        "observer": None,
                        "observer_kwargs": {},
                        "strategy": "token",
                        "symmetric": True,
                        "type": "float",
                    },
                    "output_activations": None,
                    "weights": {
                        "actorder": None,
                        "block_structure": None,
                        "dynamic": False,
                        "group_size": None,
                        "num_bits": 8,
                        "observer": "minmax",
                        "observer_kwargs": {},
                        "strategy": "channel",
                        "symmetric": True,
                        "type": "float",
                    },
                },
            )

            # Required for FP8 linear modules
            for name, module in model.named_modules():
                if isinstance(module, UninitializedModule):
                    fqn_list = name.split(".")
                    parent_name = ".".join(fqn_list[:-1])
                    setattr(
                        model.get_submodule(parent_name),
                        fqn_list[-1],
                        module.initialize(name),
                    )
        torch.set_default_dtype(torch.float32)
        return model

    def _maybe_get_initialized_parameter(self, key, parameter):
        if parameter.dtype == torch.float8_e4m3fn:
            return torch.randn_like(parameter, dtype=torch.float32).to(
                torch.float8_e4m3fn
            )
        return None


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or (torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 9)),
    reason="FP8 is only available on GPUs with device level 8.9 or higher",
)
class TestLlama31FP8(
    ModelConsistencyTestSuite, ModelCompileTestSuite, LLaMA31FP8Fixtures
):
    """
    Test LLaMA 3.1 FP8 model consistency
    """

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    def test_model_unfused(self, model, signature):
        pytest.skip("weight unfuse is not defined for FP8")
