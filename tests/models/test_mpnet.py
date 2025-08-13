import pytest
import torch
from fms.models import get_model
from fms.models.mpnet import Mpnet, MpnetConfig
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)


class MpnetFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base Mpnet Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: MpnetConfig):
        return Mpnet(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> MpnetConfig:
        return MpnetConfig(
            src_vocab_size=384,
            emb_dim=16,
            nheads=8,
            nlayers=2,
            hidden_grow_factor=2.0,
            tie_heads=True,
        )


class TestMpnet(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    MpnetFixtures,
):
    """
    Model Test Suite for Mpnet

    This suite will include tests for:
    - model configuration
    - basic load/save model
    - consistency of model output
    """

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]
    _get_signature_input_ids = torch.arange(1, 16, dtype=torch.int64).unsqueeze(0)

    @staticmethod
    def _get_signature_logits_getter_fn(f_out) -> torch.Tensor:
        # return torch.cat([f_out[0][:1], f_out[1]], dim=-1)
        return torch.cat([f_out[0][0][:1], f_out[1]], dim=-1)

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed
        kwargs into the config without mutating the original config"""
        model = type(model)(config=config, pad_id=config.pad_id + 1)
        # check not same reference
        assert model.get_config() is not config

        # modify pad_id to the new value expected and check equivalence
        config.pad_id = config.pad_id + 1
        assert model.get_config().as_dict() == config.as_dict()

    def test_mpnet_input_too_long(self):
        model = get_model("mpnet", "v2", pretrained=False)
        long_input = torch.randint(0, 100, (1, model.config.max_expected_seq_len + 10))
        with pytest.raises(ValueError):
            model(long_input)


class MpnetGPTQFixtures(ModelFixtureMixin):
    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self):
        return Mpnet(
            src_vocab_size=384,
            emb_dim=32,
            nheads=8,
            nlayers=2,
            max_pos=512,
            hidden_grow_factor=2.0,
            tie_heads=True,
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
        if "qzeros" in key:
            return torch.ones(parameter.shape, dtype=torch.int32) * 8
        if "g_idx" in key:
            return parameter
        return None


@pytest.mark.autogptq
class TestMpnetGPTQ(
    ModelConsistencyTestSuite, ModelCompileTestSuite, MpnetGPTQFixtures
):
    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    @staticmethod
    def _get_signature_logits_getter_fn(f_out) -> torch.Tensor:
        return torch.cat([f_out[0][0][:1], f_out[1]], dim=-1)

    def test_model_unfused(self, model, signature):
        pytest.skip("weight unfuse is not implemented for GPTQ")
