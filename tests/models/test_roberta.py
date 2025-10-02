import pytest
import torch

from fms.models.roberta import (
    RoBERTa,
    RoBERTaClassificationConfig,
    RoBERTaConfig,
    RoBERTaForClassification,
    RoBERTaForQuestionAnswering,
    RoBERTaQuestionAnsweringConfig,
)
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)


class RoBERTaFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base RoBERTa Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: RoBERTaConfig):
        return RoBERTa(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> RoBERTaConfig:
        return RoBERTaConfig(
            src_vocab_size=384,
            emb_dim=16,
            nheads=8,
            nlayers=2,
            max_pos=512,
            hidden_grow_factor=2.0,
            tie_heads=True,
        )


class TestRoBERTa(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    RoBERTaFixtures,
):
    """
    Model Test Suite for RoBERTa

    This suite will include tests for:
    - model configuration
    - basic load/save model
    - consistency of model output
    """

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]
    _get_signature_input_ids = torch.arange(1, 16, dtype=torch.int64).unsqueeze(0)

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config
        without mutating the original config"""
        model = type(model)(config=config, pad_id=config.pad_id + 1)
        # check not same reference
        assert model.get_config() is not config

        # modify pad_id to the new value expected and check equivalence
        config.pad_id = config.pad_id + 1
        assert model.get_config().as_dict() == config.as_dict()


class RoBERTaGPTQFixtures(ModelFixtureMixin):
    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self):
        return RoBERTa(
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
class TestRoBERTaGPTQ(
    ModelConsistencyTestSuite, ModelCompileTestSuite, RoBERTaGPTQFixtures
):
    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    def test_model_unfused(self, model, signature):
        pytest.skip("weight unfuse is not implemented for GPTQ")


class RoBERTaQuestionAnsweringFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """Define uninitialized model and config used by RoBERTaQuestionAnswering tests"""

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: RoBERTaQuestionAnsweringConfig):
        return RoBERTaForQuestionAnswering(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> RoBERTaQuestionAnsweringConfig:
        return RoBERTaQuestionAnsweringConfig(
            src_vocab_size=384,
            emb_dim=16,
            nheads=8,
            nlayers=2,
            max_pos=512,
            hidden_grow_factor=2.0,
            tie_heads=False,
        )


class TestRoBERTaQuestionAnswering(
    ModelConsistencyTestSuite, ModelCompileTestSuite, RoBERTaQuestionAnsweringFixtures
):
    """Main test class for RoBERTaQuestionAnswering"""

    # x is the main parameter for this model which is the input tensor
    # a default attention mask is generated when not provided
    _get_signature_params = ["x"]
    _get_signature_input_ids = torch.arange(1, 16, dtype=torch.int64).unsqueeze(0)

    @staticmethod
    def _get_signature_logits_getter_fn(f_out) -> torch.Tensor:
        return torch.cat([f_out[0], f_out[1]], dim=-1)

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config
        without mutating the original config"""
        model = type(model)(config=config, pad_id=config.pad_id + 1)
        # check not same reference
        assert model.get_config() is not config

        # modify pad_id to the new value expected and check equivalence
        config.pad_id = config.pad_id + 1
        assert model.get_config().as_dict() == config.as_dict()


class RoBERTaClassificationFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """Define uninitialized model and config used by RoBERTaForClassification tests"""

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: RoBERTaClassificationConfig):
        return RoBERTaForClassification(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> RoBERTaClassificationConfig:
        return RoBERTaClassificationConfig(
            src_vocab_size=384,
            emb_dim=16,
            nheads=8,
            nlayers=2,
            max_pos=512,
            hidden_grow_factor=2.0,
            tie_heads=False,
        )


class TestRoBERTaClassification(
    ModelConsistencyTestSuite, ModelCompileTestSuite, RoBERTaClassificationFixtures
):
    """Main test class for RoBERTaForClassification"""

    # x is the main parameter for this model which is the input tensor
    # a default attention mask is generated when not provided
    _get_signature_params = ["x"]
    _get_signature_input_ids = torch.arange(1, 16, dtype=torch.int64).unsqueeze(0)

    # @staticmethod
    # def _get_signature_logits_getter_fn(f_out) -> torch.Tensor:
    #     return f_out

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config
        without mutating the original config"""
        model = type(model)(config=config, pad_id=config.pad_id + 1)
        # check not same reference
        assert model.get_config() is not config

        # modify pad_id to the new value expected and check equivalence
        config.pad_id = config.pad_id + 1
        assert model.get_config().as_dict() == config.as_dict()
