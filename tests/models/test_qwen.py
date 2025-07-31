# -*- coding: utf-8 -*-

"""
Test case for Qwen3 models support.
"""

# pylint: disable=protected-access,unused-argument,abstract-method
# pylint: disable=missing-module-docstring,disable=missing-class-docstring
# pylint: disable=missing-function-docstring,line-too-long,invalid-name
# pylint: disable=unused-import,too-few-public-methods
# pylint: disable=unknown-option-value,arguments-differ
# type: ignore

import pytest
import torch

from fms.models.qwen import Qwen, QwenConfig, QwenHeadless
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig


class QwenFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base Qwen Fixtures that can be re-used for other purposes

    This will include the config and model signatures.
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: QwenConfig):
        """_summary_

        Args:
            config (QwenConfig): _description_

        Returns:
            _type_: _description_
        """
        return Qwen(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        """_summary_

        Returns:
            ModelConfig: _description_
        """
        return QwenConfig(
            src_vocab_size=384,
            emb_dim=16,
            norm_eps=1e-05,
            nheads=8,
            kvheads=2,
            nlayers=2,
            pad_id=0,
            hidden_grow_factor=3.5,
            multiple_of=2,
            tie_heads=False,
            activation_fn="swish",
            sliding_window=4000,
            rope_base=1000000.0,
            p_dropout=0.0,
            max_expected_seq_len=4096,
            linear_config={"linear_type": "torch_linear"},
            fused_weights=True,
        )


class Test_Qwen(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    QwenFixtures,
):
    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    def test_config_passed_to_model_and_updated(self, model, config):
        """
        test model constructor appropriately merges any passed kwargs into the
        config without mutating the original config
        """
        model = type(model)(config=config, pad_id=config.pad_id + 1)
        # check not same reference
        assert model.get_config() is not config

        # modify pad_id to the new value expected and check equivalence
        config.pad_id = config.pad_id + 1
        assert model.get_config().as_dict() == config.as_dict()

    @pytest.fixture
    def headless_model(self, model: Qwen) -> QwenHeadless:
        return model.base_model

class QwenGPTQFixtures(ModelFixtureMixin):
    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return Qwen(
            src_vocab_size=384,
            emb_dim=64,
            norm_eps=1e-05,
            nheads=32,
            kvheads=8,
            nlayers=2,
            pad_id=0,
            hidden_grow_factor=3.5,
            multiple_of=2,
            tie_heads=False,
            activation_fn="swish",
            sliding_window=4000,
            rope_base=1000000.0,
            p_dropout=0.0,
            max_expected_seq_len=4096,
            linear_config={"linear_type": "gptq_cpu"},
            fused_weights=True,
        )

    def _maybe_get_initialized_parameter(self, key, parameter):
        """_summary_

        Args:
            key (_type_): _description_
            parameter (_type_): _description_

        Returns:
            _type_: _description_
        """
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
class TestQwenGPTQ(
    ModelConsistencyTestSuite, ModelCompileTestSuite, QwenGPTQFixtures
):
    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    def test_model_unfused(self, model, signature):
        pytest.skip("weight unfuse is not implemented for GPTQ")
