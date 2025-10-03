import pytest
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer

from fms.models.hf.gpt_oss import convert_to_hf
from fms.models.hf.gpt_oss.configuration_gpt_oss_hf import HFAdaptedGptOssConfig
from fms.models.hf.gpt_oss.modeling_gpt_oss_hf import HFAdaptedGptOssForCausalLM
from fms.models.gpt_oss import GptOss
from fms.testing._internal.hf.model_test_suite import (
    HFAutoModelTestSuite,
    HFConfigFixtureMixin,
    HFConfigTestSuite,
    HFModelCompileTestSuite,
    HFModelEquivalenceTestSuite,
    HFModelFixtureMixin,
    HFModelGenerationTestSuite,
)
from fms.testing._internal.model_test_suite import ModelFixtureMixin

from tests.models.test_gpt_oss import GptOssFixtures


class GptOssHFFixtures(ModelFixtureMixin, HFConfigFixtureMixin, HFModelFixtureMixin):
    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_model(self, model: GptOss, fms_hf_config: PretrainedConfig, **kwargs):
        return HFAdaptedGptOssForCausalLM.from_fms_model(
            model, **fms_hf_config.to_dict()
        )

    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_config(
        self, tokenizer: PreTrainedTokenizer, model: GptOss, **kwargs
    ) -> PretrainedConfig:
        bos_token_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.eos_token_id
        )
        return HFAdaptedGptOssConfig.from_fms_config(
            model.get_config(),
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=bos_token_id,
        )

    @pytest.fixture(scope="class", autouse=True)
    def oss_hf_model(
        self, fms_hf_model: HFAdaptedGptOssForCausalLM
    ) -> PreTrainedModel:
        return convert_to_hf(fms_hf_model)


class TestGptOssHF(
    HFConfigTestSuite,
    HFModelEquivalenceTestSuite,
    HFModelGenerationTestSuite,
    HFModelCompileTestSuite,
    HFAutoModelTestSuite,
    GptOssFixtures,
    GptOssHFFixtures,
):
    """
    GptOss FMS Huggingface Tests for:

    - FMS Huggingface configuration tests
    - model equivalency tests
    - model generation tests
    """

    # implementation of abstract property _hf_specific_params
    _hf_specific_params = ["eos_token_id", "bos_token_id"]
    # implementation of abstract property _get_hf_signature_params
    _get_hf_signature_params = ["input_ids", "labels"]
