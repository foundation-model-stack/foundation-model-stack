import pytest
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from fms.models.gpt_bigcode import GPTBigCode
from fms.models.hf.gpt_bigcode import convert_to_hf
from fms.models.hf.gpt_bigcode.configuration_gpt_bigcode_hf import (
    HFAdaptedGPTBigCodeConfig,
)
from fms.models.hf.gpt_bigcode.modeling_gpt_bigcode_hf import (
    HFAdaptedGPTBigCodeForCausalLM,
)
from fms.testing._internal.hf.model_test_suite import (
    HFConfigFixtureMixin,
    HFConfigTestSuite,
    HFModelCompileTestSuite,
    HFModelEquivalenceTestSuite,
    HFModelFixtureMixin,
    HFModelGenerationTestSuite,
)
from fms.testing._internal.model_test_suite import ModelFixtureMixin

from ..test_gpt_bigcode import GPTBigCodeFixtures


class HFAdaptedGPTBigCodeFixtures(
    ModelFixtureMixin, HFConfigFixtureMixin, HFModelFixtureMixin
):
    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_model(
        self, model: GPTBigCode, fms_hf_config: PretrainedConfig, **kwargs
    ):
        return HFAdaptedGPTBigCodeForCausalLM.from_fms_model(
            model, **fms_hf_config.to_dict()
        )

    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_config(
        self, tokenizer: PreTrainedTokenizer, model: GPTBigCode, **kwargs
    ) -> PretrainedConfig:
        bos_token_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.eos_token_id
        )
        return HFAdaptedGPTBigCodeConfig.from_fms_config(
            model.get_config(),
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=bos_token_id,
        )

    @pytest.fixture(scope="class", autouse=True)
    def oss_hf_model(
        self, fms_hf_model: HFAdaptedGPTBigCodeForCausalLM
    ) -> PreTrainedModel:
        return convert_to_hf(fms_hf_model)


class TestHFAdaptedGPTBigCode(
    HFConfigTestSuite,
    HFModelEquivalenceTestSuite,
    HFModelGenerationTestSuite,
    HFModelCompileTestSuite,
    GPTBigCodeFixtures,
    HFAdaptedGPTBigCodeFixtures,
):
    """
    GPTBigCode FMS Huggingface Tests for:

    - FMS Huggingface configuration tests
    - model equivalency tests
    - model generation tests
    """

    # implementation of abstract property _hf_specific_params
    _hf_specific_params = ["eos_token_id", "bos_token_id"]
    # implementation of abstract property _get_hf_signature_params
    _get_hf_signature_params = ["input_ids", "labels"]
