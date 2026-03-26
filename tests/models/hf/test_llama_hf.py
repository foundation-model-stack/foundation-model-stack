import pytest
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from fms.models.hf.llama.configuration_llama_hf import HFAdaptedLLaMAConfig
from fms.models.hf.llama.modeling_llama_hf import HFAdaptedLLaMAForCausalLM
from fms.models.llama import LLaMA
from fms.models.hf.llama import convert_to_hf
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

from ..test_llama import LLaMA2Fixtures, LLaMA2GQAFixtures


class LLaMA2HFFixtures(ModelFixtureMixin, HFConfigFixtureMixin, HFModelFixtureMixin):
    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_model(self, model: LLaMA, fms_hf_config: PretrainedConfig, **kwargs):
        return HFAdaptedLLaMAForCausalLM.from_fms_model(
            model, **fms_hf_config.to_dict()
        )

    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_config(
        self, tokenizer: PreTrainedTokenizer, model: LLaMA, **kwargs
    ) -> PretrainedConfig:
        bos_token_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.eos_token_id
        )
        return HFAdaptedLLaMAConfig.from_fms_config(
            model.get_config(),
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=bos_token_id,
        )

    @pytest.fixture(scope="class", autouse=True)
    def oss_hf_model(self, fms_hf_model: HFAdaptedLLaMAForCausalLM) -> PreTrainedModel:
        return convert_to_hf(fms_hf_model)


class TestLLaMA2HF(
    HFConfigTestSuite,
    HFModelEquivalenceTestSuite,
    HFModelGenerationTestSuite,
    HFModelCompileTestSuite,
    HFAutoModelTestSuite,
    LLaMA2Fixtures,
    LLaMA2HFFixtures,
):
    """
    LLaMA2 FMS Huggingface Tests for:

    - FMS Huggingface configuration tests
    - model equivalency tests
    - model generation tests
    """

    # implementation of abstract property _hf_specific_params
    _hf_specific_params = ["eos_token_id", "bos_token_id"]
    # implementation of abstract property _get_hf_signature_params
    _get_hf_signature_params = ["input_ids", "labels"]


class TestLLaMA2GQAHF(
    HFModelEquivalenceTestSuite,
    HFModelGenerationTestSuite,
    HFModelCompileTestSuite,
    HFAutoModelTestSuite,
    LLaMA2GQAFixtures,
    LLaMA2HFFixtures,
):
    """
    LLaMA2-GQA FMS Huggingface Tests for:

    - model equivalency tests
    - model generation tests
    """

    # implementation of abstract property _hf_specific_params
    _hf_specific_params = ["eos_token_id", "bos_token_id"]
    # implementation of abstract property _get_hf_signature_params
    _get_hf_signature_params = ["input_ids", "labels"]
