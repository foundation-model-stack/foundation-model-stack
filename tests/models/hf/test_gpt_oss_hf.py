import pytest

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from fms.models.hf.gpt_oss import convert_to_hf
from fms.models.hf.gpt_oss.configuration_gpt_oss_hf import HFAdaptedGptOssConfig
from fms.models.hf.gpt_oss.modeling_gpt_oss_hf import HFAdaptedGptOssForCausalLM
from fms.models.gpt_oss import GptOss, GptOssConfig
from fms.testing._internal.hf.model_test_suite import (
    HFConfigFixtureMixin,
    HFConfigTestSuite,
    HFModelCompileTestSuite,
    HFModelFixtureMixin,
)
from fms.testing._internal.model_test_suite import ModelFixtureMixin
from fms.utils.config import ModelConfig

from ..test_gpt_oss import GptOssFixtures


# Shared test configuration parameters
TEST_GPT_OSS_CONFIG_PARAMS = {
    "head_dim": 16,
    "norm_eps": 1e-05,
    "nheads": 16,
    "kvheads": 8,
    "nlayers": 4,
    "num_experts": 8,
    "src_vocab_size": 384,
    "emb_dim": 1024,
    "rope_base": 150000.0,
    "rope_scaling_factor": 32.0,
    "rope_ntk_alpha": 1.0,
    "rope_ntk_beta": 32.0,
}


class GptOssHFFixtures(ModelFixtureMixin, HFConfigFixtureMixin, HFModelFixtureMixin):
    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_model(self, model: GptOss, fms_hf_config: PretrainedConfig, **kwargs):
        fms_hf_config = PretrainedConfig(**TEST_GPT_OSS_CONFIG_PARAMS)
        return HFAdaptedGptOssForCausalLM.from_fms_model(
            model, **fms_hf_config.to_dict()
        )

    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_config(
        self, tokenizer: PreTrainedTokenizer, model: GptOss, **kwargs
    ) -> PretrainedConfig:
        return HFAdaptedGptOssConfig.from_fms_config(model.get_config())

    @pytest.fixture(scope="class", autouse=True)
    def oss_hf_model(self, fms_hf_model: HFAdaptedGptOssForCausalLM) -> PreTrainedModel:
        return convert_to_hf(fms_hf_model)


class GptOssFixturesEquivalence(GptOssFixtures):
    """
    GptOss transformers class does not support overriding
    the default config in HF; So for these tests we are using
    the values as they come from the HF model.

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: GptOssConfig):
        model = GptOss(config=config)
        model.base_model.post_init()
        return model

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        return GptOssConfig(**TEST_GPT_OSS_CONFIG_PARAMS)


class TestGptOssHF(
    HFConfigTestSuite,
    HFModelCompileTestSuite,
    GptOssFixturesEquivalence,
    GptOssHFFixtures,
):
    """
    GptOss FMS Huggingface Tests for:

    - FMS Huggingface configuration tests
    - FMS Huggingface model compile tests
    - FMS GptOss HF fixtures tests
    - FMS GptOss HF equivalence tests

    GptOss model is quantized so when converting the weights the
    output generated is different from the original HF model.
    This way the test for HF parity is at
    the ../hf_equivalence/test_gpt_oss.py file.
    """

    # implementation of abstract property _hf_specific_params
    _hf_specific_params = ["eos_token_id", "bos_token_id"]
    # implementation of abstract property _get_hf_signature_params
    _get_hf_signature_params = ["input_ids", "labels"]
