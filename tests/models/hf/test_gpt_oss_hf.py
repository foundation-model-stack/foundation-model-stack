import platform
import pytest
import torch
import itertools

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoTokenizer,
)
from torch._dynamo.testing import CompileCounterWithBackend
from torch._dynamo.exc import TorchDynamoException

from fms.models.hf.gpt_oss import convert_to_hf
from fms.models.hf.gpt_oss.configuration_gpt_oss_hf import HFAdaptedGptOssConfig
from fms.models.hf.gpt_oss.modeling_gpt_oss_hf import HFAdaptedGptOssForCausalLM
from fms.models.gpt_oss import GptOss, GptOssConfig
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
from fms.testing.comparison import HFModelSignatureParams, get_signature
from fms.utils.config import ModelConfig

from ..test_gpt_oss import GptOssFixtures


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
        model = GptOss(config)
        model.base_model.post_init()
        return model

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        gpt_oss_config = GptOssConfig()
        return gpt_oss_config


class TestGptOssHF(
    HFConfigTestSuite,
    HFModelEquivalenceTestSuite,
    HFModelGenerationTestSuite,
    HFModelCompileTestSuite,
    HFAutoModelTestSuite,
    GptOssFixturesEquivalence,
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

    @staticmethod
    def _predict_text(model, tokenizer, texts, use_cache, num_beams):
        encoding = tokenizer(texts, padding=True, return_tensors="pt")

        # Fix for newer versions of transformers
        use_cache_kwarg = {}
        if use_cache is not None:
            use_cache_kwarg["use_cache"] = use_cache

        tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

        model.eval()
        with torch.no_grad():
            generated_ids = model.generate(
                **encoding,
                num_beams=num_beams,
                max_new_tokens=20,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                do_sample=False,
                **use_cache_kwarg,
            )
        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return generated_texts

    text_options = [
        ["hello how are you?"],
        ["hello how are you?", "a: this is a test. b: this is another test. a:"],
    ]
    use_cache_options = [True, False, None]
    num_beams_options = [1, 3]
    generate_equivalence_args = list(
        itertools.product(text_options, use_cache_options, num_beams_options)
    )
