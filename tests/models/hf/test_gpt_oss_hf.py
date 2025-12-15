import pytest
import torch

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoTokenizer,
)

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
from fms.utils.config import ModelConfig

from ..test_gpt_oss import GptOssFixtures


class GptOssHFFixtures(ModelFixtureMixin, HFConfigFixtureMixin, HFModelFixtureMixin):
    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_model(self, model: GptOss, fms_hf_config: PretrainedConfig, **kwargs):
        fms_hf_config = PretrainedConfig(
            sliding_window=4,
            head_dim=16,
            norm_eps=1e-05,
            nheads=4,
            kvheads=1,
            nlayers=2,
            num_experts=4,
            rope_base=150000.0,
            rope_scaling_factor=32.0,
            rope_ntk_alpha=1.0,
            rope_ntk_beta=32.0,
        )
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
        return model

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        gpt_oss_config = GptOssConfig(
            sliding_window=4,
            head_dim=16,
            norm_eps=1e-05,
            nheads=4,
            kvheads=1,
            nlayers=2,
            num_experts=4,
            rope_base=150000.0,
            rope_scaling_factor=32.0,
            rope_ntk_alpha=1.0,
            rope_ntk_beta=32.0,
        )
        return gpt_oss_config


class TestGptOssHF(
    HFConfigTestSuite,
    HFModelEquivalenceTestSuite,
    HFModelGenerationTestSuite,
    HFModelCompileTestSuite,
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
        tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        tokenizer.pad_token = tokenizer.eos_token
        print(f"tokenizer gpt-oss {tokenizer}")
        encoding = tokenizer(texts, padding=True, return_tensors="pt")

        # Fix for newer versions of transformers
        use_cache_kwarg = {}
        if use_cache is not None:
            use_cache_kwarg["use_cache"] = use_cache

        print(f"use_cache_kwarg {use_cache_kwarg}")

        model.eval()
        with torch.no_grad():
            generated_ids = model.generate(
                **encoding,
                num_beams=num_beams,
                max_new_tokens=5,
                temperature=0.0,
                do_sample=False,
                top_k=50,
                **use_cache_kwarg,
            )

        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return generated_texts
