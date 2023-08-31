import pytest

from fms.models.hf.llama.configuration_llama_hf import LLaMAHFConfig
from fms.models.hf.llama.modeling_llama_hf import LLaMAHFLMHeadModel
from fms.models.llama import LLaMA, LLaMAConfig
from ..base import _case_paths, _test_ids
from ..test_hf_model import AbstractHFModelTest

class TestLlama(AbstractHFModelTest):
    """
    Model Test Suite for llama
    """

    _model_class = LLaMA
    _config_class = LLaMAConfig
    _hf_model_class = LLaMAHFLMHeadModel
    _hf_config_class = LLaMAHFConfig
    _hf_specific_params = ["eos_token_id", "bos_token_id"]
    _hf_forward_parameters = ["input_ids", "labels"]

    @pytest.fixture(params=_case_paths("llama"), ids=_test_ids)
    def cases(self, request):
        return request.param