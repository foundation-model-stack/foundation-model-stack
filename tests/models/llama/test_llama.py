import pytest


from fms.models.llama import LLaMA, LLaMAConfig
from ..base import _case_paths, _test_ids

from ..test_model import AbstractModelTest


class TestLlama(AbstractModelTest):
    """
    Model Test Suite for llama
    """

    _model_class = LLaMA
    _config_class = LLaMAConfig
    _forward_parameters = 1

    @pytest.fixture(params=_case_paths("llama"), ids=_test_ids)
    def cases(self, request):
        return request.param
