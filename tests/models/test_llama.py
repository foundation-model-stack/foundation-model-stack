import pytest


from fms.models.llama import LLaMA, LLaMAConfig
from fms.testing._internal.common_paths import resource_path_fixture
from fms.testing._internal.common_model import AbstractModelTest


class TestLlama(AbstractModelTest):
    """
    Model Test Suite for llama
    """

    _model_class = LLaMA
    _config_class = LLaMAConfig
    _forward_parameters = 1

    @resource_path_fixture(test_name="llama", prefix="model")
    def cases(self, request):
        return request.param
