from fms.models.llama import LLaMA, LLaMAConfig
from fms.testing._internal.common_path import resource_path_fixture
from fms.testing._internal.common_model import *


class TestLlama(AbstractModelTest):
    """
    Model Test Suite for llama
    """

    _forward_parameters = 1
    _model_class = LLaMA
    _config_class = LLaMAConfig

    @resource_path_fixture(test_name="llama", prefix="model")
    def resource_path(self, request):
        return request.param
