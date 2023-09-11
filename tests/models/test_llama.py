from fms.models.llama import LLaMA, LLaMAConfig
from fms.testing._internal.common_config import CommonConfigTestMixin
from fms.testing._internal.common_model import CommonModelTestMixin
from fms.testing._internal.common_model_consistency import ModelConsistencyTestMixin
from fms.testing._internal.common_path import resource_path_fixture


class TestLlama(CommonModelTestMixin, CommonConfigTestMixin, ModelConsistencyTestMixin):
    """
    Model Test Suite for llama

    This suite will include tests for:
    - model configuration
    - basic load/save model
    - consistency of model output
    """

    _forward_parameters = 1
    _model_class = LLaMA
    _config_class = LLaMAConfig

    @resource_path_fixture(test_name="llama", prefix="model")
    def resource_path(self, request):
        return request.param
