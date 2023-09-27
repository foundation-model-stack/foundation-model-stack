import os.path

from fms.models.llama import LLaMA, LLaMAConfig
from fms.testing._internal.model_test_suite import (
    ModelAPITestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
)
from fms.testing._internal.test_resource_utils import resource_path_fixture


class TestLlama(ModelAPITestSuite, ModelConfigTestSuite, ModelConsistencyTestSuite):
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

    @resource_path_fixture(
        test_name="llama",
        prefix="model",
        common_tests_path=os.path.join(
            os.path.dirname(__file__), "..", "resources", "models"
        ),
    )
    def resource_path(self, request):
        return request.param
