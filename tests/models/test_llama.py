import os.path

from fms.models.llama import LLaMA, LLaMAConfig
from fms.testing._internal.model_test_suite import (
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
)
from fms.testing._internal.test_resource_utils import resource_path_fixture


class TestLlama(ModelConfigTestSuite, ModelConsistencyTestSuite):
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

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config without mutating the original config"""
        model = type(model)(config=config, pad_id=config.pad_id + 1)
        # check not same reference
        assert model.get_config() is not config

        # modify pad_id to the new value expected and check equivalence
        config.pad_id = config.pad_id + 1
        assert model.get_config().as_dict() == config.as_dict()
