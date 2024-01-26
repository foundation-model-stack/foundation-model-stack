import dataclasses
import logging
import os.path
import tempfile

from fms.utils.config import ModelConfig


@dataclasses.dataclass
class MockModelConfig(ModelConfig):
    required_a: int
    default_b: str = "default"


def test_round_trip():
    config = MockModelConfig(required_a=1)
    with tempfile.TemporaryDirectory() as workdir:
        config_path = f"{workdir}/config.json"
        assert not os.path.exists(config_path)
        config.save(config_path)
        assert os.path.exists(config_path) and os.path.isfile(config_path)
        config_loaded = MockModelConfig.load(config_path)
        assert config.required_a == config_loaded.required_a
        assert config.default_b == config_loaded.default_b


def test_as_dict():
    config = MockModelConfig(required_a=1, default_b="other_default")
    assert config.as_dict() == {"required_a": 1, "default_b": "other_default"}


def test_updated():
    config = MockModelConfig(required_a=1, default_b="other_default")
    config_updated = config.updated(required_a=2)
    assert config is not config_updated
    assert config_updated.required_a == 2
    assert config_updated.default_b == "other_default"


def test_updated_unknown_params_message(caplog):
    caplog.set_level(logging.INFO)
    config = MockModelConfig(required_a=1, default_b="other_default")
    config.updated(required_a=2, unknown_param_1=3, unknown_param_2=4)
    assert (
        """Found the following unknown parameters while cloning and updating the configuration: ['unknown_param_1', 'unknown_param_2']"""
        in caplog.text.strip()
    )
