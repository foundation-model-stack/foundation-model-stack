import abc
import tempfile
from typing import List, Type

import pytest

from fms.utils.config import ModelConfig
from tests.models.test_config import AbstractConfigTest
from transformers import PretrainedConfig


class AbstractHFConfigTest(AbstractConfigTest):
    """General huggingface config testing class for future use with other models"""

    @property
    @abc.abstractmethod
    def _hf_specific_params(self) -> List[str]:
        return []

    @property
    @abc.abstractmethod
    def _hf_config_class(self) -> Type[PretrainedConfig]:
        pass

    @pytest.fixture
    @abc.abstractmethod
    def hf_config(self, config: ModelConfig, **kwargs) -> PretrainedConfig:
        pass

    def test_hf_config_round_trip(self, hf_config: PretrainedConfig):
        """Test that the config can save and load properly"""

        with tempfile.TemporaryDirectory() as workdir:
            hf_config_path = f"{workdir}/config.json"
            hf_config.save_pretrained(hf_config_path)
            hf_config_loaded = type(hf_config).from_pretrained(hf_config_path)
            assert hf_config.to_dict() == hf_config_loaded.to_dict()

    def test_hf_config_from_fms_config(self, config: ModelConfig, hf_config: PretrainedConfig):
        """Test that the config can save and load properly"""

        hf_config_loaded = type(hf_config).from_fms_config(config)
        hf_config_loaded_dict = hf_config_loaded.to_dict()
        hf_config_dict = hf_config.to_dict()
        # ignoring params that are HF specific
        for p in self._hf_specific_params:
            hf_config_loaded_dict[p] = hf_config_dict[p]
        assert hf_config_dict == hf_config_loaded_dict
