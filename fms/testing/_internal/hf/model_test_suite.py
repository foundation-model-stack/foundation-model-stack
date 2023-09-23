import numpy as np
import torch
from fms.testing._internal.model_test_suite import ConfigFixtureMixin, ModelFixtureMixin
from fms.testing._internal.test_resource_utils import AbstractResourcePath

SEED = 42
torch.manual_seed(SEED)  # pytorch random seed
np.random.seed(SEED)  # numpy random seed
torch.backends.cudnn.deterministic = True


import abc
import os
import tempfile

import pytest
import torch
import torch.nn as nn
import numpy as np
import itertools
from typing import Tuple, Type, List, Optional

from transformers import (
    PreTrainedModel,
    AutoTokenizer,
    PreTrainedTokenizer,
    PretrainedConfig,
)

from fms.utils.config import ModelConfig
from ...comparison import (
    HFModelSignatureParams,
    ModelSignatureParams,
    compare_model_signatures,
)


SEED = 42
torch.manual_seed(SEED)  # pytorch random seed
np.random.seed(SEED)  # numpy random seed
torch.backends.cudnn.deterministic = True


class HFConfigFixtureMixin(ConfigFixtureMixin, metaclass=abc.ABCMeta):
    """Mix this in with another AbstractResourcePath testing class to include the config and config_class fixtures"""

    @property
    @abc.abstractmethod
    def _hf_specific_params(self) -> List[str]:
        return []

    @property
    @abc.abstractmethod
    def _hf_config_class(self) -> Type[PretrainedConfig]:
        pass

    # class specific fixtures
    @pytest.fixture(scope="class", autouse=True)
    def tokenizer(self, resource_path: str) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(os.path.join(resource_path, "tokenizer"))

    @pytest.fixture(scope="class", autouse=True)
    def hf_config(
        self,
        config: ModelConfig,
        hf_config_class: Type[PretrainedConfig],
        tokenizer: PreTrainedTokenizer,
    ) -> PretrainedConfig:
        bos_token_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.eos_token_id
        )
        return hf_config_class.from_fms_config(
            config, eos_token_id=tokenizer.eos_token_id, bos_token_id=bos_token_id
        )

    @pytest.fixture(scope="class", autouse=True)
    def hf_config_class(self) -> Type[PretrainedConfig]:
        return self._hf_config_class


class HFModelFixtureMixin(ModelFixtureMixin, metaclass=abc.ABCMeta):
    @pytest.fixture(scope="class", autouse=True)
    def hf_model_class(self) -> Type[PreTrainedModel]:
        return self._hf_model_class

    @pytest.fixture(scope="class", autouse=True)
    def hf_model(
        self,
        hf_config: PretrainedConfig,
        model: nn.Module,
        hf_model_class: Type[PreTrainedModel],
    ) -> PreTrainedModel:
        """create hf_model and load state dict from given pytorch native model then return hf_model"""
        return hf_model_class.from_fms_model(model, **hf_config.to_dict())

    @pytest.fixture(scope="class", autouse=True)
    def oss_hf_model(self, hf_model: PreTrainedModel) -> PreTrainedModel:
        return self._oss_hf_model(hf_model)

    @abc.abstractmethod
    def _oss_hf_model(self, hf_model: PreTrainedModel) -> PreTrainedModel:
        """Given an fms hf config, create the equivalent oss hf model"""
        pass

    @property
    @abc.abstractmethod
    def _hf_model_class(self) -> Type[PreTrainedModel]:
        pass

    @property
    @abc.abstractmethod
    def _hf_forward_parameters(self) -> List[str]:
        pass

    def _forward_parameters(self):
        return len(self._hf_forward_parameters) - 1


class HFConfigTestSuite(AbstractResourcePath, HFConfigFixtureMixin):
    """General huggingface config testing class for future use with other models"""

    def test_hf_config_from_fms_config(
        self, config: ModelConfig, hf_config: PretrainedConfig
    ):
        """Test that the config can save and load properly"""

        hf_config_loaded = type(hf_config).from_fms_config(config)
        hf_config_loaded_dict = hf_config_loaded.to_dict()
        hf_config_dict = hf_config.to_dict()
        # ignoring params that are HF specific
        for p in self._hf_specific_params:
            hf_config_loaded_dict[p] = hf_config_dict[p]
        assert hf_config_dict == hf_config_loaded_dict

    def test_hf_config_round_trip(self, hf_config: PretrainedConfig):
        """Test that the config can save and load properly"""

        with tempfile.TemporaryDirectory() as workdir:
            hf_config_path = f"{workdir}/hf_config.json"
            hf_config.save_pretrained(hf_config_path)
            hf_config_loaded = type(hf_config).from_pretrained(hf_config_path)
            assert hf_config.to_dict() == hf_config_loaded.to_dict()


class HFModelEquivalenceTestSuite(
    AbstractResourcePath, HFConfigFixtureMixin, HFModelFixtureMixin
):
    """General huggingface model testing class for future use with other models"""

    # common tests
    def test_hf_and_fms_model_equivalence(self, hf_model, model):
        """test model signature equivalence between huggingface model and fms model"""

        hf_model = type(hf_model).from_fms_model(model, **hf_model.config.to_dict())
        fms_signature_params = ModelSignatureParams(
            model, len(self._hf_forward_parameters) - 1
        )
        hf_signature_params = HFModelSignatureParams(
            hf_model, self._hf_forward_parameters
        )
        compare_model_signatures(fms_signature_params, hf_signature_params)

    def test_hf_and_oss_hf_model_equivalence(self, hf_model, oss_hf_model):
        inp = torch.arange(5, 15).unsqueeze(0)
        hf_signature_params = HFModelSignatureParams(
            hf_model, self._hf_forward_parameters, inp=inp
        )
        oss_hf_signature_params = HFModelSignatureParams(
            oss_hf_model, self._hf_forward_parameters, inp=inp
        )
        compare_model_signatures(hf_signature_params, oss_hf_signature_params)

    def test_hf_from_fms_and_hf_from_pretrained_equivalence(
        self, tmpdir_factory, model: nn.Module, hf_model: PreTrainedModel
    ):
        hf_path = tmpdir_factory.mktemp("hf")
        hf_model = type(hf_model).from_fms_model(model, **hf_model.config.to_dict())

        hf_model.save_pretrained(hf_path)

        hf_model_from_fms = type(hf_model).from_fms_model(
            model, **hf_model.config.to_dict()
        )
        hf_model = type(hf_model).from_pretrained(hf_path)

        hf_from_fms_signature_params = HFModelSignatureParams(
            hf_model_from_fms, self._hf_forward_parameters
        )
        hf_signature_params = HFModelSignatureParams(
            hf_model, self._hf_forward_parameters
        )
        compare_model_signatures(hf_from_fms_signature_params, hf_signature_params)

    def test_hf_model_round_trip_equivalence(
        self, hf_model: PreTrainedModel, hf_config: PretrainedConfig
    ):
        """Test that the huggingface model can save and load properly"""
        hf_model_from_config = type(hf_model)(hf_config)
        hf_model_from_config.load_state_dict(hf_model.state_dict())
        compare_model_signatures(
            HFModelSignatureParams(hf_model, self._hf_forward_parameters),
            HFModelSignatureParams(hf_model_from_config, self._hf_forward_parameters),
        )

        with tempfile.TemporaryDirectory() as workdir:
            hf_model_path = f"{workdir}/hf_model"
            hf_model_from_config.save_pretrained(hf_model_path)
            hf_model_loaded = type(hf_model).from_pretrained(hf_model_path)

        compare_model_signatures(
            HFModelSignatureParams(hf_model, self._hf_forward_parameters),
            HFModelSignatureParams(hf_model_loaded, self._hf_forward_parameters),
        )


class HFModelGenerationTestSuite(
    AbstractResourcePath, HFConfigFixtureMixin, HFModelFixtureMixin
):
    """Testing suite intended to test hf model generation"""

    @staticmethod
    def _predict_text(model, tokenizer, texts, use_cache, num_beams):
        encoding = tokenizer(texts, padding=True, return_tensors="pt")

        model.eval()
        with torch.no_grad():
            generated_ids = model.generate(
                **encoding,
                use_cache=use_cache,
                num_beams=num_beams,
                max_new_tokens=20,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                pad_token_id=model.config.eos_token_id,
            )
        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return generated_texts

    text_options = [
        ["hello how are you?"],
        ["hello how are you?", "a: this is a test. b: this is another test. a:"],
    ]
    use_cache_options = [True, False, None]
    num_beams_options = [1, 3]
    generate_equivalence_args = list(
        itertools.product(text_options, use_cache_options, num_beams_options)
    )

    @pytest.mark.parametrize("texts,use_cache,num_beams", generate_equivalence_args)
    def test_hf_generate_equivalence(
        self,
        texts: List[str],
        use_cache: Optional[bool],
        num_beams: int,
        hf_model: PreTrainedModel,
        oss_hf_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """test that an hf model created from fms and an hf model loaded from hf checkpoint produce the same output if
        they have the same weights and configs
        """
        print(texts)
        output_fms = self._predict_text(
            hf_model, tokenizer, texts, use_cache, num_beams
        )
        output_hf = self._predict_text(
            oss_hf_model, tokenizer, texts, use_cache, num_beams
        )

        assert output_fms == output_hf, f"{output_fms}\n{output_hf}"

    hf_batch_generate_args = list(
        itertools.product(use_cache_options, num_beams_options)
    )

    @pytest.mark.parametrize("use_cache,num_beams", hf_batch_generate_args)
    def test_hf_batch_generate(
        self,
        use_cache,
        num_beams,
        hf_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """Tests that the output of a given prompt done alone and with batch generation is the same"""
        text_1 = "hello how are you?"
        text_2 = "a: this is a test. b: this is another test. a:"
        text_batch = [text_1, text_2]

        # required for batch generation
        tokenizer.padding_side = "left"

        output_batch = self._predict_text(
            hf_model, tokenizer, text_batch, use_cache, num_beams
        )

        text1 = [text_1]
        output_text1 = self._predict_text(
            hf_model, tokenizer, text1, use_cache, num_beams
        )[0]

        text2 = [text_2]
        output_text2 = self._predict_text(
            hf_model, tokenizer, text2, use_cache, num_beams
        )[0]

        assert (
            output_batch[0] == output_text1
        ), f"text 1 incorrect - \n{output_batch[0]}\n{output_text1}"
        assert (
            output_batch[1] == output_text2
        ), f"text 2 incorrect - \n{output_batch[1]}\n{output_text2}"
