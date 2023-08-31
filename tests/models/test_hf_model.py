import abc
import itertools
import os
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from fms.utils.config import ModelConfig
from .test_hf_config import AbstractHFConfigTest
from .test_model import AbstractModelTest
from .utils import (
    HFModelSignatureParams,
    ModelSignatureParams,
    compare_model_signatures,
)


SEED = 42
torch.manual_seed(SEED)  # pytorch random seed
np.random.seed(SEED)  # numpy random seed
torch.backends.cudnn.deterministic = True


class AbstractHFModelTest(AbstractHFConfigTest, AbstractModelTest):
    """General huggingface model testing class for future use with other models"""

    @pytest.fixture
    def model_locations(self, tmpdir, model, hf_model):
        hf_path = tmpdir + "/hf"
        fms_path = tmpdir + "/fms"
        hf_model = type(hf_model).from_fms_model(model, **hf_model.config.to_dict())

        tmpdir.mkdir("hf")
        tmpdir.mkdir("fms")
        hf_model.save_pretrained(hf_path)

        torch.save({"model_state": model.state_dict()}, f"{fms_path}/model_state.pth")
        model.get_config().save(f"{fms_path}/config.json")
        yield fms_path, hf_path

    @pytest.fixture
    def hf_models_loaded(
        self,
        model_locations: Tuple[str, str],
        model: nn.Module,
        hf_model: PreTrainedModel,
        config: ModelConfig,
        hf_config: PretrainedConfig,
    ):
        fms_model_path, hf_model_path = model_locations

        config = type(config).load(f"{fms_model_path}/config.json")
        model = type(model)(config)
        model.load_state_dict(torch.load(f"{fms_model_path}/model_state.pth", map_location="cpu").get("model_state"))

        hf_model_from_fms = type(hf_model).from_fms_model(model, **hf_config.to_dict())
        hf_model = type(hf_model).from_pretrained(hf_model_path)
        return hf_model_from_fms, hf_model

    @pytest.fixture
    def tokenizer(self, cases) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(os.path.join(cases, "tokenizer"))

    @pytest.fixture
    def signature(self, cases) -> nn.Module:
        return torch.load(os.path.join(cases, "signature.pth"))

    @property
    def _forward_parameters(self) -> int:
        return len(self._hf_forward_parameters) - 1

    @pytest.fixture
    def hf_model(self, hf_config: PretrainedConfig, model: nn.Module) -> PreTrainedModel:
        """create hf_model and load state dict from given pytorch native model then return hf_model"""
        return self._hf_model_class.from_fms_model(model, **hf_config.to_dict())

    @pytest.fixture
    def hf_config(self, config: ModelConfig, tokenizer: PreTrainedTokenizer) -> PretrainedConfig:
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        return self._hf_config_class.from_fms_config(
            config, eos_token_id=tokenizer.eos_token_id, bos_token_id=bos_token_id
        )

    @property
    @abc.abstractmethod
    def _hf_forward_parameters(self) -> List[str]:
        pass

    def _load_model_weights(self, model: nn.Module, hf_model: PreTrainedModel) -> PreTrainedModel:
        """Load state dict to hf and return the hf model"""
        hf_model = type(hf_model).from_fms_model(model, **hf_model.config.to_dict())
        return hf_model

    def test_hf_model_equivalence(self, hf_model, model):
        """test model signature equivalence between huggingface model and fms model"""

        # todo: This will be replaced with from_pytorch_model when other models have this in branch
        #  (no need to have abstract implementation)
        hf_model = type(hf_model).from_fms_model(model, **hf_model.config.to_dict())
        fms_signature_params = ModelSignatureParams(model, len(self._hf_forward_parameters) - 1)
        hf_signature_params = HFModelSignatureParams(hf_model, self._hf_forward_parameters)
        compare_model_signatures(fms_signature_params, hf_signature_params)

    text_options = [["hello how are you?"], ["hello how are you?", "a: this is a test. b: this is another test. a:"]]
    use_cache_options = [True, False, None]
    num_beams_options = [1, 3]
    generate_equivalence_args = list(itertools.product(text_options, use_cache_options, num_beams_options))

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
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts

    @pytest.mark.parametrize("texts,use_cache,num_beams", generate_equivalence_args)
    def test_hf_generate_equivalence(
        self,
        texts: List[str],
        use_cache: Optional[bool],
        num_beams: int,
        hf_models_loaded: Tuple[PreTrainedModel, PreTrainedModel],
        tokenizer: PreTrainedTokenizer,
    ):
        """test that an hf model created from fms and an hf model loaded from hf checkpoint produce the same output if
        they have the same weights and configs
        """
        hf_model_from_fms, hf_model_from_hf = hf_models_loaded

        output_fms = self._predict_text(hf_model_from_fms, tokenizer, texts, use_cache, num_beams)
        output_hf = self._predict_text(hf_model_from_hf, tokenizer, texts, use_cache, num_beams)

        assert output_fms == output_hf

    hf_batch_generate_args = list(itertools.product(use_cache_options, num_beams_options))

    @pytest.mark.parametrize("use_cache,num_beams", hf_batch_generate_args)
    def test_hf_batch_generate(
        self,
        use_cache,
        num_beams,
        hf_models_loaded: Tuple[PreTrainedModel, PreTrainedModel],
        tokenizer: PreTrainedTokenizer,
    ):
        """Tests that the output of a given prompt done alone and with batch generation is the same"""
        text_1 = "hello how are you?"
        text_2 = "a: this is a test. b: this is another test. a:"
        text_batch = [text_1, text_2]

        # required for batch generation
        tokenizer.padding_side = "left"

        _, hf_model = hf_models_loaded
        output_batch = self._predict_text(hf_model, tokenizer, text_batch, use_cache, num_beams)

        text1 = [text_1]
        output_text1 = self._predict_text(hf_model, tokenizer, text1, use_cache, num_beams)[0]

        text2 = [text_2]
        output_text2 = self._predict_text(hf_model, tokenizer, text2, use_cache, num_beams)[0]

        assert output_batch[0] == output_text1, f"text 1 incorrect - \n{output_batch[0]}\n{output_text1}"
        assert output_batch[1] == output_text2, f"text 2 incorrect - \n{output_batch[1]}\n{output_text2}"

    def test_hf_model_round_trip(self, hf_model: PreTrainedModel, hf_config: PretrainedConfig):
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
