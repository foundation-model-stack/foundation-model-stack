import platform

import numpy as np
import torch
from torch._dynamo.exc import TorchDynamoException
from torch._dynamo.testing import CompileCounterWithBackend

from fms.models.hf.modeling_hf_adapter import (
    HFEncoderDecoderModelArchitecture,
    HFDecoderModelArchitecture,
)
from fms.models.hf.utils import register_fms_models, to_hf_api
from fms.testing._internal.model_test_suite import ConfigFixtureMixin

SEED = 42
torch.manual_seed(SEED)  # pytorch random seed
np.random.seed(SEED)  # numpy random seed
torch.backends.cudnn.deterministic = True


import abc
import tempfile

import pytest
import torch
import torch.nn as nn
import numpy as np
import itertools
from typing import List, Optional

from transformers import (
    PreTrainedModel,
    AutoTokenizer,
    PreTrainedTokenizer,
    PretrainedConfig,
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

from fms.utils.config import ModelConfig
from ...comparison import (
    HFModelSignatureParams,
    ModelSignatureParams,
    compare_model_signatures,
    get_signature,
)


SEED = 42
torch.manual_seed(SEED)  # pytorch random seed
np.random.seed(SEED)  # numpy random seed
torch.backends.cudnn.deterministic = True


class HFConfigFixtureMixin(metaclass=abc.ABCMeta):
    """Mix this in with another AbstractResourcePath testing class to include the config and config_class fixtures"""

    # class specific fixtures
    @pytest.fixture(scope="class", autouse=True)
    def tokenizer(self) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained("google/byt5-small", padding_side="left")

    @abc.abstractmethod
    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_config(
        self,
        **kwargs,
    ) -> PretrainedConfig:
        """this fixture represents and fms hf config"""
        pass


class HFModelFixtureMixin(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_model(self, **kwargs) -> PreTrainedModel:
        """this fixture represents an fms hf model"""
        pass

    @abc.abstractmethod
    @pytest.fixture(scope="class", autouse=True)
    def oss_hf_model(self, fms_hf_model: PreTrainedModel) -> PreTrainedModel:
        """this fixture is the open source version of the fms hf model"""
        pass


class HFConfigTestSuite(ConfigFixtureMixin, HFConfigFixtureMixin):
    """General huggingface config testing class for future use with other models"""

    @property
    @abc.abstractmethod
    def _hf_specific_params(self) -> List[str]:
        """
        Returns
        -------
        List[str]
            a list of all parameters that are exclusive to HF (not part of FMS)
        """
        pass

    def test_hf_config_from_fms_config(
        self, config: ModelConfig, fms_hf_config: PretrainedConfig
    ):
        """Test that the config can save and load properly"""

        fms_hf_config_loaded = type(fms_hf_config).from_fms_config(config)
        fms_hf_config_loaded_dict = fms_hf_config_loaded.to_dict()
        fms_hf_config_dict = fms_hf_config.to_dict()
        # ignoring params that are HF specific
        for p in self._hf_specific_params:
            fms_hf_config_loaded_dict[p] = fms_hf_config_dict[p]
        assert fms_hf_config_dict == fms_hf_config_loaded_dict

    def test_hf_config_round_trip(self, fms_hf_config: PretrainedConfig):
        """Test that the config can save and load properly"""

        with tempfile.TemporaryDirectory() as workdir:
            fms_hf_config_path = f"{workdir}/hf_config.json"
            fms_hf_config.save_pretrained(fms_hf_config_path)
            fms_hf_config_loaded = type(fms_hf_config).from_pretrained(
                fms_hf_config_path
            )
            assert fms_hf_config.to_dict() == fms_hf_config_loaded.to_dict()

    def test_hf_autoconfig(self, fms_hf_config: PretrainedConfig):
        """test that the config can be loaded with autoconfig after registration"""
        register_fms_models()
        with tempfile.TemporaryDirectory() as workdir:
            fms_hf_config_path = f"{workdir}/hf_config.json"
            fms_hf_config.save_pretrained(fms_hf_config_path)
            new_config = AutoConfig.from_pretrained(fms_hf_config_path)
            assert isinstance(new_config, type(fms_hf_config))


class HFModelCompileTestSuite(HFModelFixtureMixin):
    """A set of tests associated with compilation of huggingface adapted fms models"""

    @property
    @abc.abstractmethod
    def _get_hf_signature_params(self) -> List[str]:
        """the value to pass into params in get_signature function for an hf model

        Returns
        -------
        List[str]
            the params to set to the default tensor value (inp) in get_signature. If an integer, will use *args, if a
            list, will use **kwargs
        """
        pass

    @pytest.mark.skipif(
        platform.system() != "Linux",
        reason=f"pytorch compile is more stable on Linux, skipping as current platform is {platform.platform()}",
    )
    def test_hf_model_compile_no_graph_breaks(self, fms_hf_model):
        """Test that an HF-FMS model is compilable without graph breaks"""
        try:
            torch._dynamo.reset()
            cnt = CompileCounterWithBackend("inductor")
            compiled_model = torch.compile(
                model=fms_hf_model, backend=cnt, fullgraph=True
            )
            fms_hf_signature_params = HFModelSignatureParams(
                compiled_model,
                self._get_hf_signature_params,
                # default attn_algorithm won't compile on CPU
                # TODO: add non-mmath attn_algorithm when we have GPUs to run unit tests
                other_params={"return_dict": True, "attn_algorithm": "math"},
            )
            assert cnt.frame_count == 0
            get_signature(
                model=fms_hf_signature_params.model,
                params=fms_hf_signature_params.params,
                optional_params=fms_hf_signature_params.other_params,
                logits_getter_fn=fms_hf_signature_params.logits_getter_fn,
            )
            assert cnt.frame_count == 1
        except TorchDynamoException as e:
            pytest.fail(f"Failed to get signature of full-graph compiled model:\n{e}")


class HFAutoModelTestSuite(HFModelFixtureMixin):
    def test_hf_automodel_headless(self, fms_hf_model: PreTrainedModel):
        """test that the headless model can be loaded with automodel after registration"""
        register_fms_models()
        with tempfile.TemporaryDirectory() as workdir:
            fms_hf_model_path = f"{workdir}/hf_model"
            fms_hf_model.save_pretrained(fms_hf_model_path)
            new_model = AutoModel.from_pretrained(fms_hf_model_path)
            assert (
                isinstance(fms_hf_model, type(new_model)) and new_model.lm_head is None
            )

    def test_hf_automodel_language_modeling_head(self, fms_hf_model: PreTrainedModel):
        """test that the language modeling head model can be loaded with automodel"""
        if isinstance(fms_hf_model, HFEncoderDecoderModelArchitecture):
            automodel_class = AutoModelForSeq2SeqLM
        elif isinstance(fms_hf_model, HFDecoderModelArchitecture):
            automodel_class = AutoModelForCausalLM
        else:
            pytest.skip(
                "encoder-only models do not perform text generation and therefore do not use AutoModelForCausalLM or AutoModelForSeq2SeqLM"
            )
        register_fms_models()
        with tempfile.TemporaryDirectory() as workdir:
            fms_hf_model_path = f"{workdir}/hf_model"
            fms_hf_model.save_pretrained(fms_hf_model_path)
            new_model = automodel_class.from_pretrained(fms_hf_model_path)
            assert isinstance(new_model, type(fms_hf_model))


class HFModelEquivalenceTestSuite(HFConfigFixtureMixin, HFModelFixtureMixin):
    """General huggingface model testing class for future use with other models"""

    @property
    @abc.abstractmethod
    def _get_hf_signature_params(self) -> List[str]:
        """the value to pass into params in get_signature function for an hf model

        Returns
        -------
        List[str]
            the params to set to the default tensor value (inp) in get_signature. If an integer, will use *args, if a
            list, will use **kwargs
        """
        pass

    # common tests
    def test_hf_and_fms_model_equivalence(self, fms_hf_model, model):
        """test model signature equivalence between huggingface model and fms model"""

        _fms_hf_model = to_hf_api(model, **fms_hf_model.config.to_dict())
        fms_signature_params = ModelSignatureParams(
            model, len(self._get_hf_signature_params) - 1
        )
        fms_hf_signature_params = HFModelSignatureParams(
            _fms_hf_model, self._get_hf_signature_params
        )
        compare_model_signatures(fms_signature_params, fms_hf_signature_params)

    def test_hf_and_oss_hf_model_equivalence(self, fms_hf_model, oss_hf_model):
        inp = torch.arange(5, 15).unsqueeze(0)
        fms_hf_signature_params = HFModelSignatureParams(
            fms_hf_model, self._get_hf_signature_params, inp=inp
        )
        oss_hf_signature_params = HFModelSignatureParams(
            oss_hf_model, self._get_hf_signature_params, inp=inp
        )
        compare_model_signatures(fms_hf_signature_params, oss_hf_signature_params)

    def test_hf_from_fms_and_hf_from_pretrained_equivalence(
        self, tmpdir_factory, model: nn.Module, fms_hf_model: PreTrainedModel
    ):
        hf_path = tmpdir_factory.mktemp("hf")
        _fms_hf_model = type(fms_hf_model).from_fms_model(
            model, **fms_hf_model.config.to_dict()
        )

        _fms_hf_model.save_pretrained(hf_path)

        fms_hf_model_from_fms = type(_fms_hf_model).from_fms_model(
            model, **_fms_hf_model.config.to_dict()
        )
        _fms_hf_model = type(_fms_hf_model).from_pretrained(hf_path)

        fms_hf_from_fms_signature_params = HFModelSignatureParams(
            fms_hf_model_from_fms, self._get_hf_signature_params
        )
        fms_hf_signature_params = HFModelSignatureParams(
            _fms_hf_model, self._get_hf_signature_params
        )
        compare_model_signatures(
            fms_hf_from_fms_signature_params, fms_hf_signature_params
        )

    def test_hf_model_round_trip_equivalence(
        self, fms_hf_model: PreTrainedModel, fms_hf_config: PretrainedConfig
    ):
        """Test that the huggingface model can save and load properly"""
        fms_hf_model_from_config = type(fms_hf_model)(fms_hf_config)
        fms_hf_model_from_config.load_state_dict(fms_hf_model.state_dict())
        compare_model_signatures(
            HFModelSignatureParams(fms_hf_model, self._get_hf_signature_params),
            HFModelSignatureParams(
                fms_hf_model_from_config, self._get_hf_signature_params
            ),
        )

        with tempfile.TemporaryDirectory() as workdir:
            fms_hf_model_path = f"{workdir}/hf_model"
            fms_hf_model_from_config.save_pretrained(fms_hf_model_path)
            fms_hf_model_loaded = type(fms_hf_model).from_pretrained(fms_hf_model_path)

        compare_model_signatures(
            HFModelSignatureParams(fms_hf_model, self._get_hf_signature_params),
            HFModelSignatureParams(fms_hf_model_loaded, self._get_hf_signature_params),
        )


class HFModelGenerationTestSuite(HFConfigFixtureMixin, HFModelFixtureMixin):
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
                pad_token_id=model.config.pad_token_id,
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
        fms_hf_model: PreTrainedModel,
        oss_hf_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """test that an hf model created from fms and an hf model loaded from hf checkpoint produce the same output if
        they have the same weights and configs
        """
        print(texts)
        output_fms = self._predict_text(
            fms_hf_model, tokenizer, texts, use_cache, num_beams
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
        fms_hf_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        """Tests that the output of a given prompt done alone and with batch generation is the same"""
        text_1 = "hello how are you?"
        text_2 = "a: this is a test. b: this is another test. a:"
        text_batch = [text_1, text_2]

        output_batch = self._predict_text(
            fms_hf_model, tokenizer, text_batch, use_cache, num_beams
        )

        text1 = [text_1]
        output_text1 = self._predict_text(
            fms_hf_model, tokenizer, text1, use_cache, num_beams
        )[0]

        text2 = [text_2]
        output_text2 = self._predict_text(
            fms_hf_model, tokenizer, text2, use_cache, num_beams
        )[0]

        assert (
            output_batch[0] == output_text1
        ), f"text 1 incorrect - \n{output_batch[0]}\n{output_text1}"
        assert (
            output_batch[1] == output_text2
        ), f"text 2 incorrect - \n{output_batch[1]}\n{output_text2}"
