import pytest
import torch
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizer,
    RobertaForMaskedLM,
    RobertaConfig,
)

from fms.models.roberta import RoBERTa
from fms.models.hf.roberta.modeling_roberta_hf import (
    HFAdaptedRoBERTaConfig,
    HFAdaptedRoBERTaForMaskedLM,
)
from fms.testing._internal.hf.model_test_suite import (
    HFConfigFixtureMixin,
    HFModelFixtureMixin,
    HFConfigTestSuite,
    HFModelEquivalenceTestSuite,
    HFModelCompileTestSuite,
)
from fms.testing._internal.model_test_suite import ModelFixtureMixin
from fms.models.hf import to_hf_api
from ..test_roberta import RoBERTaFixtures


class HFAdaptedRoBERTaFixtures(
    ModelFixtureMixin, HFConfigFixtureMixin, HFModelFixtureMixin
):
    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_model(self, model: RoBERTa, fms_hf_config: PretrainedConfig, **kwargs):
        return to_hf_api(model, **fms_hf_config.to_dict())

    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_config(
        self, tokenizer: PreTrainedTokenizer, model: RoBERTa, **kwargs
    ) -> PretrainedConfig:
        bos_token_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.eos_token_id
        )
        return HFAdaptedRoBERTaConfig.from_fms_config(
            model.get_config(),
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=bos_token_id,
        )

    @pytest.fixture(scope="class", autouse=True)
    def oss_hf_model(
        self, fms_hf_model: HFAdaptedRoBERTaForMaskedLM
    ) -> PreTrainedModel:
        hf_config = fms_hf_model.config
        oss_hf_model = RobertaForMaskedLM(
            RobertaConfig(
                vocab_size=hf_config.vocab_size,
                hidden_size=hf_config.hidden_size,
                rms_norm_eps=hf_config.norm_eps,
                num_attention_heads=hf_config.nheads,
                num_hidden_layers=hf_config.nlayers,
                pad_token_id=hf_config.pad_token_id,
                intermediate_size=int(
                    hf_config.hidden_size * hf_config.hidden_grow_factor
                ),
                bos_token_id=hf_config.bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                max_position_embeddings=hf_config.max_pos,
            )
        )

        with torch.no_grad():

            def _copy_weight_bias(hf_module, fms_module):
                hf_module.weight.copy_(fms_module.weight)
                hf_module.bias.copy_(fms_module.bias)

            # process embeddings
            oss_hf_model.roberta.embeddings.word_embeddings.weight.copy_(
                fms_hf_model.encoder.model.embedding.weight
            )
            oss_hf_model.roberta.embeddings.position_embeddings.weight.copy_(
                fms_hf_model.encoder.model.position_embedding.weight
            )

            for i, fms_layer in enumerate(fms_hf_model.encoder.model.layers):
                oss_hf_layer = oss_hf_model.roberta.encoder.layer[i]

                # layer norm
                _copy_weight_bias(oss_hf_layer.attention.output.LayerNorm, fms_layer.ln)
                _copy_weight_bias(oss_hf_layer.output.LayerNorm, fms_layer.ff_ln)

                # attn
                _copy_weight_bias(
                    oss_hf_layer.attention.self.query, fms_layer.attn.query
                )
                _copy_weight_bias(oss_hf_layer.attention.self.key, fms_layer.attn.key)
                _copy_weight_bias(
                    oss_hf_layer.attention.self.value, fms_layer.attn.value
                )
                _copy_weight_bias(
                    oss_hf_layer.attention.output.dense, fms_layer.attn.dense
                )

                # ff
                _copy_weight_bias(
                    oss_hf_layer.intermediate.dense, fms_layer.ff_sub_layer.w1
                )
                _copy_weight_bias(oss_hf_layer.output.dense, fms_layer.ff_sub_layer.w2)

            # process model layer norm
            _copy_weight_bias(
                oss_hf_model.roberta.embeddings.LayerNorm,
                fms_hf_model.encoder.model.enc_norm,
            )

            # process model head
            _copy_weight_bias(oss_hf_model.lm_head.dense, fms_hf_model.lm_head[0].dense)
            _copy_weight_bias(
                oss_hf_model.lm_head.layer_norm, fms_hf_model.lm_head[0].ln
            )
            # oss_hf_model.lm_head.decoder.bias.copy_(fms_hf_model.lm_head[1].bias)
        return oss_hf_model


class TestHFAdaptedRoBERTa(
    HFConfigTestSuite,
    HFModelEquivalenceTestSuite,
    HFModelCompileTestSuite,
    RoBERTaFixtures,
    HFAdaptedRoBERTaFixtures,
):
    """
    RoBERTa FMS Huggingface Tests for:

    - FMS Huggingface configuration tests
    - model equivalency tests
    - model generation tests
    """

    # implementation of abstract property _hf_specific_params
    _hf_specific_params = ["eos_token_id", "bos_token_id"]
    # implementation of abstract property _get_hf_signature_params
    _get_hf_signature_params = ["input_ids", "labels"]
