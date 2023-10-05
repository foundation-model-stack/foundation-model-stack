import pytest
from torch import nn as nn
from transformers import (
    PreTrainedModel,
    GPTBigCodeForCausalLM,
    GPTBigCodeConfig,
    PretrainedConfig,
    PreTrainedTokenizer,
)
import torch

from fms.models.gpt_bigcode import GPTBigCode
from fms.models.hf.gpt_bigcode.configuration_gpt_bigcode_hf import GPTBigCodeHFConfig
from fms.models.hf.gpt_bigcode.modeling_gpt_bigcode_hf import GPTBigCodeHFForCausalLM
from fms.testing._internal.hf.model_test_suite import (
    HFConfigFixtureMixin,
    HFModelFixtureMixin,
    HFConfigTestSuite,
    HFModelEquivalenceTestSuite,
    HFModelGenerationTestSuite,
)
from fms.testing._internal.model_test_suite import ModelFixtureMixin
from ..test_gpt_bigcode import GPTBigCodeFixtures


class GPTBigCodeHFFixtures(
    ModelFixtureMixin, HFConfigFixtureMixin, HFModelFixtureMixin
):
    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_model(
        self, model: GPTBigCode, fms_hf_config: PretrainedConfig, **kwargs
    ):
        return GPTBigCodeHFForCausalLM.from_fms_model(model, **fms_hf_config.to_dict())

    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_config(
        self, tokenizer: PreTrainedTokenizer, model: GPTBigCode, **kwargs
    ) -> PretrainedConfig:
        bos_token_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.eos_token_id
        )
        return GPTBigCodeHFConfig.from_fms_config(
            model.get_config(),
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=bos_token_id,
        )

    @pytest.fixture(scope="class", autouse=True)
    def oss_hf_model(self, fms_hf_model: GPTBigCodeHFForCausalLM) -> PreTrainedModel:
        hf_config = fms_hf_model.config
        oss_hf_model = GPTBigCodeForCausalLM(
            GPTBigCodeConfig(
                vocab_size=hf_config.vocab_size,
                n_embd=hf_config.hidden_size,
                layer_norm_epsilon=hf_config.ln_eps,
                n_head=hf_config.nheads,
                n_layer=hf_config.nlayers,
                n_inner=int(hf_config.hidden_size * hf_config.hidden_grow_factor),
                pad_token_id=hf_config.pad_token_id,
                bos_token_id=hf_config.bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                n_positions=hf_config.max_pos,
                scale_attention_softmax_in_fp32=False,
            )
        )
        """
        GPTBigCodeForCausalLM(
          (transformer): GPTBigCodeModel(
            (wte): Embedding(384, 16)
            (wpe): Embedding(512, 16)
            (drop): Dropout(p=0.1, inplace=False)
            (h): ModuleList(
              (0-1): 2 x GPTBigCodeBlock(
                (ln_1): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
                (attn): GPTBigCodeAttention(
                  (c_attn): Linear(in_features=16, out_features=20, bias=True)
                  (c_proj): Linear(in_features=16, out_features=16, bias=True)
                  (attn_dropout): Dropout(p=0.1, inplace=False)
                  (resid_dropout): Dropout(p=0.1, inplace=False)
                )
                (ln_2): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
                (mlp): GPTBigCodeMLP(
                  (c_fc): Linear(in_features=16, out_features=32, bias=True)
                  (c_proj): Linear(in_features=32, out_features=16, bias=True)
                  (act): PytorchGELUTanh()
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
            )
            (ln_f): LayerNorm((16,), eps=1e-05, elementwise_affine=True)
          )
          (lm_head): Linear(in_features=16, out_features=384, bias=False)
        )
        """

        with torch.no_grad():

            oss_hf_model.transformer.wte.weight.copy_(
                fms_hf_model.decoder.model.shared.emb.weight
            )
            oss_hf_model.transformer.wpe.weight.copy_(
                fms_hf_model.decoder.model.shared.pos_emb.weight
            )
            for i, oss_hf_layer in enumerate(oss_hf_model.transformer.h):
                fms_hf_layer = fms_hf_model.decoder.model.layers[i]

                # self attn
                oss_hf_layer.attn.c_attn.weight.copy_(
                    torch.cat(
                        [
                            fms_hf_layer.attn.query.weight.data,
                            fms_hf_layer.attn.key.weight.data,
                            fms_hf_layer.attn.value.weight.data,
                        ],
                        dim=0,
                    )
                )
                oss_hf_layer.attn.c_attn.bias.copy_(
                    torch.cat(
                        [
                            fms_hf_layer.attn.query.bias.data,
                            fms_hf_layer.attn.key.bias.data,
                            fms_hf_layer.attn.value.bias.data,
                        ],
                        dim=0,
                    )
                )
                oss_hf_layer.attn.c_proj.weight.copy_(fms_hf_layer.attn.dense.weight)
                oss_hf_layer.attn.c_proj.bias.copy_(fms_hf_layer.attn.dense.bias)

                # mlp
                oss_hf_layer.mlp.c_fc.weight.copy_(fms_hf_layer.ff_sub_layer.w1.weight)
                oss_hf_layer.mlp.c_fc.bias.copy_(fms_hf_layer.ff_sub_layer.w1.bias)
                oss_hf_layer.mlp.c_proj.weight.copy_(
                    fms_hf_layer.ff_sub_layer.w2.weight
                )
                oss_hf_layer.mlp.c_proj.bias.copy_(fms_hf_layer.ff_sub_layer.w2.bias)

                # layer norm
                oss_hf_layer.ln_1.weight.copy_(fms_hf_layer.ln.weight)
                oss_hf_layer.ln_1.bias.copy_(fms_hf_layer.ln.bias)
                oss_hf_layer.ln_2.weight.copy_(fms_hf_layer.ff_ln.weight)
                oss_hf_layer.ln_2.bias.copy_(fms_hf_layer.ff_ln.bias)
            oss_hf_model.transformer.ln_f.weight.copy_(
                fms_hf_model.decoder.model.dec_norm.weight
            )
            oss_hf_model.transformer.ln_f.bias.copy_(
                fms_hf_model.decoder.model.dec_norm.bias
            )

        return oss_hf_model


class TestGPTBigCodeHF(
    HFConfigTestSuite,
    HFModelEquivalenceTestSuite,
    HFModelGenerationTestSuite,
    GPTBigCodeFixtures,
    GPTBigCodeHFFixtures,
):
    """
    LLaMA2 FMS Huggingface Tests for:

    - FMS Huggingface configuration tests
    - model equivalency tests
    - model generation tests
    """

    # implementation of abstract property _hf_specific_params
    _hf_specific_params = ["eos_token_id", "bos_token_id"]
    # implementation of abstract property _get_hf_signature_params
    _get_hf_signature_params = ["input_ids", "labels"]
