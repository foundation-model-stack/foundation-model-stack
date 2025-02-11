import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from fms.models.hf.llama.configuration_llama_hf import HFAdaptedLLaMAConfig
from fms.models.hf.llama.modeling_llama_hf import HFAdaptedLLaMAForCausalLM
from fms.models.llama import LLaMA
from fms.testing._internal.hf.model_test_suite import (
    HFAutoModelTestSuite,
    HFConfigFixtureMixin,
    HFConfigTestSuite,
    HFModelCompileTestSuite,
    HFModelEquivalenceTestSuite,
    HFModelFixtureMixin,
    HFModelGenerationTestSuite,
)
from fms.testing._internal.model_test_suite import ModelFixtureMixin

from ..test_llama import LLaMA2Fixtures, LLaMA2GQAFixtures


class LLaMA2HFFixtures(ModelFixtureMixin, HFConfigFixtureMixin, HFModelFixtureMixin):
    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_model(self, model: LLaMA, fms_hf_config: PretrainedConfig, **kwargs):
        return HFAdaptedLLaMAForCausalLM.from_fms_model(
            model, **fms_hf_config.to_dict()
        )

    @pytest.fixture(scope="class", autouse=True)
    def fms_hf_config(
        self, tokenizer: PreTrainedTokenizer, model: LLaMA, **kwargs
    ) -> PretrainedConfig:
        bos_token_id = (
            tokenizer.bos_token_id
            if tokenizer.bos_token_id is not None
            else tokenizer.eos_token_id
        )
        return HFAdaptedLLaMAConfig.from_fms_config(
            model.get_config(),
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=bos_token_id,
        )

    @pytest.fixture(scope="class", autouse=True)
    def oss_hf_model(self, fms_hf_model: HFAdaptedLLaMAForCausalLM) -> PreTrainedModel:
        hf_config = fms_hf_model.config
        oss_hf_model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=hf_config.vocab_size,
                hidden_size=hf_config.hidden_size,
                rms_norm_eps=hf_config.norm_eps,
                num_attention_heads=hf_config.nheads,
                num_key_value_heads=(
                    None if hf_config.kvheads == 0 else hf_config.kvheads
                ),
                num_hidden_layers=hf_config.nlayers,
                pad_token_id=hf_config.pad_token_id,
                intermediate_size=int(
                    hf_config.hidden_size * hf_config.hidden_grow_factor
                ),
                bos_token_id=hf_config.bos_token_id,
                eos_token_id=hf_config.eos_token_id,
                max_position_embeddings=hf_config.max_expected_seq_len,
            )
        )

        # compute the freq from rot_emb since it is gathered lazily
        rot_emb = fms_hf_model.decoder.model.rot_emb
        max_seq_len = rot_emb.max_seq_len
        alpha = rot_emb._alpha(max_seq_len)
        ratio = rot_emb.ratio
        dim = rot_emb.dim
        if rot_emb.ntk_scaling:
            ratio = ratio * alpha ** (dim / (dim - 2))
        freqs = 1.0 / (ratio ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

        with torch.no_grad():
            oss_hf_model.model.embed_tokens.weight.copy_(fms_hf_model.embedding.weight)
            oss_hf_model.model.rotary_emb.inv_freqs = freqs
            i = 0
            for oss_hf_layer in oss_hf_model.model.layers:
                fms_hf_layer = fms_hf_model.decoder.model.layers[i]

                # self attn
                q, k, v = torch.split(
                    fms_hf_layer.attn.in_proj.qkv_fused.weight,
                    fms_hf_layer.attn.in_proj.splits,
                    dim=0,
                )
                oss_hf_layer.self_attn.q_proj.weight.copy_(q)
                oss_hf_layer.self_attn.k_proj.weight.copy_(k)
                oss_hf_layer.self_attn.v_proj.weight.copy_(v)
                oss_hf_layer.self_attn.o_proj.weight.copy_(
                    fms_hf_layer.attn.dense.weight
                )

                # mlp
                wg1_fused = fms_hf_layer.ff_sub_layer.wg1_fused.weight
                wg_splits = [wg1_fused.size(0) // 2, wg1_fused.size(0) // 2]
                w1, wg = torch.split(
                    fms_hf_layer.ff_sub_layer.wg1_fused.weight, wg_splits, dim=0
                )
                oss_hf_layer.mlp.gate_proj.weight.copy_(wg)
                oss_hf_layer.mlp.up_proj.weight.copy_(w1)
                oss_hf_layer.mlp.down_proj.weight.copy_(
                    fms_hf_layer.ff_sub_layer.w2.weight
                )

                # layer norm
                oss_hf_layer.input_layernorm.weight.copy_(fms_hf_layer.ln.weight)
                oss_hf_layer.post_attention_layernorm.weight.copy_(
                    fms_hf_layer.ff_ln.weight
                )

                # adjust q, k
                q = oss_hf_layer.self_attn.q_proj.weight.data
                q = (
                    q.view(hf_config.nheads, -1, 2, q.size(1))
                    .transpose(1, 2)
                    .reshape(*q.size())
                )
                oss_hf_layer.self_attn.q_proj.weight.copy_(q)

                k = oss_hf_layer.self_attn.k_proj.weight.data
                k = (
                    k.view(hf_config.nheads, -1, 2, k.size(1))
                    .transpose(1, 2)
                    .reshape(*k.size())
                )
                oss_hf_layer.self_attn.k_proj.weight.copy_(k)

                i = i + 1
            oss_hf_model.model.norm.weight = fms_hf_model.decoder.model.dec_norm.weight
            oss_hf_model.lm_head.weight = fms_hf_model.lm_head.weight

        return oss_hf_model


class TestLLaMA2HF(
    HFConfigTestSuite,
    HFModelEquivalenceTestSuite,
    HFModelGenerationTestSuite,
    HFModelCompileTestSuite,
    HFAutoModelTestSuite,
    LLaMA2Fixtures,
    LLaMA2HFFixtures,
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


class TestLLaMA2GQAHF(
    HFModelEquivalenceTestSuite,
    HFModelGenerationTestSuite,
    HFModelCompileTestSuite,
    HFAutoModelTestSuite,
    LLaMA2GQAFixtures,
    LLaMA2HFFixtures,
):
    """
    LLaMA2-GQA FMS Huggingface Tests for:

    - model equivalency tests
    - model generation tests
    """

    # implementation of abstract property _hf_specific_params
    _hf_specific_params = ["eos_token_id", "bos_token_id"]
    # implementation of abstract property _get_hf_signature_params
    _get_hf_signature_params = ["input_ids", "labels"]
