import torch
from transformers import GraniteConfig, GraniteForCausalLM

from fms.models.hf.granite.modeling_granite_hf import (
    HFAdaptedGraniteConfig,
    HFAdaptedGraniteForCausalLM,
)


def convert_to_hf(
    fms_hf_model: HFAdaptedGraniteForCausalLM,
) -> GraniteForCausalLM:
    """
    Convert an HF-Adapted FMS Granite model to an HF model

    Parameters
    ----------
    fms_hf_model: HFAdaptedGraniteForCausalLM
        the HF-Adapted FMS Granite model

    Returns
    -------
    GraniteForCausalLM
        an HF equivalent model
    """
    hf_config: HFAdaptedGraniteConfig = fms_hf_model.config
    oss_hf_model = GraniteForCausalLM(
        GraniteConfig(
            vocab_size=hf_config.src_vocab_size,
            hidden_size=hf_config.emb_dim,
            intermediate_size=hf_config.hidden_dim,
            num_hidden_layers=hf_config.nlayers,
            num_attention_heads=hf_config.nheads,
            num_key_value_heads=hf_config.kvheads,
            max_position_embedings=hf_config.max_expected_seq_len,
            rms_norm_eps=hf_config.norm_eps,
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            rope_theta=hf_config.rope_theta,
            attention_dropout=hf_config.p_dropout,
        )
    )

    with torch.no_grad():
        oss_hf_model.model.embed_tokens.weight.copy_(
            fms_hf_model.decoder.model.embedding.weight
        )
        for i, oss_hf_layer in enumerate(oss_hf_model.model.layers):
            fms_hf_layer = fms_hf_model.decoder.model.layers[i]
            hf_q, hf_k, hf_v = torch.split(
                fms_hf_layer.attn.in_proj.qkv_fused.weight,
                fms_hf_layer.attn.in_proj.splits,
            )

            # self attn (+ HF RoPE transpose)
            hf_q = (
                hf_q.view(hf_config.nheads, 2, -1, hf_q.size(1))
                .transpose(1, 2)
                .reshape(*hf_q.size())
            )
            oss_hf_layer.self_attn.q_proj.weight.copy_(hf_q)
            hf_k = (
                hf_k.view(hf_config.kvheads, 2, -1, hf_k.size(1))
                .transpose(1, 2)
                .reshape(*hf_k.size())
            )
            oss_hf_layer.self_attn.k_proj.weight.copy_(hf_k)
            oss_hf_layer.self_attn.v_proj.weight.copy_(hf_v)
            oss_hf_layer.self_attn.o_proj.weight.copy_(fms_hf_layer.attn.dense.weight)

            # mlp
            wg1_fused = fms_hf_layer.ff_sub_layer.wg1_fused.weight
            wg_splits = [wg1_fused.size(0) // 2, wg1_fused.size(0) // 2]
            w1, wg = torch.split(
                fms_hf_layer.ff_sub_layer.wg1_fused.weight, wg_splits, dim=0
            )
            oss_hf_layer.mlp.gate_proj.weight.copy_(wg)
            oss_hf_layer.mlp.up_proj.weight.copy_(w1)
            oss_hf_layer.mlp.down_proj.weight.copy_(fms_hf_layer.ff_sub_layer.w2.weight)

            # layer norm
            oss_hf_layer.input_layernorm.weight.copy_(fms_hf_layer.ln.weight)
            oss_hf_layer.post_attention_layernorm.weight.copy_(
                fms_hf_layer.ff_ln.weight
            )

        # LM Head
        oss_hf_model.model.norm.weight.copy_(fms_hf_model.decoder.model.dec_norm.weight)
        oss_hf_model.lm_head.weight.copy_(fms_hf_model.lm_head.weight)

    return oss_hf_model
