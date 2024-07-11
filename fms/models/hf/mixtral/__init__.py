import torch
from transformers import MixtralConfig, MixtralForCausalLM

from fms.models.hf.mixtral.modeling_mixtral_hf import (
    HFAdaptedMixtralConfig,
    HFAdaptedMixtralForCausalLM,
)


def convert_to_hf(
    fms_hf_model: HFAdaptedMixtralForCausalLM,
) -> MixtralForCausalLM:
    """
    Convert an HF-Adapted FMS Mixtral model to an HF model

    Parameters
    ----------
    fms_hf_model: HFAdaptedMixtralForCausalLM
        the HF-Adapted FMS Mixtral model

    Returns
    -------
    GPTBigCodeForCausalLM
        an HF equivalent model
    """
    hf_config: HFAdaptedMixtralConfig = fms_hf_model.config
    oss_hf_model = MixtralForCausalLM(
        MixtralConfig(
            vocab_size=hf_config.src_vocab_size,
            hidden_size=hf_config.dim,
            intermediate_size=hf_config.hidden_dim,
            num_hidden_layers=hf_config.nlayers,
            num_attention_heads=hf_config.nheads,
            num_key_value_heads=hf_config.kvheads,
            max_position_embedings=hf_config.max_expected_seq_len,
            rms_norm_eps=hf_config.norm_eps,
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            rope_theta=hf_config.rope_base,
            attention_dropout=hf_config.p_dropout,
            num_experts_per_tok=hf_config.top_k_experts,
            num_local_experts=hf_config.num_experts,
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

            # MoE SwiGLU
            oss_hf_layer.block_sparse_moe.gate.weight.copy_(
                fms_hf_layer.ff_sub_layer.gate.weight
            )
            for expert_idx, expert_layer in enumerate(
                oss_hf_layer.block_sparse_moe.experts
            ):
                expert_layer.w1.weight.copy_(
                    fms_hf_layer.ff_sub_layer.cond_ffn.w13.chunk(2, dim=1)[0][
                        expert_idx
                    ]
                )
                expert_layer.w3.weight.copy_(
                    fms_hf_layer.ff_sub_layer.cond_ffn.w13.chunk(2, dim=1)[1][
                        expert_idx
                    ]
                )
                expert_layer.w2.weight.copy_(
                    fms_hf_layer.ff_sub_layer.cond_ffn.w2[expert_idx]
                )

            # layer norm
            oss_hf_layer.input_layernorm.weight.copy_(fms_hf_layer.ln.weight)
            oss_hf_layer.post_attention_layernorm.weight.copy_(
                fms_hf_layer.ff_ln.weight
            )

        # LM Head
        oss_hf_model.model.norm.weight.copy_(fms_hf_model.decoder.model.dec_norm.weight)
        oss_hf_model.lm_head.weight.copy_(fms_hf_model.lm_head.weight)

    return oss_hf_model
