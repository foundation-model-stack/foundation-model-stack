import torch

from transformers import GptOssConfig, GptOssForCausalLM

from fms.models.hf.gpt_oss.modeling_gpt_oss_hf import (
    HFAdaptedGptOssForCausalLM,
)


def convert_to_hf(
    fms_hf_model: HFAdaptedGptOssForCausalLM,
) -> GptOssForCausalLM:
    """
    Convert an HF-Adapted FMS GptOss model to an HF model

    Parameters
    ----------
    fms_hf_model: HFAdaptedGptOssForCausalLM
        the HF-Adapted FMS GptOss model

    Returns
    -------
    GptOssForCausalLM,
        an HF equivalent model
    """
    oss_hf_model = GptOssForCausalLM(
        GptOssConfig(
            head_dim=64,
            norm_eps=1e-05,
            num_attention_heads=64,
            num_key_value_heads=8,
            num_hidden_layers=4,
            num_experts_per_tok=4,
            num_local_experts=32,
            rope_base=150000.0,
            rope_scaling_factor=32.0,
            rope_ntk_alpha=1.0,
            rope_ntk_beta=32.0,
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
            # Expect splits == [q, k, v] and RoPE interleaving [even, odd] per head.
            assert len(fms_hf_layer.attn.in_proj.splits) == 3, (
                "Expected [q,k,v] fused layout"
            )

            # self attn (+ HF RoPE transpose)
            hf_q = (
                hf_q.view(16, 2, -1, hf_q.size(1)).transpose(1, 2).reshape(*hf_q.size())
            )
            oss_hf_layer.self_attn.q_proj.weight.copy_(hf_q)

            hf_k = (
                hf_k.view(1, 2, -1, hf_k.size(1)).transpose(1, 2).reshape(*hf_k.size())
            )
            oss_hf_layer.self_attn.k_proj.weight.copy_(hf_k)
            oss_hf_layer.self_attn.v_proj.weight.copy_(hf_v)

            oss_hf_layer.self_attn.o_proj.weight.copy_(fms_hf_layer.attn.dense.weight)

            # sinks
            oss_hf_layer.self_attn.sinks.copy_(fms_hf_layer.attn.sinks)

            # MoE SwiGLU
            oss_hf_layer.mlp.router.weight.copy_(fms_hf_layer.ff_sub_layer.gate.weight)

            oss_hf_layer.mlp.experts.gate_up_proj.copy_(
                fms_hf_layer.ff_sub_layer.cond_ffn.w13.transpose(1, 2)
            )
            oss_hf_layer.mlp.experts.down_proj.copy_(
                fms_hf_layer.ff_sub_layer.cond_ffn.w2
            )
            oss_hf_layer.mlp.experts.gate_up_proj_bias.copy_(
                fms_hf_layer.ff_sub_layer.cond_ffn.w13_bias
            )
            oss_hf_layer.mlp.experts.down_proj_bias.copy_(
                fms_hf_layer.ff_sub_layer.cond_ffn.w2_bias
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
