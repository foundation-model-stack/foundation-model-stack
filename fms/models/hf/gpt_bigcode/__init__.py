import os
from typing import Union

import torch
from transformers import GPTBigCodeConfig, GPTBigCodeForCausalLM, PreTrainedModel

from fms.models.hf.gpt_bigcode.modeling_gpt_bigcode_hf import (
    HFAdaptedGPTBigCodeForCausalLM,
)


def convert_to_hf(
    fms_hf_model: HFAdaptedGPTBigCodeForCausalLM,
) -> GPTBigCodeForCausalLM:
    """
    Convert an HF-Adapted FMS GPTBigCode model to and HF model

    Parameters
    ----------
    fms_hf_model: HFAdaptedGPTBigCodeForCausalLM
        the HF-Adapted FMS GPTBigCode model

    Returns
    -------
    GPTBigCodeForCausalLM
        an HF equivalent model
    """
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
            n_positions=hf_config.max_expected_seq_len,
            scale_attention_softmax_in_fp32=False,
        )
    )

    with torch.no_grad():
        oss_hf_model.transformer.wte.weight.copy_(
            fms_hf_model.decoder.model.embedding.weight
        )
        oss_hf_model.transformer.wpe.weight.copy_(
            fms_hf_model.decoder.model.position_embedding.weight
        )
        for i, oss_hf_layer in enumerate(oss_hf_model.transformer.h):
            fms_hf_layer = fms_hf_model.decoder.model.layers[i]

            # self attn
            oss_hf_layer.attn.c_attn.weight.copy_(
                fms_hf_layer.attn.in_proj.qkv_fused.weight
            )
            oss_hf_layer.attn.c_attn.bias.copy_(
                fms_hf_layer.attn.in_proj.qkv_fused.bias
            )
            oss_hf_layer.attn.c_proj.weight.copy_(fms_hf_layer.attn.dense.weight)
            oss_hf_layer.attn.c_proj.bias.copy_(fms_hf_layer.attn.dense.bias)

            # mlp
            oss_hf_layer.mlp.c_fc.weight.copy_(fms_hf_layer.ff_sub_layer.w1.weight)
            oss_hf_layer.mlp.c_fc.bias.copy_(fms_hf_layer.ff_sub_layer.w1.bias)
            oss_hf_layer.mlp.c_proj.weight.copy_(fms_hf_layer.ff_sub_layer.w2.weight)
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
