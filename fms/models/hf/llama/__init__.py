import os
from typing import Union

from transformers import LlamaForCausalLM, LlamaConfig

from fms.models.hf import to_hf_api
from fms.models.hf.llama.modeling_llama_hf import HFAdaptedLLaMAForCausalLM
import torch.nn as nn
import torch

from fms.models.llama import LLaMA


def get_model(model_name_or_path: Union[str, os.PathLike]) -> HFAdaptedLLaMAForCausalLM:
    """
    Get a Huggingface adapted FMS model from an equivalent HF model

    Parameters
    ----------
    model_name_or_path: Union[str, os.PathLike]
        Either the name of the model in huggingface hub or the absolute path to
        the huggingface model

    Returns
    -------
    HFAdaptedLLaMAForCausalLM
        A Huggingface adapted FMS implementation of LLaMA
    """
    import torch
    from fms.models.hf.utils import register_fms_models
    from fms.models.llama import convert_hf_llama
    from transformers import LlamaForCausalLM

    register_fms_models()
    hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path)
    fms_model = convert_hf_llama(hf_model).half()
    result_model: HFAdaptedLLaMAForCausalLM = HFAdaptedLLaMAForCausalLM.from_fms_model(
        fms_model,
        torch_dtype=torch.float16,
        # pad_token_id in fms is defaulted to -1
        # in generation, huggingface will add pad_tokens to the end of a sequence after the eos token is found for a
        # given sequence in the batch, if -1 is provided, our model won't be able to interpret it. We should be using
        # huggingface pad_token_id as that is where the model weights are coming from.
        pad_token_id=hf_model.config.pad_token_id,
    )
    return result_model

def convert_to_hf(model: Union[LLaMA, HFAdaptedLLaMAForCausalLM]) -> LlamaForCausalLM:
    """convert an fms or fms hf-adapted causal-lm llama model to an open-source transformers LlamaForCausalLM model

    Parameters
    ----------
    model: Union[LLaMA, HFAdaptedLLaMAForCausalLM]
        the fms model to convert

    Returns
    -------
    LlamaForCausalLM
        an transformers-based LlamaForCausalLM model
    """
    if isinstance(model, LLaMA):
        fms_hf_model = to_hf_api(model)
    else:
        fms_hf_model = model

    hf_config = fms_hf_model.config
    oss_hf_model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            rms_norm_eps=hf_config.norm_eps,
            num_attention_heads=hf_config.nheads,
            num_key_value_heads=None
            if hf_config.kvheads == 0
            else hf_config.kvheads,
            num_hidden_layers=hf_config.nlayers,
            intermediate_size=hf_config.multiple_of * round(int(
                hf_config.hidden_size * hf_config.hidden_grow_factor
            ) / hf_config.multiple_of),
            pad_token_id=None if hf_config.pad_token_id == -1 else hf_config.pad_token_id,
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
        i = 0
        for oss_hf_layer in oss_hf_model.model.layers:
            fms_hf_layer = fms_hf_model.decoder.model.layers[i]

            # self attn
            oss_hf_layer.self_attn.q_proj.weight.copy_(
                fms_hf_layer.attn.query.weight
            )
            oss_hf_layer.self_attn.k_proj.weight.copy_(fms_hf_layer.attn.key.weight)
            oss_hf_layer.self_attn.v_proj.weight.copy_(
                fms_hf_layer.attn.value.weight
            )
            oss_hf_layer.self_attn.o_proj.weight.copy_(
                fms_hf_layer.attn.dense.weight
            )
            oss_hf_layer.self_attn.rotary_emb.inv_freqs = freqs

            # mlp
            oss_hf_layer.mlp.gate_proj.weight.copy_(
                fms_hf_layer.ff_sub_layer.wg.weight
            )
            oss_hf_layer.mlp.up_proj.weight.copy_(
                fms_hf_layer.ff_sub_layer.w1.weight
            )
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
