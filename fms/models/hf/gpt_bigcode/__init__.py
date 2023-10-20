import os
from typing import Union

from fms.models.hf.gpt_bigcode.modeling_gpt_bigcode_hf import (
    HFAdaptedGPTBigCodeForCausalLM,
)


def get_model(
    model_name_or_path: Union[str, os.PathLike]
) -> HFAdaptedGPTBigCodeForCausalLM:
    """
    Get a Huggingface adapted FMS model from an equivalent HF model

    Parameters
    ----------
    model_name_or_path: Union[str, os.PathLike]
        Either the name of the model in huggingface hub or the absolute path to
        the huggingface model

    Returns
    -------
    HFAdaptedGPTBigCodeForCausalLM
        A Huggingface adapted FMS implementation of GPT-BigCode
    """
    from fms.models.hf.utils import register_fms_models
    from transformers import AutoModelForCausalLM
    import torch
    from fms.models.gpt_bigcode import GPTBigCode

    register_fms_models()
    hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    model = GPTBigCode(
        src_vocab_size=hf_model.config.vocab_size,
        emb_dim=hf_model.config.n_embd,
        ln_eps=hf_model.config.layer_norm_epsilon,
        nheads=hf_model.config.n_head,
        nlayers=hf_model.config.n_layer,
        hidden_grow_factor=hf_model.config.n_inner / hf_model.config.hidden_size,
        pad_id=-1,
        max_pos=hf_model.config.max_position_embeddings,
    )

    new_hf_sd = __rename_weights_to_fms(hf_model.transformer.state_dict())
    model.load_state_dict(new_hf_sd, strict=False)
    with torch.no_grad():
        for i, layer in enumerate(hf_model.transformer.h):
            q, k, v = layer.attn.c_attn.weight.split([2048, 128, 128], dim=0)
            q_bias, k_bias, v_bias = layer.attn.c_attn.bias.split(
                [2048, 128, 128], dim=0
            )
            model.base_model.layers[i].attn.query.weight.copy_(q)
            model.base_model.layers[i].attn.query.bias.copy_(q_bias)
            model.base_model.layers[i].attn.key.weight.copy_(k)
            model.base_model.layers[i].attn.key.bias.copy_(k_bias)
            model.base_model.layers[i].attn.value.weight.copy_(v)
            model.base_model.layers[i].attn.value.bias.copy_(v_bias)
        model.head.weight.copy_(hf_model.lm_head.weight)
    hf_model_fms = HFAdaptedGPTBigCodeForCausalLM.from_fms_model(
        model=model,
        bos_token_id=hf_model.config.bos_token_id,
        eos_token_id=hf_model.config.eos_token_id,
        pad_token_id=hf_model.config.pad_token_id,
    )
    return hf_model_fms


def __rename_weights_to_fms(orig_sd):
    import re

    replacements = [
        (r"^wte.weight", "base_model.embedding.weight"),
        (r"^wpe.weight", "base_model.position_embedding.weight"),
        (r"^ln_f", "base_model.dec_norm"),
        (r"^h", "base_model.layers"),
        # need to do kqv manually
        (r"attn\.c_proj", "attn.dense"),
        (r"mlp\.c_fc", "ff_sub_layer.w1"),
        (r"mlp\.c_proj", "ff_sub_layer.w2"),
        (r"ln_1", "ln"),
        (r"ln_2", "ff_ln"),
    ]
    new_sd = {}
    for name, param in orig_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

    return new_sd
