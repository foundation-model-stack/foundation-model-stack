import os
from typing import Union

import torch
from transformers import PreTrainedModel, GPTBigCodeForCausalLM, GPTBigCodeConfig

from fms.models.hf.gpt_bigcode.modeling_gpt_bigcode_hf import (
    HFAdaptedGPTBigCodeForCausalLM,
)


def get_model(
    model_name_or_path: Union[str, os.PathLike, PreTrainedModel],
) -> HFAdaptedGPTBigCodeForCausalLM:
    """
    Get a Huggingface adapted FMS model from an equivalent HF model

    Parameters
    ----------
    model_name_or_path: Union[str, os.PathLike, PreTrainedModel]
        Either the name of the model in huggingface hub, the absolute path to
        the huggingface model, or the huggingface model itself. If the
        model_name_or_path is a PreTrainedModel, it will not be deleted from
        memory

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
    hf_model_in_memory = isinstance(model_name_or_path, PreTrainedModel)
    if hf_model_in_memory:
        hf_model = model_name_or_path
    else:
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
            # if the model was not provided to us, we can assume we don't want
            # the layers to persist in memory
            if not hf_model_in_memory:
                del layer
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
            n_positions=hf_config.max_pos,
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
