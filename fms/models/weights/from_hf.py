from fms.models.llama import LLaMA, LLaMAConfig
from fms.utils.tokenizers import get_tokenizer, _has_hf

def load_weights(hf_model: "LlamaForCausalLM") -> LLaMA:
    """
    Convert a Llama huggingface model to an fms model
    Parameters
    ----------
    hf_model: LlamaForCausalLM
        a Llama Huggingface model
    Returns
    -------
    LLaMA
        an FMS LLaMA model
    """

    if not _has_hf:
        raise ImportError("in order to convert huggingface weights, you need to have transformers installed")
    
    import re

    config = LLaMAConfig(
        src_vocab_size=hf_model.config.vocab_size,
        emb_dim=hf_model.config.hidden_size,
        norm_eps=hf_model.config.rms_norm_eps,
        nheads=hf_model.config.num_attention_heads,
        nlayers=hf_model.config.num_hidden_layers,
        hidden_grow_factor=hf_model.config.intermediate_size / hf_model.config.hidden_size,
        multiple_of=1,  # this is set to 1 as it is encoded in the hidden dimension
        activation_fn=hf_model.config.hidden_act,
        max_expected_seq_len=hf_model.config.max_position_embeddings,
    )
    model = LLaMA(config)
    count_parameters = lambda m: sum(p.numel() for p in m.parameters())
    assert count_parameters(model) == count_parameters(hf_model)

    hf_sd = hf_model.model.state_dict()

    replacements = [
        (r"^embed_tokens.weight", "shared.emb.weight"),
        (r"^norm", "dec_norm"),
        (r"^layers", "layers"),
        (r"self_attn\.k_proj", "attn.key"),
        (r"self_attn\.v_proj", "attn.value"),
        (r"self_attn\.q_proj", "attn.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in hf_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

    model.load_state_dict(new_sd, strict=False)
    model.shared.head.weight = hf_model.lm_head.weight
    model.rot_emb.freqs = hf_model.model.layers[0].self_attn.rotary_emb.inv_freq
    for layer in model.layers:
        q = layer.attn.query.weight.data
        q = q.view(model.config.nheads, 2, -1, q.size(1)).transpose(1, 2).reshape(*q.size())
        layer.attn.query.weight.data = q

        k = layer.attn.key.weight.data
        k = k.view(model.config.nheads, 2, -1, k.size(1)).transpose(1, 2).reshape(*k.size())
        layer.attn.key.weight.data = k

    return model
