import pytest

from fms.models.gpt_bigcode import GPTBigCode
from fms.models.hf.gpt_bigcode.modeling_gpt_bigcode_hf import GPTBigCodeHFForCausalLM
from fms.testing.comparison import (
    ModelSignatureParams,
    HFModelSignatureParams,
    compare_model_signatures,
)


@pytest.mark.slow
def test_gptbigcode_equivalence():
    """Tests GPT BigCode equivalence with a known implementation. Takes approximately 1:11 on an mbp with M1 chip"""
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained("bigcode/gpt_bigcode-santacoder")
    hf_model = AutoModelForCausalLM.from_pretrained("bigcode/gpt_bigcode-santacoder")
    hf_model.config.scale_attention_softmax_in_fp32 = False

    model = GPTBigCode(
        nheads=16,
        nlayers=24,
        pad_id=-1,
        max_pos=hf_model.config.max_position_embeddings,
    )
    count_parameters = lambda m: sum(p.numel() for p in m.parameters())
    assert count_parameters(model) == count_parameters(hf_model)

    new_hf_sd = rename_weights_to_fms(hf_model.transformer.state_dict())
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
    hf_model_fms = GPTBigCodeHFForCausalLM.from_fms_model(
        model=model,
        bos_token_id=hf_model.config.bos_token_id,
        eos_token_id=hf_model.config.eos_token_id,
        pad_token_id=hf_model.config.pad_token_id,
    )

    assert count_parameters(hf_model_fms) == count_parameters(hf_model)

    model.eval()
    hf_model.eval()
    hf_model_fms.eval()

    inp = torch.arange(0, 16).unsqueeze(0)

    fms_signature_params = ModelSignatureParams(model=model, params=1, inp=inp)

    hf_fms_signature_params = HFModelSignatureParams(
        model=hf_model_fms,
        params=["input_ids", "labels"],
        other_params={"return_dict": True},
        inp=inp,
    )

    hf_signature_params = HFModelSignatureParams(
        model=hf_model,
        params=["input_ids", "labels"],
        other_params={"return_dict": True},
        inp=inp,
    )

    compare_model_signatures(fms_signature_params, hf_fms_signature_params)
    compare_model_signatures(hf_fms_signature_params, hf_signature_params)

    prompt = "def print_hello_world():"

    generator_hf = pipeline(
        task="text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        use_cache=True,
        num_beams=3,
        max_new_tokens=50,
    )
    generator_hf_fms = pipeline(
        task="text-generation",
        model=hf_model_fms,
        tokenizer=tokenizer,
        use_cache=True,
        num_beams=3,
        max_new_tokens=50,
    )
    output_hf = generator_hf(prompt)
    output_hf_fms = generator_hf_fms(prompt)
    assert output_hf == output_hf_fms
    print(output_hf_fms)


def rename_weights_to_fms(orig_sd):
    import re

    replacements = [
        (r"^wte.weight", "base_model.embedding.emb.weight"),
        (r"^wpe.weight", "base_model.embedding.pos_emb.weight"),
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
