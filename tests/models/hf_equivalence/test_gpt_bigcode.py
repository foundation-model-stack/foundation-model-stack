import pytest

from fms.models.gpt_bigcode import GPTBigCode
from fms.models.hf.gpt_bigcode import get_model
from fms.models.hf.gpt_bigcode.modeling_gpt_bigcode_hf import (
    HFAdaptedGPTBigCodeForCausalLM,
)
from fms.testing.comparison import (
    ModelSignatureParams,
    HFModelSignatureParams,
    compare_model_signatures,
)


@pytest.mark.slow
def test_gptbigcode_equivalence():
    """Tests GPT BigCode equivalence with a known implementation. Takes approximately 1:11 on an mbp with M1 chip"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained("bigcode/gpt_bigcode-santacoder")
    hf_model = AutoModelForCausalLM.from_pretrained("bigcode/gpt_bigcode-santacoder")
    hf_model.config.scale_attention_softmax_in_fp32 = False

    hf_model_fms = get_model(hf_model)
    count_parameters = lambda m: sum(p.numel() for p in m.parameters())
    assert count_parameters(hf_model_fms) == count_parameters(hf_model)

    hf_model.eval()
    hf_model_fms.eval()

    inp = torch.arange(0, 16).unsqueeze(0)

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
