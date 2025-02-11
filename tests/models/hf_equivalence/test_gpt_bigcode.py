import tempfile

import pytest

from fms.models import get_model
from fms.models.hf import to_hf_api
from fms.testing.comparison import (
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

    with tempfile.TemporaryDirectory() as workdir:
        hf_model.save_pretrained(
            f"{workdir}/gpt_bigcode-santacoder", safe_serialization=True
        )

        fms_model = get_model(
            "gpt_bigcode",
            "santacoder",
            f"{workdir}/gpt_bigcode-santacoder",
            "hf",
        )

    hf_model_fms = to_hf_api(
        fms_model,
        bos_token_id=hf_model.config.bos_token_id,
        eos_token_id=hf_model.config.eos_token_id,
        pad_token_id=hf_model.config.pad_token_id,
    )

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters())

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
