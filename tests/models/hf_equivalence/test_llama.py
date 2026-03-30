import pytest
import torch

from fms.models import get_model
from fms.models.hf import to_hf_api


@pytest.mark.slow
def test_llama_3b_equivalence():
    """Tests llama equivalence with a known implementation. Takes approximately 8:38 on an mbp with M1 chip"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    # for now, this test won't be run, but it has been verified
    llama_model_path = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=True)
    hf_model = AutoModelForCausalLM.from_pretrained(llama_model_path)

    # convert the hf model to fms
    model = get_model("hf_pretrained", llama_model_path)

    hf_model_fms = to_hf_api(
        model,
        bos_token_id=hf_model.config.bos_token_id,
        eos_token_id=hf_model.config.eos_token_id,
        pad_token_id=hf_model.config.pad_token_id,
    )

    model.eval()
    hf_model.eval()
    hf_model_fms.eval()

    # Keeping signature tests only at tests/models/hf/test_llama_hf.py
    # Testing model generation equivalency for meta-llama/Llama-3.2-3B

    prompt = """q: how are you? a: I am good. How about you? q: What is the weather like today? a:"""

    generator_hf = pipeline(
        task="text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        use_cache=True,
        num_beams=3,
        max_new_tokens=20,
        do_sample=False,
    )
    generator_hf_fms = pipeline(
        task="text-generation",
        model=hf_model_fms,
        tokenizer=tokenizer,
        use_cache=True,
        num_beams=3,
        max_new_tokens=20,
        do_sample=False,
    )
    output_hf = generator_hf(prompt)
    output_hf_fms = generator_hf_fms(prompt)
    assert output_hf == output_hf_fms

    # Test Train Loss

    inputs = torch.arange(0, 16).unsqueeze(0)
    labels = torch.arange(0, 16).unsqueeze(0)
    hf_model_loss = hf_model(input_ids=inputs, labels=labels, return_dict=True).loss
    hf_model_fms_loss = hf_model_fms(
        input_ids=inputs, labels=labels, return_dict=True
    ).loss

    import math

    torch._assert(
        math.isclose(hf_model_loss.item(), hf_model_fms_loss.item(), abs_tol=1e-2),
        "model loss is not equal",
    )
