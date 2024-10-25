import pytest
import torch

from fms.models import get_model
from fms.models.hf.utils import to_hf_api
from fms.models.llama import convert_hf_llama
from fms.testing.comparison import (
    HFModelSignatureParams,
    ModelSignatureParams,
    compare_model_signatures,
)


@pytest.mark.slow
def test_llama_7b_equivalence():
    """Tests llama equivalence with a known implementation. Takes approximately 8:38 on an mbp with M1 chip"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    # for now, this test won't be run, but it has been verified
    # if you would like to try this, set llama_model_path to the huggingface llama2 model path
    llama_model_path = ""
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

    # Test Parameter Count

    count_parameters = lambda m: sum(p.numel() for p in m.parameters())
    assert count_parameters(hf_model_fms) == count_parameters(hf_model)

    # Test Model Signatures

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

    # Test Generation Pipeline

    prompt = """q: how are you? a: I am good. How about you? q: What is the weather like today? a:"""

    generator_hf = pipeline(
        task="text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        use_cache=True,
        num_beams=3,
        max_new_tokens=20,
    )
    generator_hf_fms = pipeline(
        task="text-generation",
        model=hf_model_fms,
        tokenizer=tokenizer,
        use_cache=True,
        num_beams=3,
        max_new_tokens=20,
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
        math.isclose(hf_model_loss.item(), hf_model_fms_loss.item(), abs_tol=1e-3),
        f"model loss is not equal",
    )
