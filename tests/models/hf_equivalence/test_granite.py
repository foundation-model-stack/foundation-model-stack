import pytest
import torch

from fms.models import get_model
from fms.models.hf import to_hf_api
from fms.testing.comparison import (
    HFModelSignatureParams,
    ModelSignatureParams,
    compare_model_signatures,
)


@pytest.mark.slow
def test_granite_8b_equivalence():
    """Tests granite equivalence with a known implementation. Takes approximately 8:38 on an mbp with M1 chip"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    # for now, this test won't be run, but it has been verified
    # if you would like to try this, set granite_model_path to the huggingface granite model path
    granite_model_path = "ibm-granite/granite-3.1-8b-base"
    tokenizer = AutoTokenizer.from_pretrained(granite_model_path, use_fast=True)
    hf_model = AutoModelForCausalLM.from_pretrained(granite_model_path)

    # convert the hf model to fms
    model = get_model("hf_pretrained", granite_model_path)

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

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters())

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
        max_new_tokens=20,
    )
    generator_hf_fms = pipeline(
        task="text-generation",
        model=hf_model_fms,
        tokenizer=tokenizer,
        use_cache=True,
        max_new_tokens=20,
    )
    output_hf = generator_hf(prompt)
    output_hf_fms = generator_hf_fms(prompt)
    assert output_hf == output_hf_fms
