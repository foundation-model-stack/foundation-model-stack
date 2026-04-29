import pytest
import torch

from fms.models import get_model
from fms.models.hf import to_hf_api

from packaging.version import Version
from transformers import __version__ as tf_version


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

    use_cache = False

    if Version(tf_version) >= Version("5.0.0"):
        use_cache = True
    else:
        # for versions > 4.57.x and < 5.0.0, use_cache is disabled;
        # this way we are retro compatible with parameter called cache_position
        # https://huggingface.co/docs/transformers/cache_explanation#cache-position
        use_cache = False

    generator_hf = pipeline(
        task="text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        use_cache=use_cache,
        num_beams=1,  # Use greedy decoding for deterministic results
        max_new_tokens=20,
        do_sample=False,
    )
    generator_hf_fms = pipeline(
        task="text-generation",
        model=hf_model_fms,
        tokenizer=tokenizer,
        use_cache=use_cache,
        num_beams=1,  # Use greedy decoding for deterministic results
        max_new_tokens=20,
        do_sample=False,
    )
    output_hf = generator_hf(prompt)
    output_hf_fms = generator_hf_fms(prompt)

    # Compare generated text with helpful error message
    assert output_hf[0]["generated_text"] == output_hf_fms[0]["generated_text"], (
        f"Generated text mismatch:\n"
        f"HF: {output_hf[0]['generated_text']}\n"
        f"FMS: {output_hf_fms[0]['generated_text']}"
    )

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
