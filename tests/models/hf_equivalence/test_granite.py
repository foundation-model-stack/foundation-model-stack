import pytest

from fms.models import get_model
from fms.models.hf import to_hf_api


from packaging.version import Version
from transformers import __version__ as tf_version


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

    # Keeping signature tests only at tests/models/hf/test_granite_hf.py
    # Testing model generation equivalency for Granite 8B

    prompt = """q: how are you? a: I am good. How about you? q: What is the weather like today? a:"""

    generator_hf = pipeline(
        task="text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        use_cache=True,
        max_new_tokens=20,
        do_sample=False,
    )
    generator_hf_fms = pipeline(
        task="text-generation",
        model=hf_model_fms,
        tokenizer=tokenizer,
        use_cache=True,
        max_new_tokens=20,
        do_sample=False,
    )
    output_hf = generator_hf(prompt)
    output_hf_fms = generator_hf_fms(prompt)

    # Compare generated text
    if Version(tf_version) >= Version("5.0.0"):
        assert output_hf[0]["generated_text"] == output_hf_fms[0]["generated_text"], (
            f"Generated text mismatch:\n"
            f"HF: {output_hf[0]['generated_text']}\n"
            f"FMS: {output_hf_fms[0]['generated_text']}"
        )
    else:
        assert output_hf == output_hf_fms, (
            f"Generated text mismatch:\nHF: {output_hf}\nFMS: {output_hf_fms}"
        )
