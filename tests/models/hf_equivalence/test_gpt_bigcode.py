import tempfile

import pytest

from fms.models import get_model
from fms.models.hf import to_hf_api

from packaging.version import Version
from transformers import __version__ as tf_version


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
        pad_token_id=getattr(hf_model.config, "pad_token_id", None),
    )

    # Set both models to eval mode
    hf_model.eval()
    hf_model_fms.eval()

    # Keep signatures test only at tests/model/test_gpt_bigcode.py
    # Testing the models' generation

    prompt = "def print_hello_world():"

    use_cache = False

    if Version(tf_version) >= Version("5.0.0"):
        use_cache = True
    else:
        # for versions > 4.57.x and < 5.0.0, use_cache is disabled;
        # this way we are retro compatible with parameter called cache_position
        # https://huggingface.co/docs/transformers/cache_explanation#cache-position
        use_cache = False

    # Use greedy decoding (num_beams=1) for deterministic generation
    # Note: use_cache=False to avoid KV cache shape mismatch issues
    generator_hf = pipeline(
        task="text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        use_cache=use_cache,
        num_beams=1,
        do_sample=False,
        max_new_tokens=50,
    )
    generator_hf_fms = pipeline(
        task="text-generation",
        model=hf_model_fms,
        tokenizer=tokenizer,
        use_cache=use_cache,
        num_beams=1,
        do_sample=False,
        max_new_tokens=50,
    )
    output_hf = generator_hf(prompt)
    output_hf_fms = generator_hf_fms(prompt)

    # Compare generated text with helpful error message
    assert output_hf[0]["generated_text"] == output_hf_fms[0]["generated_text"], (
        f"Generated text mismatch:\n"
        f"HF: {output_hf[0]['generated_text']}\n"
        f"FMS: {output_hf_fms[0]['generated_text']}"
    )
    print(output_hf_fms)
