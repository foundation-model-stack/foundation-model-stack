from fms.models import get_model
from fms.models.hf import to_hf_api

import pytest


def _predict_text(model, tokenizer, texts, use_cache, num_beams):
    encoding = tokenizer(texts, return_tensors="pt")

    # Fix for newer versions of transformers
    use_cache_kwarg = {}
    if use_cache is not None:
        use_cache_kwarg["use_cache"] = use_cache

    encoding = encoding.to("cpu")
    model.eval()
    import torch

    with torch.no_grad():
        generated_ids = model.generate(
            **encoding,
            num_beams=num_beams,
            max_new_tokens=6,
            repetition_penalty=2.5,
            top_k=50,
            do_sample=False,
            temperature=0.0,
            **use_cache_kwarg,
        )
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts


@pytest.mark.slow
def test_gpt_oss_20b_equivalence():

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from difflib import SequenceMatcher

    gpt_oss = get_model("hf_pretrained", "openai/gpt-oss-20b", device_type="cpu")
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    gpt_oss.eval()  # put the model in evaluation mode

    gpt_oss_hf = to_hf_api(gpt_oss)

    text_options = [
        ["hello how are you?"],
        ["hello how are you?", "a: this is a test. b: this is another test. a:"],
    ]
    out_fms = _predict_text(gpt_oss_hf, tokenizer, text_options[0], True, 1)

    hf_model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")

    out_oss = _predict_text(hf_model, tokenizer, text_options[0], True, 1)

    print(f"{out_oss=}")
    print(f"{out_fms=}")

    # Calculate similarity ratio
    ratio = SequenceMatcher(None, out_fms[0], out_oss[0]).ratio()

    assert ratio > 0.95
