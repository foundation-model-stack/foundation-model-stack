import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from fms.models.hf import as_fms_model, to_hf_api
from fms.testing.comparison import HFModelSignatureParams, compare_model_signatures


@pytest.mark.parametrize("model_id_or_path", ["bigcode/gpt_bigcode-santacoder"])
def test_get_model_equivalency(model_id_or_path):
    fms_model = as_fms_model(model_id_or_path)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id_or_path)
    fms_model = to_hf_api(
        fms_model,
        bos_token_id=hf_model.config.bos_token_id,
        pad_token_id=hf_model.config.pad_token_id,
        eos_token_id=hf_model.config.eos_token_id,
    )

    fms_params = HFModelSignatureParams(model=fms_model, params=["input_ids"])
    hf_params = HFModelSignatureParams(model=hf_model, params=["input_ids"])

    compare_model_signatures(fms_params, hf_params)

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

    fms_gen = pipeline(task="text-generation", model=fms_model, tokenizer=tokenizer)
    hf_gen = pipeline(task="text-generation", model=hf_model, tokenizer=tokenizer)
    prompt = "q: hello how are you? a: I am good. q: What sports do you like? a: I like baseball. q: What is the weather like today? a:"

    output_fms = fms_gen(prompt, do_sample=False, max_new_tokens=100)[0][
        "generated_text"
    ]
    output_hf = hf_gen(prompt, do_sample=False, max_new_tokens=100)[0]["generated_text"]
    assert output_fms == output_hf
