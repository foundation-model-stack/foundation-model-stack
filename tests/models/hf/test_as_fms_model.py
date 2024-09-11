import math
import tempfile

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    RobertaTokenizerFast,
    pipeline,
)

from fms.models.hf import as_fms_model, to_hf_api
from fms.testing.comparison import HFModelSignatureParams, compare_model_signatures


@pytest.mark.parametrize("model_id_or_path", ["bigcode/gpt_bigcode-santacoder"])
def test_as_fms_model_equivalency_for_decoder(model_id_or_path):
    fms_model = as_fms_model(model_id_or_path, data_type=torch.float32)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path, torch_dtype=torch.float32
    )
    fms_model = to_hf_api(
        fms_model,
        bos_token_id=hf_model.config.bos_token_id,
        pad_token_id=hf_model.config.pad_token_id,
        eos_token_id=hf_model.config.eos_token_id,
    )
    hf_model = hf_model.eval()
    fms_model = fms_model.eval()
    inp = torch.arange(5, 15).unsqueeze(0)
    fms_params = HFModelSignatureParams(model=fms_model, params=["input_ids"], inp=inp)
    hf_params = HFModelSignatureParams(model=hf_model, params=["input_ids"], inp=inp)

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


@pytest.mark.parametrize("model_id_or_path", ["FacebookAI/roberta-base"])
def test_as_fms_model_equivalency_for_encoder(model_id_or_path):
    hf_model = AutoModelForMaskedLM.from_pretrained(model_id_or_path)
    with tempfile.TemporaryDirectory() as workdir:
        # robertas bin file is not working properly, and we are getting different results for safetensors, this should
        # be addressed in another PR
        hf_model.save_pretrained(
            f"{workdir}/roberta-base-masked_lm", safe_serialization=False
        )

        # loading from local rather than snapshot download
        fms_model = as_fms_model(f"{workdir}/roberta-base-masked_lm")
        fms_model = to_hf_api(
            fms_model,
            bos_token_id=hf_model.config.bos_token_id,
            pad_token_id=hf_model.config.pad_token_id,
            eos_token_id=hf_model.config.eos_token_id,
            task_specific_params=hf_model.config.task_specific_params,
        )
    fms_model = fms_model.eval()
    hf_model = hf_model.eval()
    inp = torch.arange(5, 15).unsqueeze(0)
    fms_params = HFModelSignatureParams(model=fms_model, params=["input_ids"], inp=inp)
    hf_params = HFModelSignatureParams(model=hf_model, params=["input_ids"], inp=inp)
    compare_model_signatures(fms_params, hf_params)

    with torch.no_grad():
        tokenizer = RobertaTokenizerFast.from_pretrained(model_id_or_path)
        prompt = "Hello I'm a <mask> model."
        unmasker = pipeline("fill-mask", model=hf_model, tokenizer=tokenizer)
        hf_output = unmasker(prompt)

        unmasker = pipeline("fill-mask", model=fms_model, tokenizer=tokenizer)
        hf_fms_output = unmasker(prompt)

    for res_hf, res_hf_fms in zip(hf_output, hf_fms_output):
        assert math.isclose(res_hf["score"], res_hf_fms["score"], abs_tol=1e-3)
        assert res_hf["sequence"] == res_hf_fms["sequence"]
        assert res_hf["token"] == res_hf_fms["token"]
        assert res_hf["token_str"] == res_hf_fms["token_str"]
