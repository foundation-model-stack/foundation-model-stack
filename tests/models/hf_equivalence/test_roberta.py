import math
import tempfile

import pytest
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    RobertaTokenizerFast,
)

from fms.models import get_model
from fms.models.hf import to_hf_api
from fms.models.hf.roberta.modeling_roberta_hf import (
    HFAdaptedRoBERTaForSequenceClassification,
)
from fms.testing.comparison import (
    HFModelSignatureParams,
    ModelSignatureParams,
    compare_model_signatures,
)


def test_roberta_base_for_masked_lm_equivalency():
    # create models
    hf_model = AutoModelForMaskedLM.from_pretrained("roberta-base", device_map="cpu")

    with tempfile.TemporaryDirectory() as workdir:
        hf_model.save_pretrained(
            f"{workdir}/roberta-base-masked_lm", safe_serialization=False
        )

        model = get_model(
            "roberta",
            "base",
            f"{workdir}/roberta-base-masked_lm",
            "hf",
            norm_eps=1e-5,
            tie_heads=True,
            device_type="cpu",
        )

    # test the param count is the same before we load hf fms model
    model_param_count = sum([p.numel() for p in model.parameters()])
    # note: we subtract 3*768 for the following reasons:
    #     1x768 - our model does not have a token_type_embeddings as part of its embeddings
    #     2x768 - our model uses 512 positional encodings instead of 514 (they don't use their position 0, and their position 1 is the zeros vector)
    hf_model_param_count = sum([p.numel() for p in hf_model.parameters()]) - 3 * 768
    assert model_param_count == hf_model_param_count

    hf_model_fms = to_hf_api(
        model, task_specific_params=hf_model.config.task_specific_params
    )

    # test the param count is the same between hf model and hf fms model
    hf_model_fms_param_count = sum([p.numel() for p in hf_model_fms.parameters()])
    assert hf_model_param_count == hf_model_fms_param_count

    model.eval()
    hf_model.eval()
    hf_model_fms.eval()

    inp = torch.arange(5, 15).unsqueeze(0)
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
        other_params={
            "return_dict": True,
            # "attention_mask": mask_2d_to_3d(inp),
        },
        inp=inp,
    )

    compare_model_signatures(fms_signature_params, hf_fms_signature_params)
    compare_model_signatures(hf_fms_signature_params, hf_signature_params)

    from transformers import pipeline

    with torch.no_grad():
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        prompt = "Hello I'm a <mask> model."
        unmasker = pipeline("fill-mask", model=hf_model, tokenizer=tokenizer)
        hf_output = unmasker(prompt)

        unmasker = pipeline(
            "fill-mask", model=hf_model_fms, tokenizer=tokenizer, device="cpu"
        )
        hf_fms_output = unmasker(prompt)

    for res_hf, res_hf_fms in zip(hf_output, hf_fms_output):
        assert math.isclose(res_hf["score"], res_hf_fms["score"], abs_tol=1e-3)
        assert res_hf["sequence"] == res_hf_fms["sequence"]
        assert res_hf["token"] == res_hf_fms["token"]
        assert res_hf["token_str"] == res_hf_fms["token_str"]

    # test loss
    inputs = torch.arange(0, 15).unsqueeze(0)
    labels = torch.arange(0, 15).unsqueeze(0)

    attention_mask = (inputs == 1).unsqueeze(-1) == (inputs == 1).unsqueeze(-2)
    hf_model_loss = hf_model(
        input_ids=inputs, labels=labels, attention_mask=attention_mask, return_dict=True
    ).loss
    hf_model_fms_loss = hf_model_fms(
        input_ids=inputs, labels=labels, attention_mask=attention_mask, return_dict=True
    ).loss

    torch._assert(
        math.isclose(hf_model_loss.item(), hf_model_fms_loss.item(), abs_tol=1e-3),
        "model loss is not equal",
    )


sequence_classification_params = [
    ("sentiment-analysis", "multi_label_classification"),
    ("text-classification", "single_label_classification"),
    ("text-classification", "regression"),
]


@pytest.mark.parametrize(
    "task,problem_type",
    sequence_classification_params,
    ids=[x[1] for x in sequence_classification_params],
)
def test_roberta_base_for_sequence_classification(task, problem_type):
    # create models
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        "SamLowe/roberta-base-go_emotions",
        device_map="cpu",
    )
    hf_model.config.problem_type = problem_type
    with torch.no_grad():
        hf_model.roberta.embeddings.token_type_embeddings.weight.zero_()

    with tempfile.TemporaryDirectory() as workdir:
        hf_model.save_pretrained(
            f"{workdir}/SamLowe-roberta-base-go_emotions", safe_serialization=False
        )

        model = get_model(
            "roberta",
            "base",
            f"{workdir}/SamLowe-roberta-base-go_emotions",
            "hf",
            norm_eps=1e-5,
            tie_heads=True,
            device_type="cpu",
        )

    # copy weights
    hf_model_fms = HFAdaptedRoBERTaForSequenceClassification.from_fms_model(
        model,
        id2label=hf_model.config.id2label,
        label2id=hf_model.config.label2id,
        num_labels=hf_model.config.num_labels,
        eos_token_id=hf_model.config.eos_token_id,
        bos_token_id=hf_model.config.bos_token_id,
        pad_token_id=hf_model.config.pad_token_id,
        problem_type=hf_model.config.problem_type,
    )
    hf_model_fms.lm_head.dense.weight = hf_model.classifier.dense.weight
    hf_model_fms.lm_head.dense.bias = hf_model.classifier.dense.bias
    hf_model_fms.lm_head.head.weight = hf_model.classifier.out_proj.weight
    hf_model_fms.lm_head.head.bias = hf_model.classifier.out_proj.bias

    model.eval()
    hf_model.eval()
    hf_model_fms.eval()

    from transformers import pipeline

    with torch.no_grad():
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        prompt = "Hugging Face is the best thing since sliced bread!"
        classifier = pipeline(task=task, model=hf_model, tokenizer=tokenizer)
        hf_output = classifier(prompt)
        print(hf_output)

        classifier = pipeline(
            task=task, model=hf_model_fms, tokenizer=tokenizer, device="cpu"
        )
        hf_fms_output = classifier(prompt)
        print(hf_fms_output)

    for res_hf, res_hf_fms in zip(hf_output, hf_fms_output):
        assert math.isclose(res_hf["score"], res_hf_fms["score"], abs_tol=1e-3)
        assert res_hf["label"] == res_hf_fms["label"]

    # test loss
    inputs = torch.arange(0, 16).unsqueeze(0)
    if problem_type == "single_label_classification":
        labels = torch.randint(high=hf_model.config.num_labels, size=(1,))
    else:
        labels = torch.randn(hf_model.config.num_labels).unsqueeze(0)
    attention_mask = (inputs == 1).unsqueeze(-1) == (inputs == 1).unsqueeze(-2)
    hf_model_loss = hf_model(
        input_ids=inputs, labels=labels, attention_mask=attention_mask, return_dict=True
    ).loss
    hf_model_fms_loss = hf_model_fms(
        input_ids=inputs, labels=labels, attention_mask=attention_mask, return_dict=True
    ).loss
    print(hf_model_loss)
    print(hf_model_fms_loss)
    torch._assert(
        math.isclose(hf_model_loss.item(), hf_model_fms_loss.item(), abs_tol=1e-3),
        "model loss is not equal",
    )
