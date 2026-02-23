"""
Tests in this module are for sanity checking the mapping
between HF model architectures and FMS model kwargs.

NOTE: This is technically a little different than directly
checking the model config since it doesn't include the defaults.
"""

import json
import os
import pytest
from fms.utils.config import ModelConfig

# TODO: centralize access to resources w/ the testing module
FMS_CONFIGS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "resources", "fms_model_kwargs"
)

# Models to be validated; we should have (at least) one model of each architecture
# that is supported here. To get the current kwargs for a new model, add it to this
# dictionary, and run this file as `main` to export the corresponding file to the
# resources subdir.
MODEL_CONFIG_MAP = {
    "LlamaForCausalLM": "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
    "GPTBigCodeForCausalLM": "bigcode/gpt_bigcode-santacoder",
    "MixtralForCausalLM": "mistralai/Mixtral-8x7B-v0.1",
    "RobertaForMaskedLM": "FacebookAI/roberta-base",
    "RobertaForQuestionAnswering": "deepset/roberta-base-squad2",
    "RobertaForSequenceClassification": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "GraniteForCausalLM": "ibm-granite/granite-3.3-2b-instruct",
    "MistralForCausalLM": "mistralai/Mistral-Small-24B-Base-2501",
    "Mistral3ForConditionalGeneration": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "BambaForCausalLM": "ibm-ai-platform/Bamba-9B-v2",
    "SiglipModel": "google/siglip-so400m-patch14-384",
    "LlavaNextForConditionalGeneration": "ibm-granite/granite-vision-3.2-2b",
    "MPNetForMaskedLM": "microsoft/mpnet-base",
    "BertForMaskedLM": "google-bert/bert-base-uncased",
    "BertForSequenceClassification": "nlptown/bert-base-multilingual-uncased-sentiment",
}


def get_file_kwargs_filename(model_name: str) -> str:
    """Get the name json file containing the FMS model kwargs."""
    return f"{model_name.split('/')[-1]}.json"


@pytest.mark.parametrize("arch,model_name", MODEL_CONFIG_MAP.items())
def test_config_mapping(arch: str, model_name: str):
    from fms.models.hf.utils import infer_model_configuration

    cfg_filename = get_file_kwargs_filename(model_name)
    kwargs_path = os.path.join(FMS_CONFIGS_DIR, cfg_filename)
    if not os.path.isfile(kwargs_path):
        raise FileNotFoundError(
            f"Model {model_name} has no kwargs file; have you generated it?"
        )
    with open(kwargs_path, "r") as f:
        expected_kwargs = json.load(f)
    actual_kwargs = infer_model_configuration(model_name, download_weights=False)

    for k, actual_v in actual_kwargs.items():
        expected_v = expected_kwargs[k]
        # Convert encapsulated model configs to dicts for comparison, since
        # the ground truth are reloaded from raw JSON files, which convert to
        # dicts at export time.
        if isinstance(actual_v, ModelConfig):
            actual_v = actual_v.as_dict()
        assert expected_v == actual_v


### Helper utils for validating your local configs & exporting them.
def export_config_mappings():
    """Pull the configs for all models in the model config map,
    and dump ."""
    from transformers import AutoConfig
    from fms.models.hf.utils import infer_model_configuration

    for arch, model_name in MODEL_CONFIG_MAP.items():
        # Export the HF / FMS configs if they don't exist already
        cfg_filename = get_file_kwargs_filename(model_name)
        fms_kwargs_path = os.path.join(FMS_CONFIGS_DIR, cfg_filename)
        if os.path.isfile(fms_kwargs_path):
            print(
                f"FMS model kwargs for {model_name} were already exported - skipping..."
            )
            continue

        hf_config = AutoConfig.from_pretrained(model_name)
        model_arch = hf_config.architectures[0]
        if model_arch != arch:
            raise ValueError(
                f"Arch {arch} does not match value {model_arch} inferred from the model config!"
            )

        fms_config_kwargs = infer_model_configuration(
            model_name, download_weights=False
        )

        if not os.path.isfile(fms_kwargs_path):
            with open(fms_kwargs_path, "w") as f_path:
                # Fall back to trying to convert from dataclass, which
                # may happen in composite models, e.g., multimodal, since
                # for now we're just dumping the kwargs to the model, and
                # not the model configs themselves
                json.dump(
                    fms_config_kwargs,
                    f_path,
                    indent=4,
                    sort_keys=True,
                    default=lambda obj: obj.as_dict(),
                )
            print(f"FMS model kwargs for {model_name} were exported!")


if __name__ == "__main__":
    export_config_mappings()
