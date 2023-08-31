import argparse
from typing import Type, Union

import torch
from transformers import AutoTokenizer

from fm.utils import get_signature

from fms.models.llama import LLaMA, LLaMAConfig

parser = argparse.ArgumentParser(description="generate small model tests")

parser.add_argument("--generate_config", action="store_true", help="create a new config from the model params")
parser.add_argument("--generate_weights", action="store_true", help="save the model state with weights reset")
parser.add_argument(
    "--generate_signature", action="store_true", help="generate a signature for this model and save the signature"
)
parser.add_argument("--generate_tokenizer", action="store_true", help="save the tokenizer")
parser.add_argument(
    "--model",
    default="",
    type=str,
    help="If not provided, will generate for all models, otherwise provide a comma separated list of model names",
)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
_models_path = "../../tests/resources/models"

models = set(args.model.split(","))


def save(
    model_test_name: str,
    model_class: Type[Union[LLaMA]],
    config_class: Type[Union[LLaMAConfig]],
    model_params: dict,
    num_params: int,
):
    """save the model config/weights/signature/tokenizer for a given test case"""
    model_test_path = f"{_models_path}/{model_test_name}"

    # if generate config is true, create a new config from the model params
    # if generate config is false, simply load the config that already exists
    if args.generate_config:
        print(f"saving config for {model_test_name}")
        config = config_class(**model_params)
        config.save(f"{model_test_path}/config.json")
    else:
        config = config_class.load(f"{model_test_path}/config.json")

    # create a model from a config
    model = model_class(config)

    # if generate weights is true, save the model state with weights reset
    # if generate weights is false, load the model state that already exists
    if args.generate_weights:
        print(f"saving weights for {model_test_name}")
        torch.save(model.state_dict(), f"{model_test_path}/model_state.pth")
    else:
        model.load_state_dict(torch.load(f"{model_test_path}/model_state.pth"))

    # if generate signature is true, generate a signature for this model and save the signature
    if args.generate_signature:
        print(f"saving signature for {model_test_name}")
        model.eval()
        signature = get_signature(model, num_params)
        torch.save(signature, f"{model_test_path}/signature.pth")

    # if generate tokenizer is true, save the tokenizer
    if args.generate_tokenizer:
        print(f"saving tokenizer for {model_test_name}")
        tokenizer.save_pretrained(f"{model_test_path}/tokenizer")


test_to_generate = []

############### LLAMA ###################

llama_7b_params = {
    "src_vocab_size": tokenizer.vocab_size,
    "emb_dim": 16,
    "multiple_of": 2,
    "nheads": 2,
    "nlayers": 2,
    "norm_eps": 1e-05,
    "pad_id": 0,
}

if "llama" in models or len(models) == 0:
    test_to_generate.append(["llama/mock.7b", LLaMA, LLaMAConfig, llama_7b_params, 1])

############### GENERATE TESTS ###################

for test in test_to_generate:
    save(*test)
