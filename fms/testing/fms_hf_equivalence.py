import argparse
import json
import logging
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fms.models import get_model
from fms.models.hf.utils import to_hf_api


_device = "cuda"
_device_for_diff = "cpu"

dtype_mapping = {"float16": torch.float16, "float32": torch.float32}

batch_size = 8
max_length = 2048

"""
This utility is intended to test end to end comparison between fms implementation and hf implemetation for a model. 
Though the basic test for each model implementation exists that tests the equivalence with smaller data, 
an end to end test with varying length sequences with a larger dataset can help verify how much fms implementation differs from 
hf implementation in the terms of percentage sequences and percentage tokens. The difference may not necessirly stem from model
implementation but it may result from some differences in the implementation of basic fms components, and hence differnces may be appear
at different scale based on model architecture. This utility may help identify some issues in the base components if the differences
are beyond acceptable range for a model implementation.

The token generation by the two models can be executed in parallel, if more than two gpus are available and finally, the utility can be used
to compute the diff between the two outputs.
"""


def generate_and_write_tokens(
    model, tokenizer, input_json_path, output_dir, model_impl_type
):
    """
    This function loads the prompts, generates the tokens for the given prompts using the supplied model and tokenizer.
    Finally, writes the generated tokens in a file "output_dir/model_impl_type/tokens" The output files contains serialized
    ordered dict that contains two entries for each prompt, one for the input tokens and one for the generated tokens.
    Paddings are removed before writing the tokens so it's safe to compare the outputs of two different runs with different batch sizes.

     Parameters
    ----------
    model: Either of the HF or HF adapted FMS model
    tokenizer: The tokenizer associated with the model
    input_json_path: File path to the prompt file. The prompt file is expected to contain a set of JSON records with one attribute - "prompt".
    output_dir: Path to the output directory
    model_impl_type: One of the "fms" or "hf"
    """
    prompts = load_prompts(input_json_path)
    logging.info(f"model = {model}")
    logging.info(f"model config = {model.config}")
    logging.info(f"Num prompts = {len(prompts)}")

    tokenizer.pad_token = tokenizer.eos_token
    model_pad_token_id = None
    if hasattr(model.config, "pad_token_id") and model.config.pad_token_id != None:
        model_pad_token_id = model.config.pad_token_id
    else:
        model_pad_token_id = model.config.eos_token_id

    logging.debug(f"tokenizer_pad_token_id = {tokenizer.pad_token_id}")
    logging.debug(f"model_pad_token_id = {model_pad_token_id}")

    i = 0
    global batch_size
    data = OrderedDict()

    while len(prompts) > 0:
        if len(prompts) >= batch_size:
            batch = prompts[:batch_size]
            prompts = prompts[batch_size:]
        else:
            batch = prompts
            prompts = []

        model_inputs = tokenizer(batch, return_tensors="pt", padding=True).to(
            device=_device
        )
        output = model.generate(**model_inputs, max_length=max_length)

        input_ids = model_inputs.input_ids
        _, input_seq_len = input_ids.size()
        _, output_seq_len = output.size()

        generated_tokens = output[:, input_seq_len:]
        logging.debug(
            f"input_seq_len={input_seq_len}, output_seq_len={output_seq_len}, generated_tokens_size={generated_tokens.size()}"
        )

        for j in range(len(batch)):
            logging.debug(f"input_ids = {input_ids[j]}")
            logging.debug(f"generated_tokens = {generated_tokens[j]}")

            input_ids_padding_removed = remove_paddings(
                input_ids[j], tokenizer.pad_token_id
            )
            generated_tokens_padding_removed = remove_paddings(
                generated_tokens[j], model_pad_token_id
            )

            logging.debug(f"input_ids_padding_removed = {input_ids_padding_removed}")
            logging.debug(
                f"generated_tokens_padding_removed = {generated_tokens_padding_removed}"
            )

            record_num = i * batch_size + j
            data[f"input - {record_num}"] = input_ids_padding_removed
            data[f"output - {record_num}"] = generated_tokens_padding_removed

        i = i + 1
        logging.info(f"Num batches processed = {i}")

    tokens_output_file = tokens_file_path(output_dir, model_impl_type)
    os.makedirs(os.path.dirname(tokens_output_file), exist_ok=True)
    torch.save(data, tokens_output_file)


def load_hf_model(model_path):
    logging.info(f"Loading hf Model")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path
    )
    model.to(_device)
    logging.info(f"Model loaded")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path, padding_side="left"
    )
    return model, tokenizer


def load_fms_model(model_path, model_arch, model_variant):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model_config_overrides = {"max_expected_seq_len": max_length}

    # TODO hack for now, as llama fms impl sets it to -1, and that causes a crash
    if model_arch == "llama":
        model_config_overrides["pad_token_id"] = 0

    logging.info(f"Loading fms Model")
    model = get_model(
        architecture=model_arch,
        variant=model_variant,
        model_path=model_path,
        source="hf",
        device_type=_device,
        **model_config_overrides,
    )
    model = to_hf_api(model=model, **model_config_overrides)
    logging.info(f"Model loaded")
    return model, tokenizer


def load_prompts(input_json_path: str) -> List[str]:
    prompts: List[str] = []
    with open(input_json_path, "r") as input_file:
        input_data = json.load(input_file)
        for e in input_data:
            prompts.append(e["prompt"])
    return prompts


def remove_paddings(v, pad_token_id):
    mask = v != pad_token_id
    return v[mask]


def tokens_file_path(dir, model_impl_type):
    return os.path.join(dir, model_impl_type, "tokens")


@dataclass
class DiffSummary:
    extra_tokens: int = 0
    less_tokens: int = 0
    different_tokens: int = 0
    total_tokens: int = 0
    total_sequences: int = 0
    sequences_with_different_tokens: List[int] = field(default_factory=list)
    sequences_with_extra_tokens: List[int] = field(default_factory=list)
    sequences_with_less_tokens: List[int] = field(default_factory=list)

    def toJson(self):
        return {
            "tokens_stats": {
                "total_tokens": self.total_tokens,
                "extra_tokens": f"{(self.extra_tokens * 100/self.total_tokens):.2f}%",
                "less_tokens": f"{(self.less_tokens * 100/self.total_tokens):.2f}%",
                "different_tokens": f"{(self.different_tokens * 100/self.total_tokens):.2f}%",
            },
            "sequence_stats": {
                "total_sequences": self.total_sequences,
                "sequences_with_extra_tokens": f"{(len(self.sequences_with_extra_tokens) * 100 / self.total_sequences):.2f}%",
                "sequences_with_less_tokens": f"{(len(self.sequences_with_less_tokens) * 100 / self.total_sequences):.2f}%",
                "sequences_with_different_tokens": f"{(len(self.sequences_with_different_tokens) * 100 / self.total_sequences):.2f}%",
            },
            "sequence_info": {
                "sequences_with_extra_tokens": self.sequences_with_extra_tokens,
                "sequences_with_less_tokens": self.sequences_with_less_tokens,
                "sequences_with_different_tokens": self.sequences_with_different_tokens,
            },
        }


def compute_diff(generated_tokens_dir):
    hf_file = tokens_file_path(generated_tokens_dir, "hf")
    fms_file = tokens_file_path(generated_tokens_dir, "fms")
    _compute_diff(hf_file, fms_file)


def _compute_diff(file1, file2):
    """
    Computes the difference in the tokens stored in two files, produced by the function 'generate_and_write_tokens' and prints the summary.
    """
    dict1 = torch.load(file1, map_location=torch.device(_device_for_diff))
    dict2 = torch.load(file2, map_location=torch.device(_device_for_diff))

    if len(dict1) != len(dict2):
        logging.warn(f"The number of entries differ")
        return

    summary = DiffSummary()

    for k1, v1 in dict1.items():
        v2 = dict2[k1]

        if k1.startswith("input"):
            assert torch.equal(v1, v2)
            continue

        prompt_number = k1.split("-", 1)[-1].strip()
        summary.total_sequences += 1
        v1_size = v1.size()[0]
        v2_size = v2.size()[0]
        size_diff = v1_size - v2_size

        if size_diff > 0:
            summary.less_tokens += size_diff
            summary.sequences_with_less_tokens.append(prompt_number)
            v1 = v1[:v2_size]

        elif size_diff < 0:
            summary.extra_tokens += -size_diff
            summary.sequences_with_extra_tokens.append(prompt_number)
            v2 = v2[:v1_size]

        summary.total_tokens += v1_size

        diff = torch.sub(v1, v2)
        num_diff = torch.count_nonzero(diff).item()
        summary.different_tokens += num_diff
        if num_diff > 0:
            summary.sequences_with_different_tokens.append(prompt_number)

    logging.info(f"diff summary = {json.dumps(summary.toJson(), indent=2)}")
    return summary


def main(argv):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Utility functions for comparing outputs of two implementations of a model"
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    fms = subparsers.add_parser("run_fms_model", help="Run fms model")
    fms.add_argument(
        "-d", "--dtype", type=str, required=True, choices=["float16", "float32"]
    )
    fms.add_argument("-a", "--architecture", type=str, required=True)
    fms.add_argument("-v", "--variant", type=str, required=True)
    fms.add_argument("-p", "--model_path", type=str, required=True)
    fms.add_argument("-i", "--input_json_path", type=str, required=True)
    fms.add_argument("-o", "--output_dir", type=str, required=True)

    hf = subparsers.add_parser("run_hf_model", help="Run huggingface model")
    hf.add_argument(
        "-d", "--dtype", type=str, required=True, choices=["float16", "float32"]
    )
    hf.add_argument("-p", "--model_path", type=str, required=True)
    hf.add_argument("-i", "--input_json_path", type=str, required=True)
    hf.add_argument("-o", "--output_dir", type=str, required=True)

    diff = subparsers.add_parser(
        "compute_diff", help="Compute difference in two model outputs"
    )
    diff.add_argument("-f", "--generated_tokens_dir", type=str, required=True)

    args = parser.parse_args(argv[1:])

    if args.subcommand == "run_fms_model":
        torch.set_default_dtype(dtype_mapping[args.dtype])
        model, tokenizer = load_fms_model(
            model_path=args.model_path,
            model_arch=args.architecture,
            model_variant=args.variant,
        )
        generate_and_write_tokens(
            model, tokenizer, args.input_json_path, args.output_dir, "fms"
        )

    elif args.subcommand == "run_hf_model":
        torch.set_default_dtype(dtype_mapping[args.dtype])
        model, tokenizer = load_hf_model(model_path=args.model_path)
        generate_and_write_tokens(
            model, tokenizer, args.input_json_path, args.output_dir, "hf"
        )

    elif args.subcommand == "compute_diff":
        compute_diff(args.generated_tokens_dir)

    else:
        print("Invalid subcommand")


if __name__ == "__main__":
    main(sys.argv)
