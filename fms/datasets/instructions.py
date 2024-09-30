import json
import os
import urllib
from typing import Dict

import requests
import torch
from torch.utils.data import Dataset

from fms.utils import tokenizers


_instruction_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
_instruction_nocontext_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


class JsonInstructions(Dataset):
    """
    Expects a json file containing rows of the form:
    {
        "instruction":"a question or request made to the model",
        "input":"context to be considered or referenced by the instruction",
        "output":"expected result"
    }
    This is the same format as used in the Alpaca dataset:
    https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
    """

    def __init__(
        self,
        path: str,
        tokenizer: tokenizers.BaseTokenizer,
        max_len: int = 1024,
        ignore_index=-100,
    ):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.max_len = max_len
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        if urllib.parse.urlparse(path).scheme == "":
            file = os.path.expanduser(path)
            with open(file, "r", encoding="utf-8") as reader:
                text = reader.read()
                self.instructions = json.loads(text)
        else:
            text = requests.get(path).text
            self.instructions = json.loads(text)

    def __len__(self):
        return len(self.instructions)

    def make_prompt(self, instruction: Dict) -> str:
        if "input" in instruction:
            prompt = _instruction_template.format_map(instruction)
        else:
            prompt = _instruction_nocontext_template.format_map(instruction)
        return prompt

    def __getitem__(self, index):
        instruction = self.instructions[index]
        prompt = self.make_prompt(instruction)
        prompt = self.tokenizer.tokenize(prompt)
        prompt = self.tokenizer.convert_tokens_to_ids(prompt)

        response = instruction["output"]
        response = self.tokenizer.tokenize(response)
        response = self.tokenizer.convert_tokens_to_ids(response)

        example = prompt + response

        if self.bos_token_id is not None:
            example = [self.bos_token_id] + example

        if self.eos_token_id is not None:
            example = example + [self.eos_token_id]

        example = torch.tensor(example, dtype=torch.long)

        input = example[:-1]

        label = example[1:].clone()
        label[: len(prompt)] = self.ignore_index

        if input.shape[0] > self.max_len:
            input = input[-self.max_len :]
            label = input[-self.max_len :]

        return input, label


class JsonCausal(Dataset):
    """
    Expects a json file containing rows of the form:
    {
        "full_training_prompt": "a question or request made to the model",
        "messages": [...],
    }
    """

    def __init__(
        self,
        path: str,
        tokenizer: tokenizers.BaseTokenizer,
        max_len: int = 512,
        pad_id: int = 0,
        ignore_index=-100,
    ):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.max_len = max_len
        self.pad_id = pad_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.inputs = []
        file = os.path.expanduser(path)
        with open(file, "r", encoding="utf-8") as reader:
            for line in reader:
                self.inputs.append(json.loads(line))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        text = self.inputs[index]["interaction"][0]
        example = f"Question:\n{text['utterance']}\n\n\nAnswer:\n{text['query']}"
        example = self.tokenizer.tokenize(example)
        response_idx = example.index("Answer")
        example = self.tokenizer.convert_tokens_to_ids(example)

        if self.bos_token_id is not None:
            example = [self.bos_token_id] + example

        if self.eos_token_id is not None:
            example = example + [self.eos_token_id]

        example = torch.tensor(example, dtype=torch.long)
        input = example[:-1]
        label = example[1:].clone()
        label[:response_idx] = self.ignore_index

        if self.pad_id is not None and input.shape[0] < self.max_len - 1:
            pad = torch.zeros(self.max_len - input.shape[0], dtype=torch.long)
            pad.fill_(self.pad_id)
            input = torch.cat((pad, input), dim=0)
            label = torch.cat((pad.fill_(self.ignore_index), label), dim=0)

        if input.shape[0] > self.max_len:
            input = input[-self.max_len :]
            label = input[-self.max_len :]

        return input, label
