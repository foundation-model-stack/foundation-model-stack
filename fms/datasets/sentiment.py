import json
import os
import urllib
from typing import Optional

import requests
import torch
from torch.utils.data import Dataset

from fms.utils import tokenizers


class JsonSentiment(Dataset):
    """
    Expects a json file containing rows of the form:
    {
        "Tweet text": "a complaint tweet",
        "Label": 1 or 2,
        ...
    }
    This is the same format as used in the Twitter dataset (internal url)
    """

    def __init__(
        self,
        path: str,
        tokenizer: tokenizers.BaseTokenizer,
        max_len: int = 1024,
        pad_token: Optional[str] = None,
        ignore_index=-100,
    ):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.max_len = max_len
        if pad_token is not None:
            self.pad_id = pad_token
        else:
            self.pad_id = None
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.input_data = []
        if urllib.parse.urlparse(path).scheme == "":
            file = os.path.expanduser(path)
            with open(file, "r", encoding="utf-8") as reader:
                for line in reader:
                    self.input_data.append(json.loads(line))
        else:
            text = requests.get(path).text
            for line in text.split("\n"):
                self.input_data.append(json.loads(line))

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        input_text = self.input_data[index]["Tweet text"]
        input_text = self.tokenizer.tokenize(input_text)
        input_text = self.tokenizer.convert_tokens_to_ids(input_text)

        label = self.input_data[index]["Label"] - 1

        if self.bos_token_id is not None:
            input_text = [self.bos_token_id] + input_text

        if self.eos_token_id is not None:
            input_text = input_text + [self.eos_token_id]

        input = torch.tensor(input_text, dtype=torch.long)

        if self.pad_id is not None and input.shape[0] < self.max_len:
            pad = torch.zeros(self.max_len - input.shape[0], dtype=torch.long)
            pad.fill_(self.pad_id)
            input = torch.cat((pad, input), dim=0)

        if input.shape[0] > self.max_len:
            input = input[-self.max_len :]

        return input, label


class JsonCausal(Dataset):
    """
    Expects a json file containing rows of the form:
    {
        "Tweet text": "a complaint tweet",
        "Label": 1 or 2,
        ...
    }
    This is the same format as used in the Twitter dataset (internal url)
    """

    def __init__(
        self,
        path: str,
        tokenizer: tokenizers.BaseTokenizer,
        max_len: int = 1024,
        pad_token: Optional[str] = None,
        ignore_index=-100,
    ):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.max_len = max_len
        if pad_token is not None:
            self.pad_id = pad_token
        else:
            self.pad_id = None
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.input_data = []
        if urllib.parse.urlparse(path).scheme == "":
            file = os.path.expanduser(path)
            with open(file, "r", encoding="utf-8") as reader:
                for line in reader:
                    self.input_data.append(json.loads(line))
        else:
            text = requests.get(path).text
            for line in text.split("\n"):
                self.input_data.append(json.loads(line))

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        prompt = self.input_data[index]["Tweet text"]
        prompt = self.tokenizer.tokenize(prompt)
        prompt = self.tokenizer.convert_tokens_to_ids(prompt)

        response = self.input_data[index]["output"]
        response = self.tokenizer.tokenize(response)
        response = self.tokenizer.convert_tokens_to_ids(response)

        if self.bos_token_id is not None:
            response = [self.bos_token_id] + response

        if self.eos_token_id is not None:
            response = response + [self.eos_token_id]

        response = torch.tensor(response, dtype=torch.long)

        if self.pad_id is not None and response.shape[0] < self.max_len:
            pad = torch.zeros(self.max_len - response.shape[0], dtype=torch.long)
            pad.fill_(self.pad_id)
            response = torch.cat((pad, response), dim=0)

        if response.shape[0] > self.max_len:
            response = response[-self.max_len :]

        print(self.max_len, self.pad_id)

        input_tokens = response[:-1]
        label = response[1:].clone()
        label[: len(prompt)] = self.ignore_index
        return input_tokens, label
