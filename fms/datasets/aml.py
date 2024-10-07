import csv
import os
import random
from typing import Optional

import torch
from torch.utils.data import Dataset

from fms.utils import tokenizers


class AMLDataset(Dataset):
    """
    Expects a csv file containing rows of the form:
    type,sentiment,text
    business,1,"The Institut..."
    This is the same format as used in the Twitter dataset (internal url)
    """

    def __init__(
        self,
        path: str,
        tokenizer: tokenizers.BaseTokenizer,
        max_len: int = 512,
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
        full_path = os.path.expanduser(path)
        with open(full_path, "r", newline="", encoding="utf-8") as csv_file:
            aml_reader = csv.reader(csv_file, delimiter=",")
            header = True
            for row in aml_reader:
                if header:
                    header = False
                    continue
                self.input_data.append(row)

        # shuffle the input data
        random.shuffle(self.input_data)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        label = int(self.input_data[index][1])
        input_text = self.input_data[index][2]
        input_text = self.tokenizer.tokenize(input_text)
        input_text = self.tokenizer.convert_tokens_to_ids(input_text)

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
