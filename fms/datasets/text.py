from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
import requests
import urllib
from fms.utils import tokenizers


class CausalTextDatasetFromString(Dataset):
    """
    Creates a dataset from a single text string and tokenizer.
    Since all data comes from a single text, there are no bos/eos tokens used.
    A pad token if specified, is used only on the final row. i.e.
    `pad_token=None` is similar to drop_last in DataLoader.
    """

    def __init__(
        self,
        text: str,
        tokenizer: tokenizers.BaseTokenizer,
        seq_len: int = 1024,
        pad_token: Optional[str] = None,
        device: torch.device | str = "cpu",
        ignore_index: int = -100,
    ):
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.ids = torch.tensor(ids, dtype=torch.long, device=device)
        self.ignore_index = ignore_index
        if pad_token is not None:
            self.pad_id = tokenizer.convert_tokens_to_ids([pad_token])[0]
        else:
            self.pad_id = None
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def to(self, device: torch.device):
        self.ids = self.ids.to(device)
        return self

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        if end_idx >= self.ids.shape[0]:
            end_idx = self.ids.shape[0]
        input = self.ids[start_idx : end_idx - 1]
        label = self.ids[start_idx + 1 : end_idx]

        if self.pad_id is not None and input.shape[0] < self.seq_len:
            pad = torch.zeros(
                self.seq_len - input.shape[0], device=self.ids.device, dtype=torch.long
            )
            pad.fill_(self.pad_id)
            input = torch.cat((pad, input), dim=0)
            label = torch.cat((pad.fill_(self.ignore_index), label), dim=0)
        return input, label

    def __len__(self):
        tokens = self.ids.shape[0]
        if tokens % self.seq_len == 0 or self.pad_id is None:
            return tokens // self.seq_len
        else:
            return (tokens // self.seq_len) + 1


def causaltext(
    path_or_uri: str, tokenizer: tokenizers.BaseTokenizer, *, pad_token=None, **kwargs
) -> Dataset:
    if urllib.parse.urlparse(path_or_uri).scheme == "":
        with open(path_or_uri) as f:
            text = f.read()
            return CausalTextDatasetFromString(
                text, tokenizer, pad_token=pad_token, **kwargs
            )
    else:
        text = requests.get(path_or_uri).text
        return CausalTextDatasetFromString(
            text, tokenizer, pad_token=pad_token, **kwargs
        )


__shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def shakespeare(pad_token=None, tokenizer=tokenizers.char_tokenizer) -> Dataset:
    """
    get a dataset of the complete works of shakespeare
    """
    # TODO: maybe this should cache somewhere?
    return causaltext(
        __shakespeare_url, tokenizers.get_tokenizer(tokenizer), pad_token=pad_token
    )
