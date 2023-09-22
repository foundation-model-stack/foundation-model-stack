import os
from typing import List, Union

import torch
import torch.nn.functional as F


# constants for common tokenizers

char_tokenizer = "char_tokenizer"
gpt_neox_20b = "EleutherAI/gpt-neox-20b"
gpt_neox_125m = "EleutherAI/gpt-neox-125M"


def has_package(name):
    try:
        __import__(name)
    except ImportError:
        return False
    else:
        return True


_has_hf = has_package("transformers")
_has_sp = has_package("sentencepiece")


class BaseTokenizer:
    """
    A simplistic tokenizer interface duck-type compatible with HuggingFace
    tokenizers. An implementation of this interface could be used in fm in
    cases where we'd like to write tests that don't depend on HF.
    """

    def tokenize(self, text: str):
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids: torch.LongTensor):
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens: list[str]):
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: list[str]):
        raise NotImplementedError

    def vocab_size(self):
        raise NotImplementedError


class CharTokenizer(BaseTokenizer):
    """
    This is essentially the tokenizer used by minGPT. Every character
    is a token, tokenized as ord(c). Vocab size is 256.
    """

    def __init__(self):
        super().__init__()

    def tokenize(self, text: str):
        return list(text)

    def convert_ids_to_tokens(self, ids: torch.LongTensor):
        return [chr(i) for i in ids]

    def convert_tokens_to_ids(self, tokens: list[str]):
        return [ord(t) for t in tokens]

    def convert_tokens_to_string(self, tokens: list[str]):
        return "".join(tokens)

    def vocab_size(self):
        return 256


class _SentencePieceTokenizer(BaseTokenizer):
    """
    An adapter for a sentencepiece tokenizer.
    """

    def __init__(self, path: str):
        super().__init__()
        from sentencepiece import SentencePieceProcessor

        self.sp_model = SentencePieceProcessor(model_file=path)

    def tokenize(self, text: str):
        return self.sp_model.encode_as_pieces(text)

    def convert_ids_to_tokens(self, ids: Union[List[int], torch.LongTensor]):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.sp_model.id_to_piece(ids)

    def convert_tokens_to_ids(self, tokens: list[str]):
        return self.sp_model.piece_to_id(tokens)

    def convert_tokens_to_string(self, tokens: list[str]):
        return self.sp_model.decode(tokens)

    def vocab_size(self):
        return self.sp_model.vocab_size()


class _HFTokenizer(BaseTokenizer):
    """
    An adapter over a HuggingFace tokenizer.
    """

    def __init__(self, name: str):
        super().__init__()
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)

    def tokenize(self, text: str):
        return self.tokenizer.tokenize(text)

    def convert_ids_to_tokens(self, ids: torch.LongTensor):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def convert_tokens_to_ids(self, tokens: list[str]):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_tokens_to_string(self, tokens: list[str]):
        return self.tokenizer.convert_tokens_to_string(tokens)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()


def get_tokenizer(name: str) -> BaseTokenizer:
    """
    Hack to get an instance of a tokenizer by name or path.

    Tries to derive whether the name refers to a custom tokenizer, a
    sentencepiece model file, or a HuggingFace tokenizer.
    """
    if name == "char_tokenizer":
        return CharTokenizer()
    # SentencePiece saves models as .model files.
    # It would be better to identify the type of the file accurately, e.g. using protobuf:
    # https://github.com/google/sentencepiece/issues/121
    elif len(name) >= len(".model") and name[-len(".model") :] == ".model":
        name = os.path.expanduser(name)
        if not os.path.exists(name):
            raise RuntimeError(f"Could not find SentencePiece model at '{name}'")
        if not _has_sp:
            raise RuntimeError(
                f"'{name}' appears to be a sentencepiece tokenizer but sentencepiece is not installed"
            )
        return _SentencePieceTokenizer(name)
    if not _has_hf:
        raise RuntimeError(
            f"Could not find tokenizer '{name}' and HuggingFace transformers is not installed"
        )
    return _HFTokenizer(name)
