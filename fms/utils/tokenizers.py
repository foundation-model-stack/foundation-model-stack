import os
from typing import List, Optional, Union

import torch

from fms import utils


# constants for common tokenizers

char_tokenizer = "char_tokenizer"
gpt_neox_20b = "EleutherAI/gpt-neox-20b"
gpt_neox_125m = "EleutherAI/gpt-neox-125M"


_has_hf = utils.has_package("transformers")
_has_sp = utils.has_package("sentencepiece")


class BaseTokenizer:
    """
    A simplistic tokenizer interface duck-type compatible with HuggingFace
    tokenizers. An implementation of this interface could be used in fm in
    cases where we'd like to write tests that don't depend on HF.
    """

    def __init__(self, bos_id: int, eos_id: int):
        """
        bos_id: the ID representing the beginning-of-sentence token
        eos_id: the ID representing the end-of-sentence token
        """
        self.bos_token_id = bos_id
        self.eos_token_id = eos_id

    def tokenize(self, text: str):
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids: torch.LongTensor):
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]):
        """
        for all tokenizers, a str parameter will be interpreted as a single token,
        and its output will be a single integer that represents the id.
        """
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: list[str]):
        raise NotImplementedError

    def vocab_size(self) -> int:
        raise NotImplementedError


class CharTokenizer(BaseTokenizer):
    """
    This is essentially the tokenizer used by minGPT. Every character
    is a token, tokenized as ord(c). Vocab size is 256.
    """

    def __init__(self):
        # 2, 3 from ascii tables are "start of text" and "end of text"
        super().__init__(2, 3)

    def tokenize(self, text: str):
        return list(text)

    def convert_ids_to_tokens(self, ids: torch.LongTensor):
        return [chr(i) for i in ids]

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]):
        if isinstance(tokens, str):
            # returning a single integer to be compatible with other tokenizers
            if len(tokens) != 1:
                raise RuntimeError(
                    "Only single character str tokens can be converted using the CharTokenizer."
                )
            token_id = ord(tokens)
            return token_id if token_id < 256 else 0
        return [ord(t) if len(t) == 1 and ord(t) < 256 else 0 for t in tokens]

    def convert_tokens_to_string(self, tokens: list[str]):
        return "".join(tokens)

    def vocab_size(self):
        return 256


class _SentencePieceTokenizer(BaseTokenizer):
    """
    An adapter for a sentencepiece tokenizer.
    """

    def __init__(self, path: str):
        from sentencepiece import SentencePieceProcessor  # type: ignore

        self.sp_model = SentencePieceProcessor(model_file=path)
        super().__init__(self.sp_model.bos_id(), self.sp_model.eos_id())

    def tokenize(self, text: str):
        return self.sp_model.encode_as_pieces(text)

    def convert_ids_to_tokens(self, ids: Union[List[int], torch.LongTensor]):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.sp_model.id_to_piece(ids)

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]):
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
        from transformers import AutoTokenizer  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        super().__init__(self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)

    def tokenize(self, text: str):
        return self.tokenizer.tokenize(text)

    def convert_ids_to_tokens(self, ids: torch.LongTensor):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_tokens_to_string(self, tokens: list[str]):
        return self.tokenizer.convert_tokens_to_string(tokens)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()


def get_tokenizer(name: str, style: Optional[str] = None) -> BaseTokenizer:
    """
    Hack to get an instance of a tokenizer by name or path.

    Args:

    style: 'hf', 'sentencepiece', or 'fms'. If not specified, attempt to derive
            the type based on the name.
    """
    if name == "char_tokenizer" and (style is None or style == "fms"):
        return CharTokenizer()

    # SentencePiece saves models as .model files.
    # It would be better to identify the type of the file accurately, e.g. using protobuf:
    # https://github.com/google/sentencepiece/issues/121
    if style == "sentencepiece" or (
        style is None
        and len(name) >= len(".model")
        and name[-len(".model") :] == ".model"
    ):
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
    if style is None or style == "hf":
        return _HFTokenizer(name)
    if style is None:
        raise RuntimeError(f"Could not find a tokenzier {name}")
    else:
        raise RuntimeError(f"Could not find a {style} tokenizer with name {name}")
