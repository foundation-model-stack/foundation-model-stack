import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import warnings
import torch
import json

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

    # Ref: https://github.com/huggingface/tokenizers/blob/ee2c5708bdce9d6610fa74faeb22cf6297c6390a/bindings/python/py_src/tokenizers/implementations/base_tokenizer.py#L192
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode a string into a list of token ids.

        Args:
            text (str): input text string to be tokenized and converted to ids
            add_special_tokens (bool, optional): prefix bos token to the input text

        Returns:
            List[int]: token ids of tokenized text string
        """
        raise NotImplementedError

    # Ref: https://github.com/huggingface/tokenizers/blob/ee2c5708bdce9d6610fa74faeb22cf6297c6390a/bindings/python/py_src/tokenizers/implementations/base_tokenizer.py#L262
    def decode(self, ids: List[int], skip_special_tokens: Optional[bool] = True) -> str:
        """Decode the given list of ids to a string sequence

        Args:
            ids: List[unsigned int]:
                A list of ids to be decoded

            skip_special_tokens: (`optional`) boolean:
                Whether to remove all the special tokens from the output string

        Returns:
            The decoded string
        """
        raise NotImplementedError

    def tokenize(self, text: str):
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids: torch.LongTensor):
        warnings.warn(
            "this method will be deprecated in future versions, use HF API tokenizer.decode instead",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]):
        warnings.warn(
            "this method will be deprecated in future versions, use HF API tokenizer.encode instead",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: list[str]):
        warnings.warn(
            "this method will be deprecated in future versions, use HF API tokenizer.decode instead",
            DeprecationWarning,
            stacklevel=2,
        )
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
        warnings.warn(
            "this method will be deprecated in future versions, use HF API tokenizer.decode instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return [chr(i) for i in ids]

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]):
        warnings.warn(
            "this method will be deprecated in future versions, use HF API tokenizer.encode instead",
            DeprecationWarning,
            stacklevel=2,
        )
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
        warnings.warn(
            "this method will be deprecated in future versions, use HF API tokenizer.decode instead",
            DeprecationWarning,
            stacklevel=2,
        )
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
        warnings.warn(
            "this method will be deprecated in future versions, use HF API tokenizer.decode instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.sp_model.id_to_piece(ids)

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]):
        warnings.warn(
            "this method will be deprecated in future versions, use HF API tokenizer.encode instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.sp_model.piece_to_id(tokens)

    def convert_tokens_to_string(self, tokens: list[str]):
        warnings.warn(
            "this method will be deprecated in future versions, use HF API tokenizer.decode instead",
            DeprecationWarning,
            stacklevel=2,
        )
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
        self.padding_side = self.tokenizer.padding_side
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token = self.tokenizer.unk_token
        self.unk_token_id = self.tokenizer.unk_token_id
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = False,
    ):
        return self.tokenizer.batch_decode(sequences, skip_special_tokens)

    def tokenize(self, text: str):
        return self.tokenizer.tokenize(text)

    def convert_ids_to_tokens(self, ids: torch.LongTensor):
        warnings.warn(
            "this method will be deprecated in future versions, use HF API tokenizer.decode instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.tokenizer.convert_ids_to_tokens(ids)

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]):
        warnings.warn(
            "this method will be deprecated in future versions, use HF API tokenizer.encode instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_tokens_to_string(self, tokens: list[str]):
        warnings.warn(
            "this method will be deprecated in future versions, use HF API tokenizer.decode instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.tokenizer.convert_tokens_to_string(tokens)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text, add_special_tokens=False):
        if (
            add_special_tokens is True
            and self.tokenizer.bos_token_id != self.tokenizer.eos_token_id
        ):
            return [self.tokenizer.bos_token_id] + self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(text)
            )
        else:
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))


"""
MistralAI Tokenizer implementation for foundation-model-stack
Add this class to fms/utils/tokenizers.py
"""


class _TekkenTokenizer(BaseTokenizer):
    """
    MistralAI tekken tokenizer wrapper that follows the same interface as other tokenizers
    in the foundation model stack.

    Supports Tekken (tiktoken-based)
    Requires: pip install mistral-common --upgrade
    """

    def __init__(self, model_path: Union[str, Path], **kwargs):
        self.model_path = Path(model_path)
        self.config = self._load_config()
        self.system_prompt = self._load_system_prompt()

        # Try to load the tokenizer from the model path
        self.tokenizer = self._load_tokenizer()

        # vocab_size
        self._vocab_size = self.tokenizer.n_words

        # Get special token IDs from config or use defaults
        self.bos_token_id = self.config.get("bos_token_id", 1)
        self.eos_token_id = self.config.get("eos_token_id", 2)
        self.unk_token_id = self.config.get("unk_token_id", 0)
        self.pad_token_id = self.config.get("pad_token_id", self.eos_token_id)

        # Store commonly used special tokens (strings)
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.pad_token = self.eos_token  # Mistral typically uses EOS as pad

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json"""
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load_system_prompt(self) -> str:
        """Load system promp from model path"""
        system_prompt = self.model_path / "SYSTEM_PROMPT.txt"
        if system_prompt.exists():
            with open(system_prompt, "r") as f:
                return f.read()
        return ""

    def _load_tokenizer(self):
        """
        Load the tekken tokenizer
        https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/tekken.py
        """
        # Check for Tekken tokenizer (newer models)
        tekken_path = self.model_path / "tekken.json"
        if tekken_path.exists():
            try:
                from mistral_common.tokens.tokenizers.tekken import Tekkenizer  # type: ignore
            except ImportError:
                raise ImportError(
                    "mistral-common is required for MistralAI tokenizers. "
                    "Please install it with: pip install mistral-common --upgrade"
                )

            try:
                return Tekkenizer.from_file(str(tekken_path))
            except Exception as e:
                print(f"Error: Failed to load Tekken tokenizer from {tekken_path}: {e}")
                raise RuntimeError(
                    f"Tekkenizer failed to load tekken.json at model_path {self.model_path}"
                )
        else:
            raise RuntimeError(
                f"Tekkenizer Error: Could not find the tekken.json at model_path {self.model_path}"
            )

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode a string into a list of token ids.

        Args:
            text (str): input text string to be tokenized and converted to ids
            add_special_tokens (bool, optional): prefix bos token to the input text

        Returns:
            List[int]: token ids of tokenized text string
        """
        return self.tokenizer.encode(text, bos=add_special_tokens, eos=False)

    def decode(self, ids: List[int], skip_special_tokens: Optional[bool] = True) -> str:
        """Decode the given list of ids to a string sequence

        Args:
            ids: List[unsigned int]:
                A list of ids to be decoded

            skip_special_tokens: (`optional`) boolean:
                Whether to remove all the special tokens from the output string
                Mistral: SpecialTokenPolicy
                IGNORE = 0 --> skip_special_tokens = True
                KEEP = 1  --> skip_special_tokens = False

        Returns:
            The decoded string
        """
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy  # type: ignore

        if skip_special_tokens:
            stp = SpecialTokenPolicy.IGNORE  # type: ignore
        else:
            stp = SpecialTokenPolicy.KEEP  # type: ignore

        return self.tokenizer.decode(ids, stp)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens (strings)

        Args:
            text: Input text to tokenize
        Returns:
            List of string tokens
        """
        warnings.warn(
            "this method will be deprecated in future versions, this will be a lot more inefficient than encode, directly use encode(text: str) -> List[int] method instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.ids = self.encode(text)
        return list(map(self.tokenizer.decode, [[id] for id in self.ids]))

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]) -> List[int]:
        """
        Convert tokens to token IDs

        Args:
            tokens:

        Returns:
            List of token IDs
        """
        warnings.warn(
            "this method will be deprecated in future versions, use encode(text: str) -> List[int] method instead",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            if len(self.ids) == len(tokens):
                return self.ids
        except Exception as e:
            raise RuntimeError(
                f"Misrtal tokenizer error: convert_tokens_to_ids() must be used in tandem with tokenize() Error: {type(e).__name__} occurred: {e}"
            )
        return [0]

    def convert_ids_to_tokens(self, ids) -> List[str]:
        """
        Convert token IDs to tokens

        Args:
            ids: List of token IDs

        Returns:
            List of token strings
        """
        warnings.warn(
            "this method will be deprecated in future versions, for being inefficient, use decode(token_ids: List[int]) -> str method instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        if isinstance(ids, list) and not all(
            isinstance(element, int) for element in ids
        ):
            ids = ids[0]

        return list(map(self.tokenizer.decode, [[i] for i in ids]))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Convert list of string tokens

        Args:
            tokens (list[str]): List of the strings

        Returns:
            str: joined token strings
        """
        return "".join(tokens)

    def vocab_size(self):
        """Vocabulary size of the tokenizer."""
        return self._vocab_size


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

    model_path = Path(name)
    if style == "tekken" or (model_path / "tekken.json").exists():
        return _TekkenTokenizer(name)

    if style is None or style == "hf":
        return _HFTokenizer(name)

    if style is None:
        raise RuntimeError(f"Could not find a tokenzier {name}")
    else:
        raise RuntimeError(f"Could not find a {style} tokenizer with name {name}")
