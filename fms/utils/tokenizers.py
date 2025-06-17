import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import torch
import json

from fms import utils

# git repo : https://github.com/mistralai/mistral-common 
try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.messages import SystemMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.base import Tokenized
    # from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
    _mistral_available = True
except ImportError:
    _mistral_available = False


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
        return self.tokenizer.convert_ids_to_tokens(ids)

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_tokens_to_string(self, tokens: list[str]):
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

class _MistralTokenizer(BaseTokenizer):
    """
    MistralAI tokenizer wrapper that follows the same interface as other tokenizers
    in the foundation model stack.
    
    Supports both Tekken (tiktoken-based) and SentencePiece tokenizers:
    - Tekken: uses tekken.json (newer models like Devstral, Mistral Nemo)
    - SentencePiece: uses tokenizer.model.v3 (older models like Mistral-7B-v0.3)
    
    Requires: pip install mistral-common --upgrade
    """
    
    def __init__(self, model_path: Union[str, Path], **kwargs):
        if not _mistral_available:
            raise ImportError(
                "mistral-common is required for MistralAI tokenizers. "
                "Please install it with: pip install mistral-common --upgrade"
            )
        
        self.model_path = Path(model_path)
        self.config = self._load_config()
        self.system_prompt = self._load_system_prompt()
        
        # Try to load the tokenizer from the model path
        self.tokenizer = self._load_tokenizer()
        
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
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_system_prompt(self) -> str:
        """Load system promp from model path"""
        system_prompt = self.model_path / "SYSTEM_PROMPT.txt"
        if system_prompt.exists():
            with open(system_prompt, 'r') as f:
                return f.read()
        return ""

    def _load_tokenizer(self) -> MistralTokenizer:
        """Load the appropriate tokenizer based on available files
        https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/tokens/tokenizers/mistral.py#L227  --> from_file( 
        https://github.com/mistralai/mistral-common/blob/4b9674a58a29907588afbc477663ea810b0657df/src/mistral_common/tokens/tokenizers/mistral.py#L227
        """
        # Check for Tekken tokenizer (newer models)
        tekken_path = self.model_path / "tekken.json"
        if tekken_path.exists():
            try:
                return MistralTokenizer.from_file(str(tekken_path))
            except Exception as e:
                print(f"Warning: Failed to load Tekken tokenizer from {tekken_path}: {e}")
        
        # Check for SentencePiece tokenizer v3 (older models)
        tokenizer_v3_path = self.model_path / "tokenizer.model.v3"
        if tokenizer_v3_path.exists():
            try:
                return MistralTokenizer.from_file(str(tokenizer_v3_path))
            except Exception as e:
                print(f"Warning: Failed to load SentencePiece v3 tokenizer from {tokenizer_v3_path}: {e}")
        
        # Check for standard SentencePiece tokenizer
        tokenizer_path = self.model_path / "tokenizer.model"
        if tokenizer_path.exists():
            try:
                return MistralTokenizer.from_file(str(tokenizer_path))
            except Exception as e:
                print(f"Warning: Failed to load SentencePiece tokenizer from {tokenizer_path}: {e}")
        
        # Fallback: try to load from model name if it's a known model
        model_name = self.model_path.name
        try:
            return MistralTokenizer.from_model(model_name)
        except Exception as e:
            raise ValueError(
                f"Could not load MistralAI tokenizer from {self.model_path}. "
                f"Expected files: tekken.json, tokenizer.model.v3, or tokenizer.model. "
                f"Error: {e}"
            )
    
    def tokenize(self, text: str, with_system_prompt: Optional[bool] = False) -> List[str]:
        """
        Tokenize text into tokens (strings)
        
        Args:
            text: Input text to tokenize
            with_system_prompt: Optional[bool] = True -> adds the SYSTEM_PROMPT of the model 
            
        Returns:
            List of string tokens 
        """
        if with_system_prompt == True:
            chat_completion = ChatCompletionRequest(
                messages=[
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    UserMessage(content=text)],)
        else:
            chat_completion = ChatCompletionRequest(
                messages=[UserMessage(content=text)],)
            
        self.tokenized = self.tokenizer.encode_chat_completion(chat_completion)

        return self.convert_ids_to_tokens(self.tokenized.tokens)




    def convert_tokenized_to_ids(self, tokenized: Tokenized) -> List[int]:
        """
        Convert tokens to token IDs
        
        Args:
            tokenized: Mistral Tokenized object 
            https://github.com/mistralai/mistral-common/blob/4b9674a58a29907588afbc477663ea810b0657df/src/mistral_common/tokens/tokenizers/base.py#L142
            
        Returns:
            List of token IDs
        """
        return tokenized.tokens
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to token IDs
        
        Args:
            tokenized: Mistral Tokenized object 
            https://github.com/mistralai/mistral-common/blob/4b9674a58a29907588afbc477663ea810b0657df/src/mistral_common/tokens/tokenizers/base.py#L142
            
        Returns:
            List of token IDs
        """
        # Quick check if tokens 
        if tokens == self.convert_ids_to_tokens(self.tokenized.tokens):
            return self.tokenized.tokens
        else:
            raise RuntimeError(f"Misrtral tokenizer error: tokenized list should not be modified")

    
    def convert_ids_to_tokens(self, ids) -> List[str]:
        """
        Convert token IDs to tokens
        
        Args:
            ids: List of token IDs
            
        Returns:
            List of token strings
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        if isinstance(ids, list) and not all(isinstance(element, int) for element in ids):
            ids = ids[0]

        return list(map(self.tokenizer.decode, [[i] for i in ids]))

    def convert_tokens_to_string(self, tokens: list[str]):
        return " ".join(tokens)


    def encode(self, text: str, add_special_tokens: bool = False ) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        if (
            add_special_tokens is True
            and self.tokenizer.bos_token_id != self.tokenizer.eos_token_id
        ):
            return [self.tokenizer.bos_token_id] + self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(text)
            )
        else:
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    # def batch_decode(
    #     self,
    #     sequences: Union[List[int], List[List[int]]],
    #     skip_special_tokens: bool = False,
    # ):
    #     return self.tokenizer.batch_decode(sequences, skip_special_tokens)


    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        # return self.tokenizer.decode(token_ids, special_token_policy=SpecialTokenPolicy.IGNORE)
        return self.tokenizer.decode(token_ids, special_token_policy=0)
    
    
   


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

    if style == "mistralai":
        return _MistralTokenizer(name)

    # Check for MistralAI tokenizer files (prioritize Tekken)
    model_path = Path(name)
    mistral_files = ["tekken.json", "tokenizer.model.v3", "tokenizer.model"]
    has_mistral_tokenizer = any((model_path / f).exists() for f in mistral_files)
    if has_mistral_tokenizer:
        # Check if it's likely a Mistral model
        model_name = model_path.name.lower()
        mistral_keywords = ["mistral", "codestral", "devstral", "mixtral", "pixtral"]
        if any(keyword in model_name for keyword in mistral_keywords):
            return _MistralTokenizer(name)
    
    # Also check config.json for model_type
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            if config.get("architectures")[0] == "MistralForCausalLM":
                return _MistralTokenizer(name)
        except Exception:
            pass


    if style is None:
        raise RuntimeError(f"Could not find a tokenzier {name}")
    else:
        raise RuntimeError(f"Could not find a {style} tokenizer with name {name}")
