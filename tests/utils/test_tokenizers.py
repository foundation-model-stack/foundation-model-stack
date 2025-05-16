import os
import tempfile

import pytest
import sentencepiece as spm
from transformers import AutoTokenizer

from fms.utils.tokenizers import get_tokenizer


def test_hf_compat():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("WARNING: skipping HF tokenizer test b/c transformers not installed")
        return
    tokenizer_name = "EleutherAI/gpt-neox-20b"
    fm_tokenizer = get_tokenizer(tokenizer_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    assert fm_tokenizer.tokenize("hello") == hf_tokenizer.tokenize("hello")
    assert fm_tokenizer.convert_ids_to_tokens(
        [0, 1, 2]
    ) == hf_tokenizer.convert_ids_to_tokens([0, 1, 2])
    assert fm_tokenizer.convert_tokens_to_ids(
        ["hello", "world"]
    ) == hf_tokenizer.convert_tokens_to_ids(["hello", "world"])
    assert fm_tokenizer.convert_tokens_to_string(
        ["hello", "world"]
    ) == hf_tokenizer.convert_tokens_to_string(["hello", "world"])
    assert fm_tokenizer.bos_token_id == hf_tokenizer.bos_token_id
    assert fm_tokenizer.eos_token_id == hf_tokenizer.eos_token_id


def test_styled():
    tokenizer_name = "EleutherAI/gpt-neox-20b"
    get_tokenizer(tokenizer_name, style="hf")
    with pytest.raises(RuntimeError):
        get_tokenizer(tokenizer_name, style="fms")
    with pytest.raises(RuntimeError):
        get_tokenizer(tokenizer_name, style="sentencepiece")


def test_char_tokenizer():
    char_tokenizer = get_tokenizer("char_tokenizer")
    assert char_tokenizer.tokenize("hello") == ["h", "e", "l", "l", "o"]
    assert char_tokenizer.convert_ids_to_tokens([104, 101, 108, 108, 111]) == [
        "h",
        "e",
        "l",
        "l",
        "o",
    ]
    assert char_tokenizer.convert_tokens_to_ids(["h", "e", "l", "l", "o"]) == [
        104,
        101,
        108,
        108,
        111,
    ]
    assert char_tokenizer.convert_tokens_to_string(["h", "e", "l", "l", "o"]) == "hello"
    assert char_tokenizer.bos_token_id == 2
    assert char_tokenizer.eos_token_id == 3


def test_out_of_range_ascii():
    char_tokenizer = get_tokenizer("char_tokenizer")
    tokens = char_tokenizer.convert_tokens_to_ids(["你", "好"])
    # characters out of ascii range are mapped to zero (null)
    assert tokens[0] == 0
    assert tokens[1] == 0


def test_single_token():
    """
    checks if convert_tokens_to_ids handles both single strings and lists. single tokens
    should all be converted into a single integer that represents the id.
    """
    # testing character tokenizer
    char_tokenizer = get_tokenizer("char_tokenizer")
    assert char_tokenizer.convert_tokens_to_ids("h") == 104
    # multi-char strings should error with CharTokenizer
    with pytest.raises(RuntimeError):
        char_tokenizer.convert_tokens_to_ids("le")
    assert char_tokenizer.convert_tokens_to_ids(["h", "e", "le", "l", "o"]) == [
        104,
        101,
        0,  # multi-char strings in lists should be output as 0 for the id
        108,
        111,
    ]
    # testing sentencePiece tokenizer
    with tempfile.TemporaryDirectory() as workdir:
        training_data = os.path.join(workdir, "training_data.txt")
        with open(training_data, "w", encoding="utf-8") as f:
            f.write("This is a test.\n")
            f.write("Please say hello world.")
        model = os.path.join(workdir, "mymodel")
        spm.SentencePieceTrainer.train(
            f"--input={training_data} --model_prefix={model} --vocab_size=76, --model_type=bpe --character_coverage=1.0"
        )
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model}.model")
        sp_tokenizer = get_tokenizer(name=f"{model}.model", style="sentencepiece")
        assert sp_tokenizer.convert_tokens_to_ids("h") == 66
        assert sp_tokenizer.convert_tokens_to_ids("le") == 35
        assert sp_tokenizer.convert_tokens_to_ids(["h", "e", "l", "l", "o"]) == [
            66,
            62,
            63,
            63,
            68,
        ]
    # testing HuggingFace tokenizer
    hf_tokenizer = get_tokenizer("EleutherAI/gpt-neo-125M", style="hf")
    assert hf_tokenizer.convert_tokens_to_ids("h") == 71
    assert hf_tokenizer.convert_tokens_to_ids("le") == 293
    assert hf_tokenizer.convert_tokens_to_ids(["h", "e", "l", "l", "o"]) == [
        71,
        68,
        75,
        75,
        78,
    ]


def test_encode():
    models = ["EleutherAI/gpt-neox-20b", "ibm-granite/granite-3.0-8b-base"]
    for model in models:
        hf_tokenizer = AutoTokenizer.from_pretrained(model)
        fms_tokenizer = get_tokenizer(model, style="hf")
        text = "Hello, how are you today?"
        assert hf_tokenizer.encode(
            text, add_special_tokens=False
        ) == fms_tokenizer.encode(text, add_special_tokens=False)
        assert hf_tokenizer.encode(
            text, add_special_tokens=True
        ) == fms_tokenizer.encode(text, add_special_tokens=True)
