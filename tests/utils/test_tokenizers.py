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
    assert fm_tokenizer.convert_ids_to_tokens([0, 1, 2]) == hf_tokenizer.convert_ids_to_tokens([0, 1, 2])
    assert fm_tokenizer.convert_tokens_to_ids(["hello", "world"]) == hf_tokenizer.convert_tokens_to_ids(
        ["hello", "world"]
    )
    assert fm_tokenizer.convert_tokens_to_string(["hello", "world"]) == hf_tokenizer.convert_tokens_to_string(
        ["hello", "world"]
    )


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
