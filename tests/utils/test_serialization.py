import pytest
import torch

from fms.utils import serialization


def test_register():
    with pytest.raises(KeyError):
        serialization.register_adapter("llama", "meta", lambda x: x)


def test_adapt():
    t = torch.ones(5)
    sd = {"feed_forward.w3": t}
    adapted = serialization.get_adapted("llama", "meta", sd)

    newkey = "ff_sub_layer.w1"
    assert newkey in adapted
    assert id(adapted[newkey]) == id(t)

    adapted = serialization.get_adapted("foo", "foo", sd)
    print(adapted.keys())
    assert id(adapted["feed_forward.w3"]) == id(t)


def test_list():
    assert "meta" in serialization.list_sources("llama")


def test_load():
    with pytest.raises(ValueError):
        serialization.load_state_dict(
            "path",
            None,
            "llama",
            "7b",
            checkpoint_format="fsdp",
            distributed_strategy="tp",
        )
