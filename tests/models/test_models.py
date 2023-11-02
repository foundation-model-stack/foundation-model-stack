import pytest
from fms import models


def test_register():
    with pytest.raises(KeyError):
        models.register_model("llama", "7b", lambda x: x)


def test_getmodel():
    with pytest.raises(KeyError):
        models.get_model("foo", "foo")
    with pytest.raises(KeyError):
        models.get_model("llama", "foo")

    micro = models.get_model("llama", "micro")
    assert micro.config.nlayers == 5


def test_list():
    assert "llama" in models.list_models()
    assert "7b" in models.list_variants("llama")
