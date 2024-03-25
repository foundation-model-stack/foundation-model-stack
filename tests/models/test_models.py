import tempfile
from collections import ChainMap
from pathlib import Path

import pytest
import torch

from fms import models
from fms.utils import serialization


def test_register():
    with pytest.raises(KeyError):
        models.register_model("llama", "7b", lambda x: x)


def test_getmodel():
    with pytest.raises(KeyError):
        models._get_model_instance("foo", "foo")
    with pytest.raises(KeyError):
        models._get_model_instance("llama", "foo")
    with pytest.raises(KeyError):
        models.list_variants("foo")

    micro = models._get_model_instance("llama", "micro")
    assert micro.config.nlayers == 5


def test_list():
    assert "llama" in models.list_models()
    assert "7b" in models.list_variants("llama")


def test_load():
    m = models.get_model("llama", "micro")
    sd = m.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".pth") as f:
        torch.save(sd, f.name)
        loaded = models.get_model("llama", "micro", f.name)

        keys = loaded.state_dict().keys()
        first = next(iter(keys))
        assert keys == sd.keys()
        torch.testing.assert_close(loaded.state_dict()[first], sd[first])
    with tempfile.TemporaryDirectory() as d:
        keys = sd.keys()
        count = 0
        dicts = []
        current = {}
        for key in keys:
            count += 1
            current |= {key: sd[key]}
            if count % 10 == 0:
                dicts.append(current)
                current = {}

        dicts.append(current)
        for i in range(len(dicts)):
            path = Path(d) / f"{i}.pth"
            torch.save(dicts[i], path)
        newsd = serialization.load_state_dict(d)
        as_loaded = models.get_model("llama", "micro", d).state_dict()
        # this style load, layer-sharded, has to stitch together the state dicts.
        assert type(newsd) == ChainMap
        for key in keys:
            assert key in newsd
            torch.testing.assert_close(sd[key], as_loaded[key])


def test_guess_numlayers():
    model = models._get_model_instance("llama", "micro")
    sd = model.state_dict()
    assert models._guess_num_layers(sd) == 5
