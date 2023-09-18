import pytest

from fms.models.llama import LLaMA, LLaMAConfig


def test_config_params_passed_as_kwargs_to_model():
    params = {
        "src_vocab_size": 256,
        "emb_dim": 16,
        "multiple_of": 2,
        "nheads": 2,
        "nlayers": 2,
        "norm_eps": 1e-05,
        "pad_id": 0,
    }
    config = LLaMAConfig(**params)
    model = LLaMA(**params)
    assert model.get_config().as_dict() == config.as_dict()


def test_config_passed_to_model():
    config = LLaMAConfig(
        src_vocab_size=256,
        emb_dim=16,
        multiple_of=2,
        nheads=2,
        nlayers=2,
        norm_eps=1e-05,
        pad_id=0,
    )
    model = LLaMA(config)
    assert model.get_config().as_dict() == config.as_dict()


def test_config_passed_to_model_and_updated():
    config = LLaMAConfig(
        src_vocab_size=256,
        emb_dim=16,
        multiple_of=2,
        nheads=2,
        nlayers=2,
        norm_eps=1e-05,
        pad_id=0,
    )
    model = LLaMA(config, pad_id=1)
    # check not same reference
    assert model.get_config() is not config

    # modify pad_id to the new value expected and check equivalence
    config.pad_id = 1
    assert model.get_config().as_dict() == config.as_dict()
