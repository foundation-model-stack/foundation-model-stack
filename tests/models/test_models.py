import os
import shutil
import tempfile
from collections import ChainMap
from glob import glob
from pathlib import Path

import pytest
import torch

from fms import models
from fms.modules import UninitializedModule
from fms.testing.comparison import (
    HFModelSignatureParams,
    ModelSignatureParams,
    compare_model_signatures,
)
from fms.utils import serialization


def test_register():
    with pytest.raises(KeyError):
        models.register_model("llama", "7b", lambda x: x)


def test_get_model_hf_configured():
    # we want to force download
    model_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--FacebookAI--roberta-base"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    model_inferred = models.get_model("hf_configured", "FacebookAI/roberta-base")
    model_given = models.get_model("roberta", "base")
    assert model_inferred.config.as_dict() == model_given.config.as_dict()

    found_file = False
    for filename in glob(f"{model_path}/**/config.json", recursive=True):
        found_file = True
        assert os.listdir(filename.replace("/config.json", "")) == ["config.json"]

    assert found_file


def test_get_model_hf_pretrained():
    from transformers import AutoModelForCausalLM

    model_pretrained = models.get_model(
        "hf_pretrained", "bigcode/tiny_starcoder_py", data_type=torch.float32
    )
    hf_model_pretrained = AutoModelForCausalLM.from_pretrained(
        "bigcode/tiny_starcoder_py", torch_dtype=torch.float32
    )

    model_pretrained.eval()
    hf_model_pretrained.eval()

    inp = torch.arange(5, 15).unsqueeze(0)
    fms_signature_params = ModelSignatureParams(
        model=model_pretrained, params=1, inp=inp
    )
    hf_signature_params = HFModelSignatureParams(
        model=hf_model_pretrained,
        params=["input_ids", "labels"],
        other_params={"return_dict": True},
        inp=inp,
    )

    compare_model_signatures(fms_signature_params, hf_signature_params)


def test_get_model_hf_pretrained_invalid_arguments():
    with pytest.raises(ValueError):
        models.get_model(
            "hf_pretrained", "bigcode/gpt_bigcode-santacoder", model_path="."
        )
    with pytest.raises(ValueError):
        models.get_model(
            "hf_pretrained", "bigcode/gpt_bigcode-santacoder", source="meta"
        )


def test_getmodel():
    with pytest.raises(KeyError):
        models._get_model_instance("foo", "foo")
    with pytest.raises(KeyError):
        models._get_model_instance("llama", "foo")
    with pytest.raises(KeyError):
        models.list_variants("foo")

    micro = models._get_model_instance("llama", "micro")
    assert micro.config.nlayers == 5


@pytest.mark.autogptq
def test_uninitialized_module():
    model = models._get_model_instance(
        architecture="llama",
        variant="micro",
        extra_args={
            "linear_config": {
                "linear_type": "gptq",
                "group_size": 128,
                "desc_act": False,
            }
        },
    )
    with pytest.raises(RuntimeError):
        model(torch.tensor([[0, 1, 2]]))

    model = models.get_model(
        "llama",
        "micro",
        linear_config={
            "linear_type": "gptq",
            "group_size": 128,
            "desc_act": False,
        },
    )

    for module in model.modules():
        assert not isinstance(module, UninitializedModule)


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
        assert type(newsd) is ChainMap
        for key in keys:
            assert key in newsd
            torch.testing.assert_close(sd[key], as_loaded[key])


def test_guess_numlayers():
    model = models._get_model_instance("llama", "micro")
    sd = model.state_dict()
    assert models._guess_num_layers(sd) == 5


def _make_ministral3_small():
    """Return a tiny Ministral3Config suitable for unit tests (no weights needed)."""
    from fms.models.ministral3 import Ministral3Config, Ministral3TextConfig
    from fms.models.pixtral_vision import PixtralVisionConfig

    text_cfg = Ministral3TextConfig(
        src_vocab_size=384,
        nheads=8,
        nlayers=2,
        hidden_grow_factor=3.5,
        multiple_of=2,
        emb_dim=16,
        head_dim=64,
        max_expected_seq_len=1024,
        kvheads=2,
        fused_weights=True,
        pad_id=0,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 1000.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 1.0,
            "original_max_position_embeddings": 512,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
        },
    )
    vision_cfg = PixtralVisionConfig(
        hidden_size=16,
        intermediate_size=64,
        nlayers=2,
        nheads=8,
        nchannels=3,
        image_size=280,
        patch_size=14,
        fused_weights=True,
    )
    return Ministral3Config(
        text_config=text_cfg,
        vision_config=vision_cfg,
        spatial_merge_size=2,
        image_token_index=10,
        vision_feature_layer=-1,
    )


def test_get_model_vision_only_flag_passed_to_model():
    """vision_only=True is forwarded through _get_model_instance to the constructor."""
    from fms.models.llava_next import LlavaNext

    # llava_next has no rope-params config issue and its factory accepts **kwargs cleanly
    model = models._get_model_instance(
        "llava_next",
        "granite_vision_3_2_2b",
        extra_args={"vision_only": True},
    )
    assert isinstance(model, LlavaNext)
    assert model._vision_only is True
    assert hasattr(model, "text_embedding")
    assert not hasattr(model, "language_model")


def test_get_model_vision_only_default_false():
    """vision_only defaults to False — language_model is present without the flag."""
    from fms.models.llava_next import LlavaNext

    model = models._get_model_instance(
        "llava_next",
        "granite_vision_3_2_2b",
        extra_args={"vision_only": False},
    )
    assert isinstance(model, LlavaNext)
    assert model._vision_only is False
    assert hasattr(model, "language_model")
    assert not hasattr(model, "text_embedding")


def test_get_model_vision_only_prefix_filter_applied(tmp_path):
    """get_model passes key_prefix_filter when vision_only=True, skipping LLM keys."""
    from safetensors.torch import save_file
    from fms.models.llava_next import LlavaNext

    full_model = models._get_model_instance("llava_next", "granite_vision_3_2_2b")

    # Stamp vision components with 1.0 and language components with 2.0 so we
    # can assert which were loaded without relying on random-init values.
    with torch.no_grad():
        for name, p in full_model.named_parameters():
            if name.startswith("vision_tower.") or name.startswith(
                "multi_modal_projector."
            ):
                p.fill_(1.0)
            elif name.startswith("language_model."):
                p.fill_(2.0)

    save_file(
        {k: v.contiguous() for k, v in full_model.state_dict().items()},
        tmp_path / "model.safetensors",
    )

    vision_model = models.get_model(
        "llava_next",
        "granite_vision_3_2_2b",
        model_path=str(tmp_path),
        vision_only=True,
    )
    assert isinstance(vision_model, LlavaNext)
    assert vision_model._vision_only is True
    assert hasattr(vision_model, "text_embedding")
    assert not hasattr(vision_model, "language_model")

    # Vision tower and projector params should be 1.0 (loaded from checkpoint)
    for name, p in vision_model.named_parameters():
        if name.startswith("vision_tower.") or name.startswith(
            "multi_modal_projector."
        ):
            assert p.eq(1.0).all(), f"{name} was not loaded from checkpoint"
