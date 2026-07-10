import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from fms.utils import serialization


def test_register():
    with pytest.raises(KeyError):
        serialization.register_adapter("llama", "meta", lambda x: x)


def test_adapt():
    w1 = torch.arange(0, 5)
    w3 = torch.arange(5, 10)
    sd = {"feed_forward.w1.weight": w1, "feed_forward.w3.weight": w3}
    adapted = serialization.get_adapted("llama", "meta", sd, {})

    newkey = "ff_sub_layer.wg1_fused.weight"
    assert newkey in adapted
    torch.testing.assert_close(adapted[newkey][:5], w1)
    torch.testing.assert_close(adapted[newkey][-5:], w3)

    adapted = serialization.get_adapted("foo", "foo", sd, {})
    print(adapted.keys())
    assert id(adapted["feed_forward.w1.weight"]) == id(w1)
    assert id(adapted["feed_forward.w3.weight"]) == id(w3)


def test_list():
    assert "meta" in serialization.list_sources("llama")


def test_load_state_dict():
    # Test asserts
    with pytest.raises(ValueError, match="can only be loaded into"):
        serialization.load_state_dict(
            "path",
            checkpoint_sharding="fsdp",
            distributed_strategy="tp",
        )
        serialization.load_state_dict(
            "path",
            checkpoint_sharding="tp",
            distributed_strategy="fsdp",
        )
        serialization.load_state_dict(
            "path",
            checkpoint_sharding="tp",
            distributed_strategy=None,
        )

    # Test state dict loading
    with tempfile.TemporaryDirectory() as d:
        sd = {
            "test": torch.tensor([0, 1, 2, 3], dtype=torch.float32),
            "test2": torch.tensor([4, 5, 6, 7], dtype=torch.float32),
        }

        # Single file/glob checkpoint test
        pth_sd_path = Path(d) / "test.pth"
        st_sd_path = Path(d) / "test.safetensors"
        torch.save(sd, pth_sd_path)
        save_file(sd, st_sd_path)

        loaded_sd = serialization.load_state_dict(pth_sd_path)
        torch.testing.assert_close(sd["test"], loaded_sd["test"])

        loaded_sd = serialization.load_state_dict(st_sd_path)
        assert isinstance(loaded_sd.maps[0], serialization.LazySafetensorsDict)
        torch.testing.assert_close(sd["test"], loaded_sd["test"])

        pth_glob_path = str(Path(d)) + "/*.pth"
        loaded_sd = serialization.load_state_dict(pth_glob_path)
        torch.testing.assert_close(sd["test"], loaded_sd["test"])

        # Wrong file name test
        with pytest.raises(
            AssertionError, match="Can't find the requested checkpoint data at"
        ):
            serialization.load_state_dict("path")

        # Safetensors priority test
        loaded_sd = serialization.load_state_dict(Path(d))
        assert isinstance(loaded_sd.maps[0], serialization.LazySafetensorsDict)

        # Sharded checkpoint test
        bin_sd_paths = [Path(d) / "test.00.bin", Path(d) / "test.01.bin"]
        bin_glob_path = str(Path(d)) + "/*.bin"
        torch.save({"test": sd["test"]}, bin_sd_paths[0])
        torch.save({"test2": sd["test2"]}, bin_sd_paths[1])

        layer_sd = serialization.load_state_dict(
            bin_glob_path,
            distributed_strategy="tp",
            checkpoint_sharding=None,
            rank=0,
            world_size=2,
        )
        assert layer_sd.keys() == sd.keys()

        tp_sd = serialization.load_state_dict(
            bin_glob_path,
            distributed_strategy="tp",
            checkpoint_sharding="tp",
            rank=0,
            world_size=2,
        )
        assert list(tp_sd.keys()) == ["test"]

        # Wrong world size
        with pytest.raises(AssertionError, match="sharded checkpoint with len="):
            serialization.load_state_dict(
                bin_glob_path,
                distributed_strategy="tp",
                checkpoint_sharding="tp",
                rank=0,
                world_size=4,
            )


def test_load_state_dict_into_model_key_prefix_filter():
    """key_prefix_filter skips non-matching keys and loads matching ones.

    Keys must have a numeric layer index so _find_key_neighbors groups them
    independently (one key per layer) rather than treating all non-layered keys
    as a single neighbor group.
    """
    import torch.nn as nn

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4, 4))

    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_layers = nn.ModuleList([_Layer()])
            self.language_layers = nn.ModuleList([_Layer()])

    arch = "_test_prefix_filter"
    serialization.register_adapter_step(arch, "noop", lambda sd, **kw: sd)
    serialization.register_adapter(arch, "fms", ["noop"])

    model = _TinyModel()
    vision_weight = torch.ones(4, 4)
    language_weight = torch.full((4, 4), 2.0)
    state_dict = {
        "vision_layers.0.weight": vision_weight,
        "language_layers.0.weight": language_weight,
    }

    serialization.load_state_dict_into_model(
        model=model,
        state_dict=dict(state_dict),
        architecture=arch,
        source="fms",
        key_prefix_filter=("vision_layers.",),
    )

    # vision layer loaded, language layer still zeros (not loaded)
    torch.testing.assert_close(model.vision_layers[0].weight, vision_weight)
    torch.testing.assert_close(
        model.language_layers[0].weight,
        torch.zeros(4, 4),
        msg="language weights should NOT have been loaded",
    )


def test_load_state_dict_into_model_no_filter_loads_all():
    """Without key_prefix_filter all keys are loaded as normal."""
    import torch.nn as nn

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4, 4))

    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_layers = nn.ModuleList([_Layer()])
            self.language_layers = nn.ModuleList([_Layer()])

    arch = "_test_no_filter"
    serialization.register_adapter_step(arch, "noop", lambda sd, **kw: sd)
    serialization.register_adapter(arch, "fms", ["noop"])

    model = _TinyModel()
    vision_weight = torch.ones(4, 4)
    language_weight = torch.full((4, 4), 2.0)
    state_dict = {
        "vision_layers.0.weight": vision_weight,
        "language_layers.0.weight": language_weight,
    }

    serialization.load_state_dict_into_model(
        model=model,
        state_dict=dict(state_dict),
        architecture=arch,
        source="fms",
        key_prefix_filter=None,
    )

    torch.testing.assert_close(model.vision_layers[0].weight, vision_weight)
    torch.testing.assert_close(model.language_layers[0].weight, language_weight)
