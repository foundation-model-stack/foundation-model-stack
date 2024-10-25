import tempfile
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
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
