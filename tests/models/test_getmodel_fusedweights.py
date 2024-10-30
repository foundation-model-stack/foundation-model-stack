import tempfile

import pytest
import torch

from fms.models import get_model


# How to set fused_weights to obtain desired `target_model` fusion
#
# ckpt       target_model   fused_weights     tested
#                            FP16    GPTQ
# --------------------------------------------------
# none       fused           True    True     (Y, Y)
# none       unfused         False   False    (Y, Y)
# fused      fused           True    True     (Y, N)
# fused      unfused         False   False    (Y, N)
# unfused    fused           True    Error    (Y, N)
# unfused    unfused         False   False    (Y, N)


# FP16 model
fused_weights = [True, False]
unfuse_ids = ["fused=True", "fused=False"]
# gptq model
fused_weights_gptq = [True, False]
unfuse_ids_gptq = ["fused=True", "fused=False"]

expected_layers_from_fusion = {
    "fused": [".qkv_fused.", ".wg1_fused."],
    "unfused": [".query.", ".key.", ".value.", ".w1.", ".wg."],
}


class TestUnfuseStrategy:
    @pytest.fixture(
        scope="class",
        params=fused_weights,
        ids=unfuse_ids,
    )
    def get_state_dict(self, request):
        # instantiate FP16 model fused/unfused
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        sd = get_model(
            architecture="llama",
            variant="micro",
            model_path=None,
            source="hf",
            fused_weights=request.param,
            linear_config={"linear_type": "torch_linear"},  # same as None
        ).state_dict()
        torch.set_default_dtype(orig_dtype)
        return (sd, request.param)

    @pytest.fixture(
        scope="class",
        params=fused_weights_gptq,
        ids=unfuse_ids_gptq,
    )
    def get_gptq_state_dict(self, request):
        # instantiate GPTQ model fused/unfused
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        sd = get_model(
            architecture="llama",
            variant="micro",
            model_path=None,
            source="hf",
            fused_weights=request.param,
            linear_config={
                "linear_type": "gptq",
                "group_size": 2,
                "use_marlin": False,
                "disable_exllama": True,
                "disable_exllamav2": False,
            },
        ).state_dict()
        torch.set_default_dtype(orig_dtype)
        return (sd, request.param)

    def test_fusion_no_ckpt(self, get_state_dict):
        # validate fused/unfused output after instantiating FP16 model without ckpt
        sd, strategy = get_state_dict
        fusion = {True: "fused", False: "unfused"}
        expected_layers = expected_layers_from_fusion[fusion[strategy]]
        assert all(
            [
                any([layer_key in sd_key for sd_key in sd])
                for layer_key in expected_layers
            ]
        )

    def test_fused_weights_none_from_ckpt(self, get_state_dict):
        # reload unfused or fused state dict from file
        # fused_weights=None => always expect fused output model
        sd, _ = get_state_dict
        expected_layers = expected_layers_from_fusion["fused"]
        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            torch.save(sd, f.name)
            sd_fused = get_model(
                architecture="llama",
                variant="micro",
                model_path=f.name,
                source="hf",
            ).state_dict()
            assert all(
                [
                    any([layer_key in sd_key for sd_key in sd_fused])
                    for layer_key in expected_layers
                ]
            )

    def test_fused_weights_false_from_ckpt(self, get_state_dict):
        # reload unfused or fused state dict from file
        # fused_weights=False => always expect unfused output model
        sd, _ = get_state_dict
        expected_layers = expected_layers_from_fusion["unfused"]
        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            torch.save(sd, f.name)
            sd_fused = get_model(
                architecture="llama",
                variant="micro",
                model_path=f.name,
                fused_weights=False,
            ).state_dict()
            assert all(
                [
                    any([layer_key in sd_key for sd_key in sd_fused])
                    for layer_key in expected_layers
                ]
            )

    @pytest.mark.autogptq
    def test_gptq_fusion_no_ckpt(self, get_gptq_state_dict):
        # validate fused/unfused output after instantiating GPTQ model without ckpt
        sd, strategy = get_gptq_state_dict
        fusion = "fused" if strategy == True else "unfused"
        expected_layers = expected_layers_from_fusion[fusion]
        assert all(
            [
                any([layer_key in sd_key for sd_key in sd])
                for layer_key in expected_layers
            ]
        )
