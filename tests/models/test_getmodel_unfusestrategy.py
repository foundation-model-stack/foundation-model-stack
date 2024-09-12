import tempfile

import pytest
import torch

from fms.models import get_model


# How to set unfuse_strategy to obtain desired `target_model` fusion
#
# ckpt       target_model   unfuse_strategy   tested
#                            FP16    GPTQ
# --------------------------------------------------
# none       fused           None    None     (Y, Y)
# none       unfused         post    pre      (Y, Y)
# fused      fused           None    None     (Y, N)
# fused      unfused         post    n/a      (Y, N)
# unfused    fused           None    n/a      (Y, N)
# unfused    unfused         post    pre      (Y, N)


# FP16 model
unfuse_strategies = [None, "post"]
unfuse_ids = ["unfuse=None", "unfuse=post"]
# gptq model
unfuse_strategies_gptq = [None, "pre"]
unfuse_ids_gptq = ["unfuse=None", "unfuse=pre"]

expected_layers_from_fusion = {
    "fused": [".qkv_fused.", ".wg1_fused."],
    "unfused": [".query.", ".key.", ".value.", ".w1.", ".wg."],
}


class TestUnfuseStrategy:
    @pytest.fixture(
        scope="class",
        params=unfuse_strategies,
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
            unfuse_strategy=request.param,
            linear_config={"linear_type": "torch_linear"},  # same as None
        ).state_dict()
        torch.set_default_dtype(orig_dtype)
        return (sd, request.param)

    @pytest.fixture(
        scope="class",
        params=unfuse_strategies_gptq,
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
            unfuse_strategy=request.param,
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
        fusion = {None: "fused", "post": "unfused"}
        expected_layers = expected_layers_from_fusion[fusion[strategy]]
        assert all(
            [
                any([layer_key in sd_key for sd_key in sd])
                for layer_key in expected_layers
            ]
        )

    def test_strategy_none_from_ckpt(self, get_state_dict):
        # reload unfused or fused state dict from file
        # unfuse_strategy=None => always expect fused output model
        sd, _ = get_state_dict
        expected_layers = expected_layers_from_fusion["fused"]
        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            torch.save(sd, f.name)
            sd_fused = get_model(
                architecture="llama",
                variant="micro",
                model_path=f.name,
                source="hf",
                unfuse_strategy=None,
            ).state_dict()
            assert all(
                [
                    any([layer_key in sd_key for sd_key in sd_fused])
                    for layer_key in expected_layers
                ]
            )

    def test_strategy_post_from_ckpt(self, get_state_dict):
        # reload unfused or fused state dict from file
        # unfuse_strategy="post" => always expect unfused output model
        sd, _ = get_state_dict
        expected_layers = expected_layers_from_fusion["unfused"]
        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            torch.save(sd, f.name)
            sd_fused = get_model(
                architecture="llama",
                variant="micro",
                model_path=f.name,
                unfuse_strategy="post",
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
        fusion = "fused" if strategy == None else "unfused"
        expected_layers = expected_layers_from_fusion[fusion]
        assert all(
            [
                any([layer_key in sd_key for sd_key in sd])
                for layer_key in expected_layers
            ]
        )
