import tempfile

import pytest
import torch

from fms.models import get_model


# Expected data_type of model returned by get_model
#
# model_path    data_type      output model dtype
# ================================================
# None          None           default
# path/to/sd    None           state dict
# path/to/sd    custom         custom


class TestDatatype:
    @pytest.fixture(
        scope="class",
        params=[torch.float32, torch.float16],
        ids=["input=fp32_sd", "input=fp16_sd"],
    )
    def get_state_dict(self, request):
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(request.param)
        sd = get_model(
            architecture="llama",
            variant="micro",
            model_path=None,
        ).state_dict()
        torch.set_default_dtype(orig_dtype)
        return (sd, request.param)

    def test_datatype_default(self, get_state_dict):
        # model_path: None
        # data type: None (torch default)
        # expected output dtype: default (FP32 or FP16)
        sd, model_dtype = get_state_dict
        assert all([v.dtype == model_dtype for v in sd.values()])

    def test_datatype_as_sd(self, get_state_dict):
        # model_path: sd (FP32 or FP16)
        # data type: None
        # expected output dtype: as sd (FP32 or FP16)
        sd, model_dtype = get_state_dict
        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            torch.save(sd, f.name)
            sd_from_file = get_model(
                architecture="llama",
                variant="micro",
                model_path=f.name,
                data_type=None,
            ).state_dict()
            assert all([v.dtype == model_dtype for v in sd_from_file.values()])

    def test_datatype_force_fp16(self, get_state_dict):
        # model_path: sd (FP32 or FP16)
        # data_type: FP16
        # expected output dtype: FP16
        sd, _ = get_state_dict
        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            torch.save(sd, f.name)
            sd_from_file = get_model(
                architecture="llama",
                variant="micro",
                model_path=f.name,
                data_type="float16",
            ).state_dict()
            assert all([v.dtype == torch.float16 for v in sd_from_file.values()])

    def test_datatype_force_fp32(self, get_state_dict):
        # model_path: sd (FP32 or FP16)
        # data_type: FP32
        # expected output dtype: FP32
        sd, _ = get_state_dict
        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
            torch.save(sd, f.name)
            sd_from_file = get_model(
                architecture="llama",
                variant="micro",
                model_path=f.name,
                data_type="float32",
            ).state_dict()
            assert all([v.dtype == torch.float32 for v in sd_from_file.values()])
