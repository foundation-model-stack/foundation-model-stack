import pytest
import torch

from fms.models import get_model


try:
    from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import (
        QuantLinear as qlinear_cuda_old,
    )
    from auto_gptq.nn_modules.qlinear.qlinear_exllama import (
        QuantLinear as qlinear_exllama,
    )
    from auto_gptq.nn_modules.qlinear.qlinear_exllamav2 import (
        QuantLinear as qlinear_exllamav2,
    )
    from auto_gptq.nn_modules.qlinear.qlinear_marlin import (
        QuantLinear as qlinear_marlin,
    )
except ImportError:
    print(
        "One or more AutoGPTQ QuantLinear (cuda_old, exllama, exllamav2, marlin) "
        "could not be imported"
    )

# TODO: support for marlin kernels to be implemented

qlinear_configs = [
    (
        "cuda",
        {
            "linear_type": "gptq",
            "group_size": 2,
            "use_marlin": False,
            "disable_exllama": True,
            "disable_exllamav2": True,
        },
    ),
    (
        "exllama",
        {
            "linear_type": "gptq",
            "group_size": 2,
            "use_marlin": False,
            "disable_exllama": False,
            "disable_exllamav2": True,
        },
    ),
    (
        "exllamav2",
        {
            "linear_type": "gptq",
            "group_size": 2,
            "use_marlin": False,
            "disable_exllama": True,
            "disable_exllamav2": False,
        },
    ),
    # (
    #     "marlin",
    #     {
    #         "linear_type": "gptq",
    #         "group_size": 2,
    #         "use_marlin": True,
    #         "disable_exllama": True,
    #         "disable_exllamav2": True,
    #     }
    # ),
]
qlinear_ids = ["cuda", "exllama", "exllamav2"]  # , "marlin"]


class TestGPTQModel:
    @pytest.fixture(
        scope="class",
        params=qlinear_configs,
        ids=qlinear_ids,
    )
    def get_gptq_model(self, request):
        id, linear_config = request.param

        # instantiate GPTQ model
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        gptq_model = get_model(
            architecture="llama",
            variant="micro",
            model_path=None,
            source="hf",
            linear_config=linear_config,
        )
        torch.set_default_dtype(orig_dtype)
        return (id, gptq_model)

    @pytest.mark.autogptq
    def test_gptq_quantlinear(self, get_gptq_model):
        # verify that all fused linear modules in GPTQ model are instances
        # of a QuantLinear of the expected type (cuda_old, exllama, exllamav2)
        qlinear_id_to_module = {
            "cuda": qlinear_cuda_old,
            "exllama": qlinear_exllama,
            "exllamav2": qlinear_exllamav2,
            "marlin": qlinear_marlin,
        }
        id, gptq_model = get_gptq_model
        fused_linear = ["qkv_fused", "dense", "wg1_fused", "w2"]
        not_quantlinear = {}
        for k, v in gptq_model.named_modules():
            if k.split(".")[-1] in fused_linear and not isinstance(
                v, qlinear_id_to_module[id]
            ):
                not_quantlinear[k] = v
        assert not_quantlinear == {}
