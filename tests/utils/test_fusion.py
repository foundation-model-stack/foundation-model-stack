import pytest
import torch.testing

from fms.modules.attention import FusedQKV, MultiHeadAttention, UnfusedQKV
from fms.modules.feedforward import GatedLinearUnit
from fms.utils.fusion import apply_unfuse_weights


def test_unfuse_qkv():
    fused_module = MultiHeadAttention(16, 16, 16, 8, 8, use_bias=True, fused=True)
    fused_module_weight = fused_module.in_proj.qkv_fused.weight.data
    assert isinstance(fused_module.in_proj, FusedQKV)
    unfused_module = apply_unfuse_weights(fused_module)
    assert isinstance(unfused_module.in_proj, UnfusedQKV)

    unfused_module_weight = torch.cat(
        (
            unfused_module.in_proj.query.weight.data,
            unfused_module.in_proj.key.weight.data,
            unfused_module.in_proj.value.weight.data,
        ),
        dim=0,
    )
    assert fused_module_weight.tolist() == unfused_module_weight.tolist()


def test_unfuse_glu():
    glu_module = GatedLinearUnit(16, 4, use_bias=True, fused=True)
    assert hasattr(glu_module, "wg1_fused")
    fused_module_weight = glu_module.wg1_fused.weight.data
    unfused_module = apply_unfuse_weights(glu_module)
    assert hasattr(unfused_module, "wg")
    assert hasattr(unfused_module, "w1")

    unfused_module_weight = torch.cat(
        (
            unfused_module.wg.weight.data,
            unfused_module.w1.weight.data,
        ),
        dim=0,
    )
    assert fused_module_weight.tolist() == unfused_module_weight.tolist()
