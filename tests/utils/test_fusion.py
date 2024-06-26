import pytest
import torch.testing

from fms.modules.attention import FusedQKV, MultiHeadAttention, UnfusedQKV
from fms.utils.fusion import apply_fusion


def test_unfuse():
    fused_module = MultiHeadAttention(16, 16, 16, 8, 8, use_bias=True, fused=True)
    fused_module_weight = fused_module.in_proj.qkv_fused.weight.data
    assert isinstance(fused_module.in_proj, FusedQKV)
    unfused_module = apply_fusion(fused_module, fuse=False)
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


def test_fuse():
    unfused_module = MultiHeadAttention(16, 16, 16, 8, 8, use_bias=True, fused=False)
    unfused_module_weight = torch.cat(
        (
            unfused_module.in_proj.query.weight.data,
            unfused_module.in_proj.key.weight.data,
            unfused_module.in_proj.value.weight.data,
        ),
        dim=0,
    )
    assert isinstance(unfused_module.in_proj, UnfusedQKV)
    fused_module = apply_fusion(unfused_module, fuse=True)
    assert isinstance(fused_module.in_proj, FusedQKV)

    fused_module_weight = fused_module.in_proj.qkv_fused.weight.data
    assert fused_module_weight.tolist() == unfused_module_weight.tolist()
