import pytest
import torch
import torch.distributed
import torch.nn

from fms.modules.attention import MultiHeadAttention, TPMultiHeadAttention
from fms.modules.feedforward import FeedForwardBlock


class MockGroup:
    def __init__(self, world_size) -> None:
        self.world_size = world_size
        self.current_rank = 0

    def size(self):
        return self.world_size

    def rank(self):
        self.current_rank += 1
        return self.current_rank - 1


def test_attention_tp_fused():
    # Unused, just for asserts
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "gloo", store=torch.distributed.HashStore(), rank=0, world_size=1
        )
    attention = MultiHeadAttention(
        emb_dim=4096, emb_kq=128, emb_v=128, nheads=32, kvheads=8, fused=True
    )
    q_weight = torch.randn((128 * 32, 4096))
    k_weight = torch.randn((128 * 8, 4096))
    v_weight = torch.randn((128 * 8, 4096))
    qkv_fused_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
    d_weight = torch.randn((4096, 128 * 32))

    def _test_for_world_size(group, q_weight, k_weight, v_weight, d_weight):
        for rank in range(group.size()):
            with torch.no_grad():
                tp_mod = attention.to_tp(group)
                tp_mod.load_weights(
                    {
                        "in_proj.qkv_fused.weight": qkv_fused_weight,
                        "dense.weight": d_weight,
                    }
                )
            q_mult = 4096 // min(group.size(), 32)
            q_rank = rank // max(1, group.size() // 32)

            tp_q, tp_k, tp_v = torch.split(
                tp_mod.in_proj.qkv_fused.weight, tp_mod.in_proj.splits
            )

            torch.testing.assert_close(
                q_weight[q_mult * q_rank : q_mult * (q_rank + 1)], tp_q
            )
            k_mult = 128 * 8 // min(group.size(), 8)
            k_rank = rank // max(1, group.size() // 8)
            torch.testing.assert_close(
                k_weight[k_mult * k_rank : k_mult * (k_rank + 1)], tp_k
            )
            v_mult = 128 * 8 // min(group.size(), 8)
            v_rank = rank // max(1, group.size() // 8)
            torch.testing.assert_close(
                v_weight[v_mult * v_rank : v_mult * (v_rank + 1)], tp_v
            )
            d_mult = 4096 // min(group.size(), 32)
            d_rank = rank // max(1, group.size() // 32)
            torch.testing.assert_close(
                d_weight[:, d_mult * d_rank : d_mult * (d_rank + 1)],
                tp_mod.dense.weight,
            )

    # Test world_size < kvheads
    group = MockGroup(4)
    _test_for_world_size(group, q_weight, k_weight, v_weight, d_weight)

    # Test kvheads <= world_size < nheads
    group = MockGroup(8)
    _test_for_world_size(group, q_weight, k_weight, v_weight, d_weight)

    group = MockGroup(16)
    _test_for_world_size(group, q_weight, k_weight, v_weight, d_weight)

    # Test nheads <= world_size
    group = MockGroup(32)
    _test_for_world_size(group, q_weight, k_weight, v_weight, d_weight)

    with pytest.raises(AssertionError):
        group = MockGroup(64)
        _test_for_world_size(group, q_weight, k_weight, v_weight, d_weight)


def test_attention_tp_unfused():
    # Unused, just for asserts
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "gloo", store=torch.distributed.HashStore(), rank=0, world_size=1
        )
    attention = MultiHeadAttention(
        emb_dim=4096, emb_kq=128, emb_v=128, nheads=32, kvheads=8, fused=False
    )
    q_weight = torch.randn((128 * 32, 4096))
    k_weight = torch.randn((128 * 8, 4096))
    v_weight = torch.randn((128 * 8, 4096))
    d_weight = torch.randn((4096, 128 * 32))

    def _test_for_world_size(group, q_weight, k_weight, v_weight, d_weight):
        for rank in range(group.size()):
            with torch.no_grad():
                tp_mod = attention.to_tp(group)
                tp_mod.load_weights(
                    {
                        "in_proj.query.weight": q_weight,
                        "in_proj.key.weight": k_weight,
                        "in_proj.value.weight": v_weight,
                        "dense.weight": d_weight,
                    }
                )
            q_mult = 4096 // min(group.size(), 32)
            q_rank = rank // max(1, group.size() // 32)
            torch.testing.assert_close(
                q_weight[q_mult * q_rank : q_mult * (q_rank + 1)],
                tp_mod.in_proj.query.weight,
            )
            k_mult = 128 * 8 // min(group.size(), 8)
            k_rank = rank // max(1, group.size() // 8)
            torch.testing.assert_close(
                k_weight[k_mult * k_rank : k_mult * (k_rank + 1)],
                tp_mod.in_proj.key.weight,
            )
            v_mult = 128 * 8 // min(group.size(), 8)
            v_rank = rank // max(1, group.size() // 8)
            torch.testing.assert_close(
                v_weight[v_mult * v_rank : v_mult * (v_rank + 1)],
                tp_mod.in_proj.value.weight,
            )
            d_mult = 4096 // min(group.size(), 32)
            d_rank = rank // max(1, group.size() // 32)
            torch.testing.assert_close(
                d_weight[:, d_mult * d_rank : d_mult * (d_rank + 1)],
                tp_mod.dense.weight,
            )

    # Test world_size < kvheads
    group = MockGroup(4)
    _test_for_world_size(group, q_weight, k_weight, v_weight, d_weight)

    # Test kvheads <= world_size < nheads
    group = MockGroup(8)
    _test_for_world_size(group, q_weight, k_weight, v_weight, d_weight)

    group = MockGroup(16)
    _test_for_world_size(group, q_weight, k_weight, v_weight, d_weight)

    # Test nheads <= world_size
    group = MockGroup(32)
    _test_for_world_size(group, q_weight, k_weight, v_weight, d_weight)

    with pytest.raises(AssertionError):
        group = MockGroup(64)
        _test_for_world_size(group, q_weight, k_weight, v_weight, d_weight)


def test_feedforward_tp():
    # Unused, just for asserts
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "gloo", store=torch.distributed.HashStore(), rank=0, world_size=1
        )

    feedforward = FeedForwardBlock(
        emb_dim=4096,
        p_dropout=0.0,
    )
    in_weight = torch.randn((4096 * 4, 4096))
    in_bias = torch.randn((4096 * 4,))
    out_weight = torch.randn((4096, 4 * 4096))
    out_bias = torch.randn((4096,))

    def _test_for_world_size(group, in_weight, in_bias, out_weight, out_bias):
        for rank in range(group.size()):
            with torch.no_grad():
                tp_mod = feedforward.to_tp(group)
                tp_mod.load_weights(
                    {
                        "w1.weight": in_weight,
                        "w1.bias": in_bias,
                        "w2.weight": out_weight,
                        "w2.bias": out_bias,
                    }
                )
            in_mult = 4096 * 4 // group.size()
            torch.testing.assert_close(
                in_weight[in_mult * rank : in_mult * (rank + 1)], tp_mod.w1.weight
            )
            torch.testing.assert_close(
                in_bias[in_mult * rank : in_mult * (rank + 1)], tp_mod.w1.bias
            )
            out_mult = 4096 * 4 // group.size()
            torch.testing.assert_close(
                out_weight[:, out_mult * rank : out_mult * (rank + 1)], tp_mod.w2.weight
            )
            if rank == 0:
                torch.testing.assert_close(out_bias, tp_mod.w2.bias)
            else:
                torch.testing.assert_close(
                    torch.zeros(tp_mod.w2.bias.shape), tp_mod.w2.bias
                )

    # Test world_size < kvheads
    group = MockGroup(4)
    _test_for_world_size(group, in_weight, in_bias, out_weight, out_bias)

    with pytest.raises(AssertionError):
        group = MockGroup(6)
        _test_for_world_size(group, in_weight, in_bias, out_weight, out_bias)
