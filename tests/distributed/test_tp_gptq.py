import pytest
import torch
import torch.distributed
import torch.nn

from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import FeedForwardBlock
from fms.utils.config import ModelConfig


class MockGPTQConfig(ModelConfig):
    emb_dim: int = 4096
    emb_kq: int = 128
    emb_v: int = 128
    nheads: int = 32
    kvheads: int = 8
    use_bias: bool = False  # AutoGPTQ QuantLinear has bias zero if False (not None)
    fused: bool = True
    group_size: int = 2


class MockGroup:
    def __init__(self, world_size) -> None:
        self.world_size = world_size
        self.current_rank = 0

    def size(self):
        return self.world_size

    def rank(self):
        self.current_rank += 1
        return self.current_rank - 1


class TestGPTQwithTP:

    @pytest.fixture(scope="class")
    def get_config(self) -> MockGPTQConfig:
        # defined as fixture to support future parameterization
        return MockGPTQConfig()

    @pytest.fixture(scope="class")
    def get_attention(self, get_config) -> MultiHeadAttention:
        config = get_config
        attention = MultiHeadAttention(
            config.emb_dim,
            config.emb_kq,
            config.emb_v,
            config.nheads,
            config.kvheads,
            config.use_bias,
            config.fused,
            linear_config={
                "linear_type": "gptq",
                "group_size": config.group_size,
                "use_marlin": False,
                "disable_exllama": True,
                "disable_exllamav2": False,
            }
        )
        return attention

    def test_gptq_tp_fused(self, get_attention, get_config):
        # Unused, just for asserts
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "gloo", store=torch.distributed.HashStore(), rank=0, world_size=1
            )

        # params to test for GPTQ:
        # qweight [in_feat / 8, out_feat]
        # bias    [out_feat]
        # scales  [n_grp, out_feat]
        # qzeros  [n_grp, out_feat / 8]
        # g_idx   [in_feat]
        config = get_config
        packing = 8  # 8x INT4 --> 1x INT32
        int32_max = torch.iinfo(torch.int32).max

        # QKV
        in_feat = config.emb_dim
        out_feat = (
            config.emb_kq * config.nheads
            + config.emb_kq * config.kvheads
            + config.emb_v * config.kvheads
        )
        n_grp = in_feat // config.group_size

        qkv_fused = {
            "qweight": torch.randint(
                0, int32_max, (in_feat // packing, out_feat)
            ).to(torch.int32),
            "bias": torch.zeros((out_feat, ), dtype=torch.float16),
            "scales": torch.randn((n_grp, out_feat), dtype=torch.float16),
            "qzeros": torch.randint(
                0, int32_max, (n_grp, out_feat // packing)
            ).to(torch.int32),
            "g_idx": torch.randint(0, n_grp, (in_feat, )).to(torch.int32),
        }

        # DENSE
        in_feat = config.emb_kq * config.nheads
        out_feat = config.emb_dim
        n_grp = in_feat // config.group_size

        dense = {
            "qweight": torch.randint(
                0, int32_max, (in_feat // packing, out_feat)
            ).to(torch.int32),
            "bias": torch.zeros((out_feat, ), dtype=torch.float16),
            "scales": torch.randn((n_grp, out_feat), dtype=torch.float16),
            "qzeros": torch.randint(
                0, int32_max, (n_grp, out_feat // packing)
            ).to(torch.int32),
            "g_idx": torch.randint(0, n_grp, (in_feat, )).to(torch.int32),
        }

        def _test_gptq_for_world_size(group, qkv_fused, dense):
            for rank in range(group.size()):
                print(f"Rank {rank}")
                qparams = {}
                for k, v in qkv_fused.items():
                    qparams["in_proj.qkv_fused." + k] = v
                for k, v in dense.items():
                    qparams["dense." + k] = v

                with torch.no_grad():
                    # create TP-sharded attention module
                    tp_mod = get_attention.to_tp(group)

                    print("=== TP MODULE ==========================================")
                    print("\n".join(f"{k:40} {v.size()}" for k, v in tp_mod.named_buffers()))
                    print("=== CKPT ===============================================")
                    print("\n".join(f"{k:40} {v.size()}" for k, v in qparams.items()))
                    print("")

                    tp_mod.load_weights(qparams)

                # load split qparams from TP attention module
                tp_q_qw, tp_k_qw, tp_v_qw = torch.split(
                    tp_mod.in_proj.qkv_fused.qweight, tp_mod.in_proj.splits, dim=1
                )
                tp_q_bias, tp_k_bias, tp_v_bias = torch.split(
                    tp_mod.in_proj.qkv_fused.bias, tp_mod.in_proj.splits, dim=0
                )
                tp_q_scales, tp_k_scales, tp_v_scales = torch.split(
                    tp_mod.in_proj.qkv_fused.scales, tp_mod.in_proj.splits, dim=1
                )
                tp_q_qzeros, tp_k_qzeros, tp_v_qzeros = torch.split(
                    tp_mod.in_proj.qkv_fused.qzeros, [x // packing for x in tp_mod.in_proj.splits], dim=1
                )
                tp_gidx = tp_mod.in_proj.qkv_fused.g_idx

                # load pre-TP tensors
                qkv_fused_qw = qparams["in_proj.qkv_fused.qweight"]
                qkv_fused_bias = qparams["in_proj.qkv_fused.bias"]
                qkv_fused_scales = qparams["in_proj.qkv_fused.scales"]
                qkv_fused_qzeros = qparams["in_proj.qkv_fused.qzeros"]
                qkv_fused_gidx = qparams["in_proj.qkv_fused.g_idx"]

                # assert on Query linear module
                q_out_dim = config.emb_kq * config.nheads
                q_mult = q_out_dim // min(group.size(), config.nheads)
                q_mult_packed = q_mult // packing
                q_rank = rank // max(1, group.size() // config.nheads)
                torch.testing.assert_close(
                    qkv_fused_qw[:, q_mult * q_rank : q_mult * (q_rank + 1)],
                    tp_q_qw
                )
                torch.testing.assert_close(
                    qkv_fused_bias[q_mult * q_rank : q_mult * (q_rank + 1)],
                    tp_q_bias
                )
                torch.testing.assert_close(
                    qkv_fused_scales[:, q_mult * q_rank : q_mult * (q_rank + 1)],
                    tp_q_scales
                )
                torch.testing.assert_close(
                    qkv_fused_qzeros[:, q_mult_packed * q_rank : q_mult_packed * (q_rank + 1)],
                    tp_q_qzeros
                )
                torch.testing.assert_close(
                    qkv_fused_gidx - min(qkv_fused_gidx),
                    tp_gidx
                )

                # assert on Key linear module
                k_out_dim = config.emb_kq * config.kvheads
                k_mult = k_out_dim // min(group.size(), config.kvheads)
                k_rank = rank // max(1, group.size() // config.kvheads)
                k_out_dim_start = k_mult * k_rank + q_out_dim
                k_out_dim_end = k_mult * (k_rank + 1) + q_out_dim
                torch.testing.assert_close(
                    qkv_fused_qw[:, k_out_dim_start : k_out_dim_end],
                    tp_k_qw
                )
                torch.testing.assert_close(
                    qkv_fused_bias[k_out_dim_start : k_out_dim_end],
                    tp_k_bias
                )
                torch.testing.assert_close(
                    qkv_fused_scales[:, k_out_dim_start : k_out_dim_end],
                    tp_k_scales
                )
                torch.testing.assert_close(
                    qkv_fused_qzeros[:, k_out_dim_start // packing : k_out_dim_end // packing],
                    tp_k_qzeros
                )

                # assert on Value linear module
                v_mult = config.emb_v * config.kvheads // min(group.size(), config.kvheads)
                v_rank = rank // max(1, group.size() // config.kvheads)
                v_out_idx_start = v_mult * v_rank + q_out_dim + k_out_dim
                v_out_idx_end = v_mult * (v_rank + 1) + q_out_dim + k_out_dim
                torch.testing.assert_close(
                    qkv_fused_qw[:, v_out_idx_start : v_out_idx_end],
                    tp_v_qw
                )
                torch.testing.assert_close(
                    qkv_fused_bias[v_out_idx_start : v_out_idx_end],
                    tp_v_bias
                )
                torch.testing.assert_close(
                    qkv_fused_scales[:, v_out_idx_start : v_out_idx_end],
                    tp_v_scales
                )
                torch.testing.assert_close(
                    qkv_fused_qzeros[:, v_out_idx_start // packing : v_out_idx_end // packing],
                    tp_v_qzeros
                )

                # assert on Dense linear module
                d_qw = qparams["dense.qweight"]
                d_bias = qparams["dense.bias"]
                d_scales = qparams["dense.scales"]
                d_qzeros = qparams["dense.qzeros"]
                d_gidx = qparams["dense.g_idx"]
                d_mult = config.emb_dim // min(group.size(), config.nheads)
                d_mult_packed = d_mult // packing
                d_mult_grp = d_mult // config.group_size
                d_rank = rank // max(1, group.size() // config.nheads)
                torch.testing.assert_close(
                    d_qw[d_mult_packed * d_rank : d_mult_packed * (d_rank + 1)],
                    tp_mod.dense.qweight,
                )
                torch.testing.assert_close(
                    d_bias,
                    tp_mod.dense.bias,
                )
                torch.testing.assert_close(
                    d_scales[d_mult_grp * d_rank : d_mult_grp * (d_rank + 1)],
                    tp_mod.dense.scales,
                )
                torch.testing.assert_close(
                    d_qzeros[d_mult_grp * d_rank : d_mult_grp * (d_rank + 1)],
                    tp_mod.dense.qzeros,
                )
                d_gidx_unscaled = d_gidx[d_mult * d_rank : d_mult * (d_rank + 1)]
                torch.testing.assert_close(
                    d_gidx_unscaled - min(d_gidx_unscaled),
                    tp_mod.dense.g_idx,
                )

        # Test world_size < kvheads
        group = MockGroup(4)
        _test_gptq_for_world_size(group, qkv_fused, dense)

        # Test kvheads <= world_size < nheads
        group = MockGroup(8)
        _test_gptq_for_world_size(group, qkv_fused, dense)

        group = MockGroup(16)
        _test_gptq_for_world_size(group, qkv_fused, dense)

        # Test nheads <= world_size
        group = MockGroup(32)
        _test_gptq_for_world_size(group, qkv_fused, dense)
