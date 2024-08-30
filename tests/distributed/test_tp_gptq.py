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
            "bias": torch.randn((out_feat, ), dtype=torch.float16),
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
            "bias": torch.randn((out_feat, ), dtype=torch.float16),
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
                    tp_mod = get_attention.to_tp(group)

                    print("=== TP MODULE ==========================================")
                    print("\n".join(f"{k:40} {v.size()}" for k, v in tp_mod.named_buffers()))
                    print("=== CKPT ===============================================")
                    print("\n".join(f"{k:40} {v.size()}" for k, v in qparams.items()))
                    print("")

                    tp_mod.load_weights(qparams)

                # qweights are transposed
                tp_q, tp_k, tp_v = torch.split(
                    tp_mod.in_proj.qkv_fused.qweight, tp_mod.in_proj.splits, dim=1
                )

                qkv_fused_qw = qparams["in_proj.qkv_fused.qweight"]
                q_mult = config.emb_kq * config.nheads // min(group.size(), config.nheads)
                q_rank = rank // max(1, group.size() // config.nheads)
                torch.testing.assert_close(
                    qkv_fused_qw[:, q_mult * q_rank : q_mult * (q_rank + 1)],
                    tp_q
                )

                k_mult = config.emb_kq * config.kvheads // min(group.size(), config.kvheads)
                k_rank = rank // max(1, group.size() // config.kvheads)
                torch.testing.assert_close(
                    qkv_fused_qw[:, k_mult * k_rank + config.emb_kq * config.nheads : k_mult * (k_rank + 1) +  config.emb_kq * config.nheads],
                    tp_k
                )

                v_mult = config.emb_v * config.kvheads // min(group.size(), config.kvheads)
                v_rank = rank // max(1, group.size() // config.kvheads)
                torch.testing.assert_close(
                    qkv_fused_qw[:, v_mult * v_rank + config.emb_kq * config.nheads + config.emb_kq * config.kvheads : v_mult * (v_rank + 1) + config.emb_kq * config.nheads + config.emb_kq * config.kvheads],
                    tp_v
                )

                d_qw = qparams["dense.qweight"]
                d_mult = config.emb_dim // packing // min(group.size(), config.nheads)
                d_rank = rank // max(1, group.size() // config.nheads)
                torch.testing.assert_close(
                    d_qw[d_mult * d_rank : d_mult * (d_rank + 1)],
                    tp_mod.dense.qweight,
                )

        # Test world_size < kvheads
        group = MockGroup(2)
        _test_gptq_for_world_size(group, qkv_fused, dense)

        # Test kvheads <= world_size < nheads
        # group = MockGroup(8)
        # _test_gptq_for_world_size(group, qkv_fused, dense)

        assert True
