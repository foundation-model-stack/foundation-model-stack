import pytest
import torch
import torch.distributed
import torch.nn

from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import FeedForwardBlock, GatedLinearUnit
from fms.utils.config import ModelConfig


class MockGPTQConfig(ModelConfig):
    emb_dim: int = 4096
    emb_kq: int = 128
    emb_v: int = 128
    nheads: int = 32
    kvheads: int = 8
    multiple_of: int = 4
    use_bias: bool = False
    fused: bool = True  # only testing fused=True
    group_size: int = 2
    use_marlin: bool = False  # Marlin kernels not supported yet
    disable_exllama: bool = True
    disable_exllamav2: bool = False


class MockGroup:
    def __init__(self, world_size) -> None:
        self.world_size = world_size
        self.current_rank = 0

    def size(self):
        return self.world_size

    def rank(self):
        self.current_rank += 1
        return self.current_rank - 1

    def reset_rank(self):
        self.current_rank = 0


class TestGPTQwithTP:
    @pytest.fixture(scope="class")
    def get_config(self) -> MockGPTQConfig:
        # defined as fixture to support future parameterization
        return MockGPTQConfig()

    @pytest.fixture(
        scope="class",
        params=[4, 8, 16, 32],
        ids=["TP=4", "TP=8", "TP=16", "TP=32"],
    )
    def get_mockgroup(self, request) -> MockGroup:
        return MockGroup(request.param)

    @pytest.fixture(scope="class")
    def get_attention(self, get_config) -> MultiHeadAttention:
        config = get_config
        assert config.fused  # only testing fused=True
        attention = MultiHeadAttention(
            emb_dim=config.emb_dim,
            emb_kq=config.emb_kq,
            emb_v=config.emb_v,
            nheads=config.nheads,
            kvheads=config.kvheads,
            use_bias=config.use_bias,
            fused=config.fused,
            linear_config={
                "linear_type": "gptq",
                "group_size": config.group_size,
                "use_marlin": config.use_marlin,
                "disable_exllama": config.disable_exllama,
                "disable_exllamav2": config.disable_exllamav2,
            },
        )
        return attention

    @pytest.fixture(scope="class")
    def get_ffn(self, get_config) -> FeedForwardBlock:
        config = get_config
        # ffn does not have fusion
        ffn = FeedForwardBlock(
            emb_dim=config.emb_dim,
            use_bias=config.use_bias,
            linear_config={
                "linear_type": "gptq",
                "group_size": config.group_size,
                "use_marlin": config.use_marlin,
                "disable_exllama": config.disable_exllama,
                "disable_exllamav2": config.disable_exllamav2,
            },
        )
        return ffn

    @pytest.fixture(scope="class")
    def get_glu(self, get_config) -> GatedLinearUnit:
        config = get_config
        assert config.fused  # only testing fused=True
        glu = GatedLinearUnit(
            emb_dim=config.emb_dim,
            use_bias=config.use_bias,
            fused=config.fused,
            linear_config={
                "linear_type": "gptq",
                "group_size": config.group_size,
                "use_marlin": config.use_marlin,
                "disable_exllama": config.disable_exllama,
                "disable_exllamav2": config.disable_exllamav2,
            },
        )
        return glu

    @pytest.mark.autogptq
    def test_gptq_kernel(self, get_config, get_attention, get_ffn, get_glu):
        config = get_config
        if config.use_marlin:
            raise NotImplementedError("Marlin kernel not supported at this time")
        if config.disable_exllama:
            if config.disable_exllamav2:
                kernel = "cuda-old"
            else:
                kernel = "exllamav2"
        elif config.disable_exllamav2:
            kernel = "exllama"
        else:
            raise ValueError(
                "ambiguous configuration: disable_exllama and disable_exllamav2 are both True in MockGPTQConfig"
            )

        assert get_attention.in_proj.qkv_fused.QUANT_TYPE == kernel
        assert get_attention.dense.QUANT_TYPE == kernel
        assert get_ffn.w1.QUANT_TYPE == kernel
        assert get_ffn.w2.QUANT_TYPE == kernel
        assert get_glu.wg1_fused.QUANT_TYPE == kernel
        assert get_glu.w2.QUANT_TYPE == kernel

    @pytest.mark.autogptq
    def test_gptq_tp_attn_fused(self, get_attention, get_config, get_mockgroup):
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

        qkv_fused_qparam = {
            "qweight": torch.randint(0, int32_max, (in_feat // packing, out_feat)).to(
                torch.int32
            ),
            "scales": torch.randn((n_grp, out_feat), dtype=torch.float16),
            "qzeros": torch.randint(0, int32_max, (n_grp, out_feat // packing)).to(
                torch.int32
            ),
            "g_idx": torch.randint(0, n_grp, (in_feat,)).to(torch.int32),
        }
        if config.use_bias:
            qkv_fused_qparam["bias"] = torch.zeros((out_feat,), dtype=torch.float16)

        # DENSE
        in_feat = config.emb_kq * config.nheads
        out_feat = config.emb_dim
        n_grp = in_feat // config.group_size

        dense_qparam = {
            "qweight": torch.randint(0, int32_max, (in_feat // packing, out_feat)).to(
                torch.int32
            ),
            "scales": torch.randn((n_grp, out_feat), dtype=torch.float16),
            "qzeros": torch.randint(0, int32_max, (n_grp, out_feat // packing)).to(
                torch.int32
            ),
            "g_idx": torch.randint(0, n_grp, (in_feat,)).to(torch.int32),
        }
        if config.use_bias:
            dense_qparam["bias"] = torch.zeros((out_feat,), dtype=torch.float16)

        def _test_gptq_for_world_size(group, qkv_fused_qparam, dense_qparam):
            for rank in range(group.size()):
                qparams = {}
                for k, v in qkv_fused_qparam.items():
                    qparams["in_proj.qkv_fused." + k] = v
                for k, v in dense_qparam.items():
                    qparams["dense." + k] = v

                with torch.no_grad():
                    # create TP-sharded attention module
                    tp_mod = get_attention.to_tp(group)
                    tp_mod.load_weights(qparams)

                # load split qparams from TP attention module
                tp_q_qw, tp_k_qw, tp_v_qw = torch.split(
                    tp_mod.in_proj.qkv_fused.qweight, tp_mod.in_proj.splits, dim=1
                )
                tp_q_scales, tp_k_scales, tp_v_scales = torch.split(
                    tp_mod.in_proj.qkv_fused.scales, tp_mod.in_proj.splits, dim=1
                )
                tp_q_qzeros, tp_k_qzeros, tp_v_qzeros = torch.split(
                    tp_mod.in_proj.qkv_fused.qzeros,
                    [x // packing for x in tp_mod.in_proj.splits],
                    dim=1,
                )
                tp_gidx = tp_mod.in_proj.qkv_fused.g_idx
                if config.use_bias:
                    tp_q_bias, tp_k_bias, tp_v_bias = torch.split(
                        tp_mod.in_proj.qkv_fused.bias, tp_mod.in_proj.splits, dim=0
                    )

                # load pre-TP tensors of qkv_fused linear module
                qkv_fused_qw = qparams["in_proj.qkv_fused.qweight"]
                qkv_fused_scales = qparams["in_proj.qkv_fused.scales"]
                qkv_fused_qzeros = qparams["in_proj.qkv_fused.qzeros"]
                qkv_fused_gidx = qparams["in_proj.qkv_fused.g_idx"]
                if config.use_bias:
                    qkv_fused_bias = qparams["in_proj.qkv_fused.bias"]

                # assert on Query linear module
                q_out_dim = config.emb_kq * config.nheads
                q_mult = q_out_dim // min(group.size(), config.nheads)
                q_mult_packed = q_mult // packing
                q_rank = rank // max(1, group.size() // config.nheads)
                torch.testing.assert_close(
                    qkv_fused_qw[:, q_mult * q_rank : q_mult * (q_rank + 1)], tp_q_qw
                )
                torch.testing.assert_close(
                    qkv_fused_scales[:, q_mult * q_rank : q_mult * (q_rank + 1)],
                    tp_q_scales,
                )
                torch.testing.assert_close(
                    qkv_fused_qzeros[
                        :, q_mult_packed * q_rank : q_mult_packed * (q_rank + 1)
                    ],
                    tp_q_qzeros,
                )
                torch.testing.assert_close(
                    qkv_fused_gidx - min(qkv_fused_gidx), tp_gidx
                )
                if config.use_bias:
                    torch.testing.assert_close(
                        qkv_fused_bias[q_mult * q_rank : q_mult * (q_rank + 1)],
                        tp_q_bias,
                    )

                # assert on Key linear module
                k_out_dim = config.emb_kq * config.kvheads
                k_mult = k_out_dim // min(group.size(), config.kvheads)
                k_rank = rank // max(1, group.size() // config.kvheads)
                k_out_dim_start = k_mult * k_rank + q_out_dim
                k_out_dim_end = k_mult * (k_rank + 1) + q_out_dim
                torch.testing.assert_close(
                    qkv_fused_qw[:, k_out_dim_start:k_out_dim_end], tp_k_qw
                )
                torch.testing.assert_close(
                    qkv_fused_scales[:, k_out_dim_start:k_out_dim_end], tp_k_scales
                )
                torch.testing.assert_close(
                    qkv_fused_qzeros[
                        :, k_out_dim_start // packing : k_out_dim_end // packing
                    ],
                    tp_k_qzeros,
                )
                if config.use_bias:
                    torch.testing.assert_close(
                        qkv_fused_bias[k_out_dim_start:k_out_dim_end], tp_k_bias
                    )

                # assert on Value linear module
                v_mult = (
                    config.emb_v * config.kvheads // min(group.size(), config.kvheads)
                )
                v_rank = rank // max(1, group.size() // config.kvheads)
                v_out_idx_start = v_mult * v_rank + q_out_dim + k_out_dim
                v_out_idx_end = v_mult * (v_rank + 1) + q_out_dim + k_out_dim
                torch.testing.assert_close(
                    qkv_fused_qw[:, v_out_idx_start:v_out_idx_end], tp_v_qw
                )
                torch.testing.assert_close(
                    qkv_fused_scales[:, v_out_idx_start:v_out_idx_end], tp_v_scales
                )
                torch.testing.assert_close(
                    qkv_fused_qzeros[
                        :, v_out_idx_start // packing : v_out_idx_end // packing
                    ],
                    tp_v_qzeros,
                )
                if config.use_bias:
                    torch.testing.assert_close(
                        qkv_fused_bias[v_out_idx_start:v_out_idx_end], tp_v_bias
                    )
                else:
                    assert (
                        not hasattr(tp_mod.in_proj.qkv_fused, "bias")
                        or tp_mod.in_proj.qkv_fused.bias is None
                        or torch.sum(tp_mod.in_proj.qkv_fused.bias) == 0
                    )
                    assert (
                        qkv_fused_qparam.get("bias", None) is None
                        or torch.sum(qkv_fused_qparam.get("bias")) == 0
                    )

                # load pre-TP tensors of Dense linear module
                d_qw = qparams["dense.qweight"]
                d_scales = qparams["dense.scales"]
                d_qzeros = qparams["dense.qzeros"]
                d_gidx = qparams["dense.g_idx"]
                if config.use_bias:
                    d_bias = qparams["dense.bias"]

                # assert on Dense linear module
                d_mult = config.emb_dim // min(group.size(), config.nheads)
                d_mult_packed = d_mult // packing
                d_mult_grp = d_mult // config.group_size
                d_rank = rank // max(1, group.size() // config.nheads)
                torch.testing.assert_close(
                    d_qw[d_mult_packed * d_rank : d_mult_packed * (d_rank + 1)],
                    tp_mod.dense.qweight,
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
                if config.use_bias:
                    torch.testing.assert_close(
                        d_bias,
                        tp_mod.dense.bias,
                    )
                else:
                    assert (
                        not hasattr(tp_mod.dense, "bias")
                        or tp_mod.dense.bias is None
                        or torch.sum(tp_mod.dense.bias) == 0
                    )
                    assert (
                        dense_qparam.get("bias", None) is None
                        or torch.sum(dense_qparam.get("bias")) == 0
                    )
            group.reset_rank()

        # TP = 4: test world_size < kvheads
        # TP = 8, 16: test kvheads <= world_size < nheads
        # TP = 32: test nheads <= world_size
        _test_gptq_for_world_size(get_mockgroup, qkv_fused_qparam, dense_qparam)

    @pytest.mark.autogptq
    def test_gptq_tp_ffn(self, get_ffn, get_config, get_mockgroup):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "gloo", store=torch.distributed.HashStore(), rank=0, world_size=1
            )

        config = get_config
        packing = 8  # 8x INT4 --> 1x INT32
        int32_max = torch.iinfo(torch.int32).max

        # w1
        in_feat = config.emb_dim
        out_feat = config.multiple_of * in_feat
        n_grp = in_feat // config.group_size

        w1_qparam = {
            "qweight": torch.randint(0, int32_max, (in_feat // packing, out_feat)).to(
                torch.int32
            ),
            "scales": torch.randn((n_grp, out_feat), dtype=torch.float16),
            "qzeros": torch.randint(0, int32_max, (n_grp, out_feat // packing)).to(
                torch.int32
            ),
            "g_idx": torch.randint(0, n_grp, (in_feat,)).to(torch.int32),
        }
        if config.use_bias:
            w1_qparam["bias"] = torch.zeros((out_feat,), dtype=torch.float16)

        # w2
        in_feat = config.multiple_of * config.emb_dim
        out_feat = config.emb_dim
        n_grp = in_feat // config.group_size

        w2_qparam = {
            "qweight": torch.randint(0, int32_max, (in_feat // packing, out_feat)).to(
                torch.int32
            ),
            "scales": torch.randn((n_grp, out_feat), dtype=torch.float16),
            "qzeros": torch.randint(0, int32_max, (n_grp, out_feat // packing)).to(
                torch.int32
            ),
            "g_idx": torch.randint(0, n_grp, (in_feat,)).to(torch.int32),
        }
        if config.use_bias:
            w2_qparam["bias"] = torch.zeros((out_feat,), dtype=torch.float16)

        def _test_gptq_for_world_size(group, w1_qparam, w2_qparam):
            for rank in range(group.size()):
                qparams = {}
                for k, v in w1_qparam.items():
                    qparams["w1." + k] = v
                for k, v in w2_qparam.items():
                    qparams["w2." + k] = v

                with torch.no_grad():
                    # create TP-sharded attention module
                    tp_mod = get_ffn.to_tp(group)
                    tp_mod.load_weights(qparams)

                # load pre-TP tensors
                w1_qw = qparams["w1.qweight"]
                w1_scales = qparams["w1.scales"]
                w1_qzeros = qparams["w1.qzeros"]
                w1_gidx = qparams["w1.g_idx"]
                if config.use_bias:
                    w1_bias = qparams["w1.bias"]

                # # assert on w1 linear module
                w1_mult = config.multiple_of * config.emb_dim // group.size()
                w1_mult_packed = w1_mult // packing
                torch.testing.assert_close(
                    w1_qw[:, w1_mult * rank : w1_mult * (rank + 1)], tp_mod.w1.qweight
                )
                torch.testing.assert_close(
                    w1_scales[:, w1_mult * rank : w1_mult * (rank + 1)],
                    tp_mod.w1.scales,
                )
                torch.testing.assert_close(
                    w1_qzeros[:, w1_mult_packed * rank : w1_mult_packed * (rank + 1)],
                    tp_mod.w1.qzeros,
                )
                torch.testing.assert_close(w1_gidx - min(w1_gidx), tp_mod.w1.g_idx)
                if config.use_bias:
                    torch.testing.assert_close(
                        w1_bias[w1_mult * rank : w1_mult * (rank + 1)], tp_mod.w1.bias
                    )
                else:
                    assert (
                        not hasattr(tp_mod.w1, "bias")
                        or tp_mod.w1.bias is None
                        or torch.sum(tp_mod.w1.bias) == 0
                    )
                    assert (
                        w1_qparam.get("bias", None) is None
                        or torch.sum(w1_qparam.get("bias")) == 0
                    )

                # load pre-TP tensors
                w2_qw = qparams["w2.qweight"]
                w2_scales = qparams["w2.scales"]
                w2_qzeros = qparams["w2.qzeros"]
                w2_gidx = qparams["w2.g_idx"]
                if config.use_bias:
                    w2_bias = qparams["w2.bias"]

                # assert on w2 linear module
                w2_mult = config.multiple_of * config.emb_dim // group.size()
                w2_mult_packed = w2_mult // packing
                w2_mult_grp = w2_mult // config.group_size
                torch.testing.assert_close(
                    w2_qw[w2_mult_packed * rank : w2_mult_packed * (rank + 1)],
                    tp_mod.w2.qweight,
                )
                torch.testing.assert_close(
                    w2_scales[w2_mult_grp * rank : w2_mult_grp * (rank + 1)],
                    tp_mod.w2.scales,
                )
                torch.testing.assert_close(
                    w2_qzeros[w2_mult_grp * rank : w2_mult_grp * (rank + 1)],
                    tp_mod.w2.qzeros,
                )
                w2_gidx_unscaled = w2_gidx[w2_mult * rank : w2_mult * (rank + 1)]
                torch.testing.assert_close(
                    w2_gidx_unscaled - min(w2_gidx_unscaled),
                    tp_mod.w2.g_idx,
                )
                if config.use_bias:
                    torch.testing.assert_close(
                        w2_bias,
                        tp_mod.w2.bias,
                    )
                else:
                    assert (
                        not hasattr(tp_mod.w2, "bias")
                        or tp_mod.w2.bias is None
                        or torch.sum(tp_mod.w2.bias) == 0
                    )
                    assert (
                        w2_qparam.get("bias", None) is None
                        or torch.sum(w2_qparam.get("bias")) == 0
                    )
            group.reset_rank()

        _test_gptq_for_world_size(get_mockgroup, w1_qparam, w2_qparam)

    @pytest.mark.autogptq
    def test_gptq_tp_glu_fused(self, get_glu, get_config, get_mockgroup):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "gloo", store=torch.distributed.HashStore(), rank=0, world_size=1
            )

        config = get_config
        packing = 8  # 8x INT4 --> 1x INT32
        int32_max = torch.iinfo(torch.int32).max

        # wg1_fused
        in_feat = config.emb_dim
        out_feat = 2 * config.multiple_of * in_feat
        n_grp = in_feat // config.group_size

        wg1_fused_qparam = {
            "qweight": torch.randint(0, int32_max, (in_feat // packing, out_feat)).to(
                torch.int32
            ),
            "scales": torch.randn((n_grp, out_feat), dtype=torch.float16),
            "qzeros": torch.randint(0, int32_max, (n_grp, out_feat // packing)).to(
                torch.int32
            ),
            "g_idx": torch.randint(0, n_grp, (in_feat,)).to(torch.int32),
        }
        if config.use_bias:
            wg1_fused_qparam["bias"] = torch.zeros((out_feat,), dtype=torch.float16)

        # w2
        in_feat = config.multiple_of * config.emb_dim
        out_feat = config.emb_dim
        n_grp = in_feat // config.group_size

        w2_qparam = {
            "qweight": torch.randint(0, int32_max, (in_feat // packing, out_feat)).to(
                torch.int32
            ),
            "scales": torch.randn((n_grp, out_feat), dtype=torch.float16),
            "qzeros": torch.randint(0, int32_max, (n_grp, out_feat // packing)).to(
                torch.int32
            ),
            "g_idx": torch.randint(0, n_grp, (in_feat,)).to(torch.int32),
        }
        if config.use_bias:
            w2_qparam["bias"] = torch.zeros((out_feat,), dtype=torch.float16)

        def _test_gptq_for_world_size(group, w1_fused_qparam, w2_qparam):
            for rank in range(group.size()):
                qparams = {}
                for k, v in w1_fused_qparam.items():
                    qparams["wg1_fused." + k] = v
                for k, v in w2_qparam.items():
                    qparams["w2." + k] = v

                with torch.no_grad():
                    # create TP-sharded GLU module
                    tp_mod = get_glu.to_tp(group)
                    tp_mod.load_weights(qparams)

                # load split qparams from TP GLU module
                hidden_size_per_rank = (
                    config.multiple_of * config.emb_dim // group.size()
                )
                tp_wg_qw, tp_w1_qw = torch.split(
                    tp_mod.wg1_fused.qweight,
                    [hidden_size_per_rank, hidden_size_per_rank],
                    dim=1,
                )
                tp_wg_scales, tp_w1_scales = torch.split(
                    tp_mod.wg1_fused.scales,
                    [hidden_size_per_rank, hidden_size_per_rank],
                    dim=1,
                )
                tp_wg_qzeros, tp_w1_qzeros = torch.split(
                    tp_mod.wg1_fused.qzeros,
                    [hidden_size_per_rank // packing, hidden_size_per_rank // packing],
                    dim=1,
                )
                tp_gidx = tp_mod.wg1_fused.g_idx
                if config.use_bias:
                    tp_wg_bias, tp_w1_bias = torch.split(
                        tp_mod.wg1_fused.bias,
                        [hidden_size_per_rank, hidden_size_per_rank],
                        dim=0,
                    )

                # load pre-TP tensors of wg1_fused linear module
                wg1_fused_qw = qparams["wg1_fused.qweight"]
                wg1_fused_scales = qparams["wg1_fused.scales"]
                wg1_fused_qzeros = qparams["wg1_fused.qzeros"]
                wg1_fused_gidx = qparams["wg1_fused.g_idx"]
                if config.use_bias:
                    wg1_fused_bias = qparams["wg1_fused.bias"]

                # assert on wg linear module
                wg_out_dim = config.multiple_of * config.emb_dim
                wg_mult = wg_out_dim // group.size()
                wg_mult_packed = wg_mult // packing
                torch.testing.assert_close(
                    wg1_fused_qw[:, wg_mult * rank : wg_mult * (rank + 1)], tp_wg_qw
                )
                torch.testing.assert_close(
                    wg1_fused_scales[:, wg_mult * rank : wg_mult * (rank + 1)],
                    tp_wg_scales,
                )
                torch.testing.assert_close(
                    wg1_fused_qzeros[
                        :, wg_mult_packed * rank : wg_mult_packed * (rank + 1)
                    ],
                    tp_wg_qzeros,
                )
                torch.testing.assert_close(
                    wg1_fused_gidx - min(wg1_fused_gidx), tp_gidx
                )
                if config.use_bias:
                    torch.testing.assert_close(
                        wg1_fused_bias[wg_mult * rank : wg_mult * (rank + 1)],
                        tp_wg_bias,
                    )

                # assert on wg linear module
                w1_mult = config.multiple_of * config.emb_dim // group.size()
                w1_out_idx_start = w1_mult * rank + wg_out_dim
                w1_out_idx_end = w1_mult * (rank + 1) + wg_out_dim
                torch.testing.assert_close(
                    wg1_fused_qw[:, w1_out_idx_start:w1_out_idx_end], tp_w1_qw
                )
                torch.testing.assert_close(
                    wg1_fused_scales[:, w1_out_idx_start:w1_out_idx_end], tp_w1_scales
                )
                torch.testing.assert_close(
                    wg1_fused_qzeros[
                        :, w1_out_idx_start // packing : w1_out_idx_end // packing
                    ],
                    tp_w1_qzeros,
                )
                if config.use_bias:
                    torch.testing.assert_close(
                        wg1_fused_bias[w1_out_idx_start:w1_out_idx_end], tp_w1_bias
                    )
                else:
                    assert (
                        not hasattr(tp_mod.wg1_fused, "bias")
                        or tp_mod.wg1_fused.bias is None
                        or torch.sum(tp_mod.wg1_fused.bias) == 0
                    )
                    assert (
                        wg1_fused_qparam.get("bias", None) is None
                        or torch.sum(wg1_fused_qparam.get("bias")) == 0
                    )

                # load pre-TP tensors of w2 linear module
                w2_qw = qparams["w2.qweight"]
                w2_scales = qparams["w2.scales"]
                w2_qzeros = qparams["w2.qzeros"]
                w2_gidx = qparams["w2.g_idx"]
                if config.use_bias:
                    w2_bias = qparams["w2.bias"]

                # assert on w2 linear module
                w2_mult = config.multiple_of * config.emb_dim // group.size()
                w2_mult_packed = w2_mult // packing
                w2_mult_grp = w2_mult // config.group_size
                torch.testing.assert_close(
                    w2_qw[w2_mult_packed * rank : w2_mult_packed * (rank + 1)],
                    tp_mod.w2.qweight,
                )
                torch.testing.assert_close(
                    w2_scales[w2_mult_grp * rank : w2_mult_grp * (rank + 1)],
                    tp_mod.w2.scales,
                )
                torch.testing.assert_close(
                    w2_qzeros[w2_mult_grp * rank : w2_mult_grp * (rank + 1)],
                    tp_mod.w2.qzeros,
                )
                w2_gidx_unscaled = w2_gidx[w2_mult * rank : w2_mult * (rank + 1)]
                torch.testing.assert_close(
                    w2_gidx_unscaled - min(w2_gidx_unscaled),
                    tp_mod.w2.g_idx,
                )
                if config.use_bias:
                    torch.testing.assert_close(
                        w2_bias,
                        tp_mod.w2.bias,
                    )
                else:
                    assert (
                        not hasattr(tp_mod.w2, "bias")
                        or tp_mod.w2.bias is None
                        or torch.sum(tp_mod.w2.bias) == 0
                    )
                    assert (
                        w2_qparam.get("bias", None) is None
                        or torch.sum(w2_qparam.get("bias")) == 0
                    )
            group.reset_rank()

        _test_gptq_for_world_size(get_mockgroup, wg1_fused_qparam, w2_qparam)
