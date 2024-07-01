import pytest
import torch

import fms.triton.pytorch_ops as pt_ops


def torch_moe(a: torch.Tensor, w1: torch.Tensor, topk_ids: torch.Tensor):
    # T: tokens; D: emb size
    T, D = a.shape
    a = a.view(T, -1, D).repeat(1, topk_ids.shape[1], 1).reshape(-1, D)
    out = torch.zeros(
        T * topk_ids.shape[1], w1.shape[1], dtype=a.dtype, device=a.device
    )

    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = a[mask] @ w1[i].transpose(0, 1)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPUs not available")
@pytest.mark.parametrize("m", [2, 4, 8, 16, 32, 64, 128, 512, 1024, 2048])
@pytest.mark.parametrize("n", [14336 // 2])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("e", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    torch.cuda.manual_seed(3227)
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    score = torch.softmax(score, dim=-1)

    topk_weight, topk_ids = torch.topk(score, topk)

    if topk_ids.numel() <= e:
        padding_size = 16
    else:
        padding_size = 64

    (
        padded_token_ids_per_block,
        expert_block_mapping,
        total_padded_tokens,
    ) = pt_ops.moe_align_block_size(topk_ids, padding_size, e)

    triton_out = torch.ops.moe.moe_mm(
        a,
        w1,
        topk_ids,
        padded_token_ids_per_block,
        expert_block_mapping,
        total_padded_tokens,
        topk_ids.shape[1],
        padding_size,
    )

    torch_base = torch_moe(a, w1, topk_ids)

    # Given the difference in scheduling the internal matmuls
    # and the low precision of FP16/BF16, there are some cases
    # in which the biggest matrices accumulate up to 1e-2 abs
    # error. This only happens for around ~0.03% of all outputs
    torch.testing.assert_close(
        triton_out.view(2 * m, -1), torch_base, atol=1e-2, rtol=1e-2
    )
