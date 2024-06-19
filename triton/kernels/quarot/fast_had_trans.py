import scipy.linalg
import torch
import triton
import triton.language as tl
import scipy

# def is_cuda():
#     # return triton.runtime.driver.active.get_current_target().backend == "cuda"
#     return triton.runtime.driver.active.get_current_target()[0] == "cuda"
#     # return isinstance(triton.runtime.driver, triton.runtime.CudaDriver) #triton.runtime.driver.active.get_current_target().backend == "cuda"

# def is_hip_mi200():
#     target = triton.runtime.driver.active.get_current_target()
#     return target.backend == 'hip' and target.arch == 'gfx90a'


# def get_cuda_autotune_config():
#     return [
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=5,
#                       num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=5,
#                       num_warps=2),
#         # Good config for fp8 inputs.
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128}, num_stages=3,
#                       num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=4,
#                       num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=4,
#                       num_warps=4)
#     ]


# def get_hip_autotune_config():
#     return [
#         triton.Config(
#             {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'waves_per_eu': 2},
#             num_warps=4, num_stages=0),
#         triton.Config(
#             {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'waves_per_eu': 2},
#             num_warps=8, num_stages=0),
#         triton.Config(
#             {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'waves_per_eu': 2},
#             num_warps=8, num_stages=0),
#         triton.Config(
#             {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'waves_per_eu': 3},
#             num_warps=4, num_stages=0),
#         triton.Config(
#             {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'waves_per_eu': 8},
#             num_warps=4, num_stages=0),
#     ]

# def get_autotune_config():
#     if is_cuda():
#         return get_cuda_autotune_config()
#     else:
#         return get_hip_autotune_config()


# # `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
# #   - A list of `triton.Config` objects that define different configurations of
# #       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
# #   - An auto-tuning *key* whose change in values will trigger evaluation of all the
# #       provided configs
# @triton.autotune(
#     configs=get_autotune_config(),
#     key=[], #'M', 'N'
# )



# import os
# os.environ["TRITON_INTERPRET"] = "1"

# @torch.compile()
# def trans(a: torch.Tensor):
#     d = a.shape[-1]
#     h = 1
#     while h < d:
#         i_range = torch.arange(0, d // (h * 2), device='cuda') * h * 2
#         j_range = torch.arange(0, h, device='cuda')
#         idxs = (i_range.view(-1, 1) + j_range.view(1, -1)).view(-1)
#         x = a[idxs, :]
#         y = a[idxs + h, :]
#         a[idxs, :] = x + y
#         a[idxs + h, :] = x - y
#         h *= 2
#         # TODO: divide by sqrt(2)
#     return a

@triton.jit
def triton_trans(a_ptr, b_ptr, M: tl.constexpr, N: tl.constexpr, stride_m, stride_n, h: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    m_range = tl.arange(0, BLOCK_SIZE_M) + pid_0 * BLOCK_SIZE_M
    n_range = tl.arange(0, BLOCK_SIZE_N) + pid_1 * BLOCK_SIZE_N

    my_id = (m_range % h) + (m_range // h) * h * 2
    my_col = n_range
    my_col_mask = my_col < N

    idx1 = (my_id * stride_m).expand_dims(1) + (my_col * stride_n).expand_dims(0)
    idx1_mask = tl.broadcast_to(my_col_mask.expand_dims(0), BLOCK_SIZE_M, BLOCK_SIZE_N)
    x = tl.load(a_ptr + idx1, idx1_mask)
    y = tl.load(a_ptr + idx1 + h * stride_m, idx1_mask)
    tl.store(b_ptr + idx1, (x + y) * tl.rsqrt(2.0), idx1_mask)
    tl.store(b_ptr + idx1 + h * stride_m, (x - y) * tl.rsqrt(2.0), idx1_mask)

def triton_fast_had(a):
    BLOCK_SIZE_M, BLOCK_SIZE_N = 4, 4

    # grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    grid = lambda META: (triton.cdiv(M, 2 * META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']), )

    b = torch.empty_like(a)
    h = 1
    while h < M:
        triton_trans[grid](a, b, M, N, a.stride(0), a.stride(1), h, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
        a, b = b, a
        h *= 2
    return a

# import random
if __name__ == "__main__":
# for _ in range(3):
    M, N = 32, 32
    a = torch.eye(M).cuda()
    c_true = (torch.tensor(scipy.linalg.hadamard(M)).to(torch.float32) / torch.sqrt(torch.tensor(M))).cuda() @ a
    c = triton_fast_had(a)
    print(f"triton: {c}")
    print(f"truth: {c_true}")

# @torch.compile()
# def trans(a: torch.Tensor, h):
#     i_range = torch.arange(0, d // (h * 2), device='cuda') * h * 2
#     j_range = torch.arange(0, h, device='cuda')
#     idxs = (i_range.view(-1, 1) + j_range.view(1, -1)).view(-1)
#     x = a[idxs, :]
#     y = a[idxs + h, :]
#     a[idxs, :] = x + y
#     a[idxs + h, :] = x - y
#     return a

# if __name__ == "__main__":
#     a = torch.eye(32).cuda()
#     h = 1
#     d = a.shape[-1]
#     while h < d:
#         trans(a, h)
#         h *= 2
#     print(a)