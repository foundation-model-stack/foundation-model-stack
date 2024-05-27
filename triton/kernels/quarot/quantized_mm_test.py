import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import quantize, dequantize, random_rotation_almost_hadamard, print_test_results

def quantized_mm(x: torch.Tensor, w: torch.Tensor):
    # make sure x and w are fp16
    x = x.type(torch.float16)
    w = w.type(torch.float16)
    # quantize to int8
    x_q, x_qs = quantize(x)
    w_q, w_qs = quantize(w)

    # get truth/actual result for fp16
    c = (x @ w)
    # get int8 quantized result
    # for CPU, we need to cast to int16 otherwise it won't accumilate in int16
    c_q = x_q.type(torch.int16) @ w_q.type(torch.int16)
    c_q = dequantize(c_q, x_qs * w_qs)

    return c, c_q

m, k, n = 1024, 1024, 1024 # 4096, 4096, 4096

# seed = 3
# torch.manual_seed(seed)

runs_per_test = 1

results_basic = []
results_rot_rand = []
results_rot_rand_transp = []
results_rot_hard = []

for i in range(0, runs_per_test):
    x = torch.randn((m, k), dtype=torch.float16).uniform_(-0.1, 0.1)
    for i in range(m * k // 100):
        i, j = random.randrange(0, m), random.randrange(0, k)
        x[i, j] = random.uniform(-0.4, 0.4)
    # w = torch.randn((k, n), dtype=torch.float16).uniform_(-0.1, 0.1)
    w = torch.tensor(np.random.normal(0, 0.1, (k, n)), dtype=torch.float16)

    r_rand, r_inv_rand = random_rotation_almost_hadamard(k, use_hardcoded=False, run_full_orthogonality_tests=False, check_inv_max=False)
    r_hard, r_inv_hard = random_rotation_almost_hadamard(k, use_hardcoded=True, run_full_orthogonality_tests=False, check_inv_max=False)

    x_rot_rand = x @ r_rand
    w_rot_rand = r_inv_rand.type(torch.float64) @ w.type(torch.float64)
    x_rot_hard = x @ r_hard
    w_rot_hard = r_inv_hard.type(torch.float64) @ w.type(torch.float64)
    w_rot_rand_transp = r_rand.T.type(torch.float64) @ w.type(torch.float64)

    # TODO: generate rotation matrices every time
    results_basic.append(quantized_mm(x, w))
    results_rot_rand.append(quantized_mm(x_rot_rand, w_rot_rand))
    results_rot_rand_transp.append(quantized_mm(x_rot_rand, w_rot_rand_transp))
    results_rot_hard.append(quantized_mm(x_rot_hard, w_rot_hard))

print_test_results(results_basic, 'basic quantization')
print_test_results(results_rot_rand, 'rotated quantization (random)')
print_test_results(results_rot_rand_transp, 'rotated quantization (random, transpose)')
print_test_results(results_rot_hard, 'rotated quantization (hadamard hardcoded)')