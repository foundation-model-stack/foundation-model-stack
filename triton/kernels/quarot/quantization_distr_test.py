import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from utils import quantize, dequantize, print_test_results, random_rotation_almost_hadamard

def quantize_m_distr(x, distr):
    # make sure x and w are fp16
    x = x.type(torch.float16)
    # quantize to int8
    x_q, x_qs = quantize(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            distr[x_q[i, j] + 128] += 1

    # immediately dequantize
    x_q = dequantize(x_q, x_qs)
    
    return x, x_q

m, k, n = 1024, 1024, 1024 # 4096, 4096, 4096

# seed = 3
# torch.manual_seed(seed)

runs_per_test = 1

x_results_basic = []
x_results_rot_rand = []
x_results_rot_rand_transp = []
x_results_rot_hard = []
w_results_basic = []
w_results_rot_rand = []
w_results_rot_rand_transp = []
w_results_rot_hard = []

x_distr_basic = [0] * 256
x_distr_rot_rand = [0] * 256
x_distr_rot_rand_transp = [0] * 256
x_distr_rot_hard = [0] * 256
w_distr_basic = [0] * 256
w_distr_rot_rand = [0] * 256
w_distr_rot_rand_transp = [0] * 256
w_distr_rot_hard = [0] * 256

all_x = []
all_w = []
all_x_rot_rand = []
all_w_rot_rand = []
all_x_rot_rand_transp = []
all_w_rot_rand_transp = []
all_x_rot_hard = []
all_w_rot_hard = []

for i in range(0, runs_per_test):
    x = torch.randn((m, k), dtype=torch.float16).uniform_(-0.1, 0.1)
    for i in range(m * k // 100):
        i, j = random.randrange(0, m), random.randrange(0, k)
        x[i, j] = random.uniform(-0.4, 0.4)
    # w = torch.randn((k, n), dtype=torch.float16).uniform_(-0.1, 0.1)
    w = torch.tensor(np.random.normal(0, 0.1, (k, n)), dtype=torch.float16)

    r_rand, r_inv_rand = random_rotation_almost_hadamard(k, use_hardcoded=False, run_full_orthogonality_tests=False, check_inv_max=False)
    r_rand_transp, r_inv_rand_transp = r_rand, r_rand.T
    r_hard, r_inv_hard = random_rotation_almost_hadamard(k, use_hardcoded=True, run_full_orthogonality_tests=False, check_inv_max=False)

    all_x += x.reshape(-1)
    all_w += w.reshape(-1)

    x_rot_rand = x @ r_rand
    w_rot_rand = r_inv_rand.type(torch.float64) @ w.type(torch.float64)
    x_rot_rand_transp = x @ r_rand_transp
    w_rot_rand_transp = r_inv_rand_transp.type(torch.float64) @ w.type(torch.float64)
    x_rot_hard = x @ r_hard
    w_rot_hard = r_inv_hard.type(torch.float64) @ w.type(torch.float64)

    all_x_rot_rand += x_rot_rand.reshape(-1)
    all_x_rot_rand_transp += x_rot_rand_transp.reshape(-1)
    all_x_rot_hard += x_rot_hard.reshape(-1)
    all_w_rot_rand += w_rot_rand.reshape(-1)
    all_w_rot_rand_transp += w_rot_rand_transp.reshape(-1)
    all_w_rot_hard += w_rot_hard.reshape(-1)

    x_results_basic.append(quantize_m_distr(x, x_distr_basic))
    x_results_rot_rand.append(quantize_m_distr(x_rot_rand, x_distr_rot_rand))
    x_results_rot_rand_transp.append(quantize_m_distr(x_rot_rand_transp, x_distr_rot_rand_transp))
    x_results_rot_hard.append(quantize_m_distr(x_rot_hard, x_distr_rot_hard))

    w_results_basic.append(quantize_m_distr(w, w_distr_basic))
    w_results_rot_rand.append(quantize_m_distr(w_rot_rand, w_distr_rot_rand))
    w_results_rot_rand_transp.append(quantize_m_distr(w_rot_rand_transp, w_distr_rot_rand_transp))
    w_results_rot_hard.append(quantize_m_distr(w_rot_hard, w_distr_rot_hard))

print_test_results(x_results_basic, 'x basic quantization')
print_test_results(w_results_basic, 'w basic quantization')
print_test_results(x_results_rot_rand, 'x rotated quantization (random)')
print_test_results(w_results_rot_rand, 'w rotated quantization (random)')
print_test_results(x_results_rot_rand_transp, 'x rotated quantization (random, transpose)')
print_test_results(w_results_rot_rand_transp, 'w rotated quantization (random, transpose)')
print_test_results(x_results_rot_hard, 'x rotated quantization (hadamard hardcoded)')
print_test_results(w_results_rot_hard, 'w rotated quantization (hadamard hardcoded)')

fig, axs = plt.subplots(8, 2, sharey=True, tight_layout=True)
n_bins = 64

int_128_vals = list(range(-128, 128))

axs[0, 0].hist(int_128_vals, weights=x_distr_basic, bins=n_bins)
axs[0, 1].hist(int_128_vals, weights=w_distr_basic, bins=n_bins)
axs[1, 0].hist(int_128_vals, weights=x_distr_rot_rand, bins=n_bins)
axs[1, 1].hist(int_128_vals, weights=w_distr_rot_rand, bins=n_bins)
axs[2, 0].hist(int_128_vals, weights=x_distr_rot_rand_transp, bins=n_bins)
axs[2, 1].hist(int_128_vals, weights=w_distr_rot_rand_transp, bins=n_bins)
axs[3, 0].hist(int_128_vals, weights=x_distr_rot_hard, bins=n_bins)
axs[3, 1].hist(int_128_vals, weights=w_distr_rot_hard, bins=n_bins)
axs[4, 0].hist(all_x, bins=n_bins)
axs[4, 1].hist(all_w, bins=n_bins)
axs[5, 0].hist(all_x_rot_rand, bins=n_bins)
axs[5, 1].hist(all_w_rot_rand, bins=n_bins)
axs[6, 0].hist(all_x_rot_rand_transp, bins=n_bins)
axs[6, 1].hist(all_w_rot_rand_transp, bins=n_bins)
axs[7, 0].hist(all_x_rot_hard, bins=n_bins)
axs[7, 1].hist(all_w_rot_hard, bins=n_bins)

plt.show()