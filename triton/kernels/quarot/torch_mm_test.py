import random
from scipy.linalg import hadamard
import torch
import numpy as np
import matplotlib.pyplot as plt

int8_mag_max = 127

# TODO: look into gradient scaling

def quantize(x: torch.HalfTensor):
    quantize_scale = int8_mag_max / torch.max(torch.abs(x))
    return (x.type(torch.float64) * quantize_scale).type(torch.int8), quantize_scale.type(torch.float64)

def dequantize(x: torch.CharTensor, quantize_scale):
    return (x.type(torch.float64) / quantize_scale).type(torch.float16)

num_test_hadamards = 1
num_orthogonality_tests = 0
max_allowed_inv_value = 100

def random_rotation_almost_hadamart(size: int, use_hardcoded=False):
    if use_hardcoded:
        m = torch.FloatTensor(hadamard(size)) / torch.sqrt(torch.tensor(k))
    else:
        potential = []
        while len(potential) < num_test_hadamards:
            try:
                m = torch.where(torch.rand((size, size)) >= 0.5, -1, 1).type(torch.float32) / torch.sqrt(torch.tensor(k))
                avg_row_dot_prod = 0
                tests_passed = True

                # for i in range(size):
                #     for j in range(i + 1, size):
                #         dot_prod = torch.abs(torch.dot(m[i], m[j]))
                #         if dot_prod > 0.5:
                #             tests_passed = False
                #             break
                #         avg_row_dot_prod += dot_prod
                #     if not tests_passed:
                #         break

                for _ in range(num_orthogonality_tests):
                    i, j = 0, 0
                    while i == j:
                        i, j = random.randrange(k), random.randrange(k)
                    dot_prod = torch.abs(torch.dot(m[i], m[j]))
                    if dot_prod > 0.5:
                        tests_passed = False
                    avg_row_dot_prod += dot_prod
                avg_row_dot_prod /= (size - 1) * (size - 2)
                if not tests_passed:
                    continue

                # print(f"m: {m}")
                # since this isn't quite a hadamard matrix, it might have an extreme inverse
                # if it's too extreme, it could cause inf in float16 when multiplied
                m_inv = torch.inverse(m).type(torch.float16)
                # TODO: determine what max value is acceptable
                # if torch.max(torch.square(m_inv).sum(dim=1)) < max_allowed_inv_value:
                potential.append((m, avg_row_dot_prod))
                # else:       random matrix was bad, regenerating...")
            except Exception as e:
                print(e)
                pass
        m, _ = min(potential, key=lambda x: x[1])
    # m = m / torch.sqrt(torch.tensor(k)) # torch.sqrt(m.shape)
    m_inv = torch.inverse(m)
    m = m.type(torch.float16)
    m_inv = m_inv.type(torch.float16)
    return m, m_inv

def quantize_mm_avg_err(x, w):
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

    c_diff = torch.abs(c.type(torch.float64) - c_q.type(torch.float64))
    c_diff_frac = torch.abs(c_diff.type(torch.float64) / (c.type(torch.float64) + 0.001)) # TODO: silly fix
    c_diff_avg = torch.mean(c_diff.type(torch.float64), (0, 1))
    c_diff_frac_avg = torch.mean(c_diff_frac.type(torch.float64), (0, 1))
    c_diff_max = torch.max(c_diff.type(torch.float64))
    c_diff_frac_max = torch.max(c_diff_frac.type(torch.float64))

    return c, c_q, c_diff, c_diff_frac, c_diff_avg, c_diff_frac_avg, c_diff_max, c_diff_frac_max

def quantize_mm_activ_avg_err(x, w, activ):
    # make sure x and w are fp16
    x = x.type(torch.float16)
    w = w.type(torch.float16)
    # quantize to int8
    x_q, x_qs = quantize(x)
    w_q, w_qs = quantize(w)

    # get truth/actual result for fp16
    c = activ(x @ w)
    # get int8 quantized result
    # for CPU, we need to cast to int16 otherwise it won't accumilate in int16
    c_q = x_q.type(torch.int16) @ w_q.type(torch.int16)
    c_q = activ(dequantize(c_q, x_qs * w_qs))

    c_diff = torch.abs(c.type(torch.float64) - c_q.type(torch.float64))
    c_diff_frac = torch.abs(c_diff.type(torch.float64) / (c.type(torch.float64) + 0.001)) # TODO: silly fix
    c_diff_avg = torch.mean(c_diff.type(torch.float64), (0, 1))
    c_diff_frac_avg = torch.mean(c_diff_frac.type(torch.float64), (0, 1))
    c_diff_max = torch.max(c_diff.type(torch.float64))
    c_diff_frac_max = torch.max(c_diff_frac.type(torch.float64))

    return c, c_q, c_diff, c_diff_frac, c_diff_avg, c_diff_frac_avg, c_diff_max, c_diff_frac_max

def quantize_ffn_err(x, wg, wu, wd, activ):
    # make sure x and w are fp16
    x = x.type(torch.float16)
    wg = wg.type(torch.float16)
    wu = wu.type(torch.float16)
    wd = wd.type(torch.float16)

    # quantize to int8
    x_q, x_qs = quantize(x)
    wg_q, wg_qs = quantize(wg)
    wu_q, wu_qs = quantize(wu)
    wd_q, wd_qs = quantize(wd)

    xwg = x @ wg
    xwu = x @ wu
    xwg_activ = activ(xwg)
    temp = xwg_activ * xwu
    

    # get truth/actual result for fp16
    c = activ(x @ w)
    # get int8 quantized result
    # for CPU, we need to cast to int16 otherwise it won't accumilate in int16
    c_q = x_q.type(torch.int16) @ w_q.type(torch.int16)
    c_q = activ(dequantize(c_q, x_qs * w_qs))

    c_diff = torch.abs(c.type(torch.float64) - c_q.type(torch.float64))
    c_diff_frac = torch.abs(c_diff.type(torch.float64) / (c.type(torch.float64) + 0.001)) # TODO: silly fix
    c_diff_avg = torch.mean(c_diff.type(torch.float64), (0, 1))
    c_diff_frac_avg = torch.mean(c_diff_frac.type(torch.float64), (0, 1))
    c_diff_max = torch.max(c_diff.type(torch.float64))
    c_diff_frac_max = torch.max(c_diff_frac.type(torch.float64))

    return c, c_q, c_diff, c_diff_frac, c_diff_avg, c_diff_frac_avg, c_diff_max, c_diff_frac_max

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

    x_diff = torch.abs(x.type(torch.float64) - x_q.type(torch.float64))
    x_diff_frac = torch.abs(x_diff.type(torch.float64) / (x.type(torch.float64) + 0.001)) # TODO: silly fix
    x_diff_avg = torch.mean(x_diff.type(torch.float64), (0, 1))
    x_diff_frac_avg = torch.mean(x_diff_frac.type(torch.float64), (0, 1))
    x_diff_max = torch.max(x_diff.type(torch.float64))
    x_diff_frac_max = torch.max(x_diff_frac.type(torch.float64))

    return x, x_q, x_diff, x_diff_frac, x_diff_avg, x_diff_frac_avg, x_diff_max, x_diff_frac_max

def print_test_results(results, test_name='test'):
    diff_avg = 0
    cos_avg = 0
    diff_frac_avg = 0
    diff_max_avg = 0
    diff_frac_max_avg = 0
    for result in results:
        c, c_q, c_diff, c_diff_frac, c_diff_avg, c_diff_frac_avg, c_diff_max, c_diff_frac_max = result
        c_diff_avg = c_diff_avg.type(torch.float64)
        assert c_diff_avg != torch.nan and c_diff_avg != torch.inf
        diff_avg += c_diff_avg
        diff_frac_avg += c_diff_frac_avg
        cos_avg += torch.nn.functional.cosine_similarity(c.reshape(-1).type(torch.float64), c_q.reshape(-1).type(torch.float64), dim=0)
        diff_max_avg += c_diff_max
        diff_frac_max_avg += c_diff_frac_max
    diff_avg /= len(results)
    cos_avg /= len(results)
    diff_frac_avg /= len(results)
    diff_max_avg /= len(results)
    diff_frac_max_avg /= len(results)
    print("==========================")
    print(f"running test: {test_name}")
    # try:
    #     torch.testing.assert_close(c_q, c, rtol=0, atol=1e-2)
    # except Exception as e:
    #     print(e)
    print(f"Average mean absolute diff: {float(diff_avg):0.5f}")
    print(f"Average mean relative diff: {float(diff_frac_avg):0.5f}")
    print(f"Average cossim: {float(cos_avg):0.5f}")
    print(f"Average max absolute diff: {float(diff_max_avg):0.5f}")
    print(f"Average max relative diff: {float(diff_frac_max_avg):0.5f}")
    print("==========================")
    assert diff_avg != torch.inf

m, k, n = 1024, 1024, 1024 # 4096, 4096, 4096

# seed = 3
# torch.manual_seed(seed)

runs_per_test = 1

results_basic = []
results_rot_rand = []
results_rot_transp_rand = []
results_rot_hard = []

x_results_basic = []
x_results_rot_rand = []
x_results_rot_hard = []
w_results_basic = []
w_results_rot_rand = []
w_results_rot_hard = []

x_distr_basic = [0] * 256
x_distr_rot_rand = [0] * 256
x_distr_rot_hard = [0] * 256
w_distr_basic = [0] * 256
w_distr_rot_rand = [0] * 256
w_distr_rot_hard = [0] * 256

all_x = []
all_w = []
all_x_rot_rand = []
all_w_rot_rand = []
all_x_rot_hard = []
all_w_rot_hard = []

for i in range(0, runs_per_test):
    x = torch.randn((m, k), dtype=torch.float16).uniform_(-0.1, 0.1)
    for i in range(m * k // 100):
        i, j = random.randrange(0, m), random.randrange(0, k)
        x[i, j] = random.uniform(-0.4, 0.4)
    # w = torch.randn((k, n), dtype=torch.float16).uniform_(-0.1, 0.1)
    w = torch.tensor(np.random.normal(0, 0.1, (k, n)), dtype=torch.float16)

    r_rand, r_inv_rand = random_rotation_almost_hadamart(k, use_hardcoded=False)
    r_hard, r_inv_hard = random_rotation_almost_hadamart(k, use_hardcoded=True)

    all_x += x.reshape(-1)
    all_w += w.reshape(-1)

    x_rot_rand = x @ r_rand
    w_rot_rand = r_inv_rand.type(torch.float64) @ w.type(torch.float64)
    x_rot_hard = x @ r_hard
    w_rot_hard = r_inv_hard.type(torch.float64) @ w.type(torch.float64)
    w_rot_transp_rand = r_rand.T.type(torch.float64) @ w.type(torch.float64)

    all_x_rot_rand += x_rot_rand.reshape(-1)
    all_x_rot_hard += x_rot_hard.reshape(-1)
    all_w_rot_rand += w_rot_rand.reshape(-1)
    all_w_rot_hard += w_rot_hard.reshape(-1)

    x_results_basic.append(quantize_m_distr(x, x_distr_basic))
    x_results_rot_rand.append(quantize_m_distr(x_rot_rand, x_distr_rot_rand))
    x_results_rot_hard.append(quantize_m_distr(x_rot_hard, x_distr_rot_hard))

    w_results_basic.append(quantize_m_distr(w, w_distr_basic))
    w_results_rot_rand.append(quantize_m_distr(w_rot_rand, w_distr_rot_rand))
    w_results_rot_hard.append(quantize_m_distr(w_rot_hard, w_distr_rot_hard))

    # TODO: generate rotation matrices every time
    # results_basic.append(quantize_mm_avg_err(x, w))
    # results_rot_rand.append(quantize_mm_avg_err(x_rot_rand, w_rot_rand))
    # results_rot_transp_rand.append(quantize_mm_avg_err(x_rot_rand, w_rot_transp_rand))
    # results_rot_hard.append(quantize_mm_avg_err(x_rot_hard, w_rot_hard))

    # softmax = lambda x: torch.nn.functional.silu(x, 1)
    # results_basic.append(quantize_mm_activ_avg_err(x, w, softmax))
    # results_rot_rand.append(quantize_mm_activ_avg_err(x_rot_rand, w_rot_rand, softmax))
    # results_rot_transp_rand.append(quantize_mm_activ_avg_err(x_rot_rand, w_rot_transp_rand, softmax))
    # results_rot_hard.append(quantize_mm_activ_avg_err(x_rot_hard, w_rot_hard, softmax))
    

# print_test_results(results_basic, 'basic quantization')
# print_test_results(results_rot_rand, 'rotated quantization (random)')
# print_test_results(results_rot_hard, 'rotated quantization (hadamard hardcoded)')

print_test_results(x_results_basic, 'x basic quantization')
print_test_results(w_results_basic, 'w basic quantization')
print_test_results(x_results_rot_rand, 'x rotated quantization (random)')
print_test_results(w_results_rot_rand, 'w rotated quantization (random)')
print_test_results(x_results_rot_hard, 'x rotated quantization (hadamard hardcoded)')
print_test_results(w_results_rot_hard, 'w rotated quantization (hadamard hardcoded)')

fig, axs = plt.subplots(6, 2, sharey=True, tight_layout=True)
n_bins = 64

axs[0, 0].hist(list(range(-128, 128)), weights=x_distr_basic, bins=n_bins)
axs[0, 1].hist(list(range(-128, 128)), weights=w_distr_basic, bins=n_bins)
axs[1, 0].hist(list(range(-128, 128)), weights=x_distr_rot_rand, bins=n_bins)
axs[1, 1].hist(list(range(-128, 128)), weights=w_distr_rot_rand, bins=n_bins)
axs[2, 0].hist(list(range(-128, 128)), weights=x_distr_rot_hard, bins=n_bins)
axs[2, 1].hist(list(range(-128, 128)), weights=w_distr_rot_hard, bins=n_bins)
axs[3, 0].hist(all_x, bins=n_bins)
axs[3, 1].hist(all_w, bins=n_bins)
axs[4, 0].hist(all_x_rot_rand, bins=n_bins)
axs[4, 1].hist(all_w_rot_rand, bins=n_bins)
axs[5, 0].hist(all_x_rot_hard, bins=n_bins)
axs[5, 1].hist(all_w_rot_hard, bins=n_bins)

# plt.show()

# print_test_results(results_basic, 'basic quantization')
# print_test_results(results_rot_rand, 'rotated quantization (random)')
# print_test_results(results_rot_transp_rand, 'rotated quantization (random, transposed)')
# print_test_results(results_rot_hard, 'rotated quantization (hadamard hardcoded)')






# k = 4096

# r_rand, r_inv_rand = random_rotation_almost_hadamart(k, use_hardcoded=False)
# r_inv_rand_test = r_rand.T
# identity = torch.eye(k)
# test_identity = r_rand.type(torch.float64) @ r_inv_rand_test.type(torch.float64)
# identity_diff = torch.abs(test_identity - identity)
# print("Max identity diff:", torch.max(identity_diff))
# print("mean identity diff:", torch.mean(identity_diff))

# r_rand_dots = []
# r_inv_rand_dots = []

# comparison_count = 10000

# for _ in range(comparison_count):
#     i, j = 0, 0
#     while i == j:
#         i, j = random.randrange(k), random.randrange(k)
#     dot_prod = torch.nn.functional.cosine_similarity(r_rand[:, i], r_rand[:, j], dim=0)
#     dot_prod_inv = torch.nn.functional.cosine_similarity(r_inv_rand_test[i], r_inv_rand_test[j], dim=0)
#     r_rand_dots.append(dot_prod)
#     r_inv_rand_dots.append(dot_prod_inv)

# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
# n_bins = 64

# axs[0].hist(r_rand_dots, bins=n_bins)
# axs[1].hist(r_inv_rand_dots, bins=n_bins)

plt.show()
