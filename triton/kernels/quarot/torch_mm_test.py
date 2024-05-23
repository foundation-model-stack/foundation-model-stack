import random
from scipy.linalg import hadamard
import torch
import numpy as np

int8_mag_max = 127

# TODO: look into gradient scaling

def quantize(x: torch.HalfTensor):
    quantize_scale = int8_mag_max / torch.max(torch.abs(x))
    return (x.type(torch.float64) * quantize_scale).type(torch.int8), quantize_scale.type(torch.float64)

def dequantize(x: torch.CharTensor, quantize_scale):
    return (x.type(torch.float64) / quantize_scale).type(torch.float16)

def random_rotation_almost_hadamart(size: int, use_hardcoded=False):
    if use_hardcoded:
        m = torch.FloatTensor(hadamard(size))
    else:
        while True:
            try:
                m = torch.where(torch.rand((size, size)) >= 0.5, -1, 1).type(torch.float32)
                # print(f"m: {m}")
                # since this isn't quite a hadamard matrix, it might have an extreme inverse
                # if it's too extreme, it could cause inf in float16 when multiplied
                m_inv = torch.inverse(m).type(torch.float16)
                # TODO: determine what max value is acceptable
                if torch.max(torch.abs(m_inv)) < 65504.0 / 10.0:
                    break
                else:
                    print("random matrix was bad, regenerating...")
            except Exception as e:
                print(e)
                pass
    m = m / torch.sqrt(torch.tensor(k)) # torch.sqrt(m.shape)
    m_inv = torch.inverse(m)
    m = m.type(torch.float16)
    m_inv = m_inv.type(torch.float16)
    return m, m_inv

def quantize_mm_avg_err(x, w, print_quantized=False):
    # make sure x and w are fp16
    x = x.type(torch.float16)
    w = w.type(torch.float16)
    # quantize to int8
    x_q, x_qs = quantize(x)
    w_q, w_qs = quantize(w)

    if print_quantized:
        print(f"x quantized: {x_q}")
        print(f"w quantized: {w_q}")

    # get truth/actual result for fp16
    c = (x @ w)
    # get int8 quantized result
    # for CPU, we need to cast to int16 otherwise it won't accumilate in int16
    c_q = x_q.type(torch.int16) @ w_q.type(torch.int16)
    # debug: scale 
    # print(f"x scale: {x_qs} w scale: {w_qs}")
    c_q = dequantize(c_q, x_qs * w_qs)

    c_diff = torch.abs(c.type(torch.float64) - c_q.type(torch.float64))
    c_diff_frac = torch.abs(c_diff.type(torch.float64) / (c.type(torch.float64) + 0.001)) # TODO: silly fix
    c_diff_avg = torch.mean(c_diff_frac.type(torch.float64), (0, 1))

    return c, c_q, c_diff, c_diff_frac, c_diff_avg

def print_test_results(results, test_name='test'):
    diff_avg = 0
    for result in results:
        c, c_q, c_diff, c_diff_frac, c_diff_avg = result
        c_diff_avg = c_diff_avg.type(torch.float64)
        assert c_diff_avg != torch.nan and c_diff_avg != torch.inf
        diff_avg += c_diff_avg
    diff_avg /= len(results)
    print("==========================")
    print(f"running test: {test_name}")
    # try:
    #     torch.testing.assert_close(c_q, c, rtol=0, atol=1e-2)
    # except Exception as e:
    #     print(e)
    print(diff_avg)
    print(f"finished test: {test_name}")
    print("==========================")
    assert diff_avg != torch.inf

m, k, n = 16, 16, 16 # 512, 512, 512

seed = 3
torch.manual_seed(seed)

runs_per_test = 100

results_basic = []
results_rot_rand = []
results_rot_hard = []

for i in range(0, runs_per_test):
    x = torch.randn((m, k), dtype=torch.float16).uniform_(-0.1, 0.1)
    w = torch.randn((k, n), dtype=torch.float16).uniform_(-0.1, 0.1)
    # w = torch.tensor(np.random.normal(0, 0.1, (k, n)), dtype=torch.float16)

    r_rand, r_inv_rand = random_rotation_almost_hadamart(k, use_hardcoded=False)
    r_hard, r_inv_hard = random_rotation_almost_hadamart(k, use_hardcoded=True)

    results_basic.append(quantize_mm_avg_err(x, w))
    results_rot_rand.append(quantize_mm_avg_err(x @ r_rand, r_inv_rand @ w))
    results_rot_hard.append(quantize_mm_avg_err(x @ r_hard, r_inv_hard @ w))

print_test_results(results_basic, 'basic quantization')
print_test_results(results_rot_rand, 'rotated quantization (random)')
print_test_results(results_rot_hard, 'rotated quantization (hadamard hardcoded)')