import torch
from scipy.linalg import hadamard
import random

int8_mag_max = 127

def quantize(x: torch.Tensor):
    quantize_scale = int8_mag_max / torch.max(torch.abs(x.type(torch.float64)))
    return (x.type(torch.float64) * quantize_scale).type(torch.int8), quantize_scale.type(torch.float64)

def dequantize(x: torch.Tensor, quantize_scale):
    return (x.type(torch.float64) / quantize_scale).type(torch.float16)

num_test_hadamards = 1
num_orthogonality_tests = 100 # 1000
max_allowed_inv_value = 1000 # 1000050
dot_threshold = 0.5 # 0.15

fail_print_interval = 10

def random_rotation_almost_hadamard(size: int, use_hardcoded, run_full_orthogonality_tests, check_inv_max):
    if use_hardcoded:
        m = torch.tensor(hadamard(size), dtype=torch.float32) / torch.sqrt(torch.tensor(size))
    else:
        fail_count = 0
        potential = []
        while len(potential) < num_test_hadamards:
            try:
                m = torch.where(torch.rand((size, size)) >= 0.5, -1, 1).type(torch.float32) / torch.sqrt(torch.tensor(size))

                avg_row_dot_prod = 0
                tests_passed = True
                if run_full_orthogonality_tests:
                    for i in range(size):
                        for j in range(i + 1, size):
                            dot_prod = torch.abs(torch.nn.functional.cosine_similarity(m[i], m[j], dim=0))
                            if dot_prod > dot_threshold:
                                tests_passed = False
                                break
                            avg_row_dot_prod += dot_prod
                        if not tests_passed:
                            break
                else:
                    for _ in range(num_orthogonality_tests):
                        i, j = 0, 0
                        while i == j:
                            i, j = random.randrange(size), random.randrange(size)
                        dot_prod = torch.abs(torch.nn.functional.cosine_similarity(m[i], m[j], dim=0))
                        if dot_prod > dot_threshold:
                            tests_passed = False
                            break
                        avg_row_dot_prod += dot_prod
                
                if not tests_passed:
                    fail_count += 1
                    if fail_count % fail_print_interval == 0:
                        print(f"failed {fail_count} times")
                    continue
                avg_row_dot_prod /= (size - 1) * (size - 2)

                # since this isn't quite a hadamard matrix, it might have an extreme inverse
                # if it's too extreme, it could cause inf in float16 when multiplied; also
                # restricting maximum value in inverse could make the inverse closer to a
                # rotation matrix, which is ideal
                m_inv = torch.inverse(m).type(torch.float16)
                # TODO: determine what max value is acceptable
                if not check_inv_max or torch.max(torch.square(m_inv).sum(dim=1).sqrt()) < max_allowed_inv_value:
                    potential.append((m, avg_row_dot_prod))
                else:
                    fail_count += 1
                    if fail_count % fail_print_interval == 0:
                        print(f"failed {fail_count} times")
                
            except Exception as e:
                print(e)
                pass
        m, _ = min(potential, key=lambda x: x[1])

    m_inv = torch.inverse(m)
    m = m.type(torch.float16)
    m_inv = m_inv.type(torch.float16)
    return m, m_inv

def print_test_results(results, test_name='test'):
    stat_names = ["mean abs diff", "mean rel diff", "cossim", "max abs diff", "max rel diff"]
    stat_formulas = [
        lambda x, x_q, abs_diff, rel_diff: torch.mean(abs_diff, dim=(0, 1)),
        lambda x, x_q, abs_diff, rel_diff: torch.mean(rel_diff, dim=(0, 1)),
        lambda x, x_q, abs_diff, rel_diff: torch.nn.functional.cosine_similarity(x.reshape(-1), x_q.reshape(-1), dim=0),
        lambda x, x_q, abs_diff, rel_diff: torch.max(abs_diff),
        lambda x, x_q, abs_diff, rel_diff: torch.max(rel_diff),
    ]
    stat_vals = [0] * len(stat_names)
    for result in results:
        x, x_q = result
        x, x_q = x.type(torch.float64), x_q.type(torch.float64)
        abs_diff = torch.abs(x - x_q)
        rel_diff = torch.abs(abs_diff / (x + 0.001)) # TODO: silly fix
        for i, stat in enumerate(stat_formulas):
            stat_vals[i] += stat(x, x_q, abs_diff, rel_diff)
    
    for i in range(len(stat_vals)):
        stat_vals[i] /= len(results)
    print("==========================")
    print(test_name)
    for stat_name, stat_val in zip(stat_names, stat_vals):
        print(f"avg {stat_name}: {float(stat_val):0.10f}")
    print("==========================")

    # try:
    #     torch.testing.assert_close(c_q, c, rtol=0, atol=1e-2)
    # except Exception as e:
    #     print(e)

    # assert diff_avg != torch.inf