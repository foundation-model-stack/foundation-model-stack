import itertools
import pytest
import torch

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# import mmul_triton
import utils

test_sizes = list(itertools.product([16], repeat=3))
close_threshold = 0.05

class TestGEMM:

    

    @pytest.mark.parametrize('m,n,k', test_sizes)
    def test_gemm(self, m, n, k):
        a = torch.randn((m, k), dtype=torch.float32, device='cuda')
        b = torch.randn((k, n), dtype=torch.float32, device='cuda')

        b = b.T

        c_truth = a @ b
        # c_test = mmul_triton.matmul(a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn))
        c_test, _ = torch._scaled_mm(a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn))

        assert torch.allclose(c_truth, c_test.to(torch.float32), atol=0.2, rtol=0.2)

    @pytest.mark.parametrize('m,n,k', test_sizes)
    def test_had_gemm_pytorch(self, m, n, k):
        a = torch.randn((m, k), dtype=torch.float32, device='cuda')
        b = torch.randn((k, n), dtype=torch.float32, device='cuda')

        H, H_inv = utils.random_rotation_almost_hadamard(k, True, False, False)
        H = H.to('cuda', dtype=torch.float32)
        H_inv = H_inv.to('cuda', dtype=torch.float32)

        c_truth = a @ b
        c_test = (a @ H) @ (H_inv @ b)

        assert torch.allclose(c_truth, c_test.to(torch.float32), atol=close_threshold, rtol=0)
    
    @pytest.mark.parametrize('m,n,k', test_sizes)
    def test_had_float8_gemm_pytorch(self, m, n, k):
        a = torch.randn((m, k), dtype=torch.float32, device='cuda')
        b = torch.randn((k, n), dtype=torch.float32, device='cuda')

        H, H_inv = utils.random_rotation_almost_hadamard(k, True, False, False)
        H = H.to('cuda', dtype=torch.float32)
        H_inv = H_inv.to('cuda', dtype=torch.float32)

        c_truth = a @ b
        c_test, _ = torch._scaled_mm((a @ H).to(torch.float8_e4m3fn), (b.T @ H).T.to(torch.float8_e4m3fn))

        assert torch.allclose(c_truth, c_test.to(torch.float32), atol=0.5, rtol=0.5)


if __name__ == "__main__":
    tester = TestGEMM()
    tests_passed = 0
    tests_failed = 0
    tests = [getattr(TestGEMM, func) for func in dir(TestGEMM) if callable(getattr(TestGEMM, func)) and func.startswith("test")]
    for test in tests:
        for size in test_sizes:
            print('---------------------')
            print(f'running {test.__name__} with {size}')
            try:
                test(tester, *size)
                print('test passed')
                tests_passed += 1
            except Exception as e:
                print(e)
                print('test failed')
                tests_failed += 1
            print('---------------------')
    print('******************')
    print(f'{tests_passed} passed, {tests_failed} failed')

