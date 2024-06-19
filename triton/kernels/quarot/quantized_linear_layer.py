import random
import numpy as np
import torch
from torch import Tensor
from torch.nn import Linear
from utils import quantize, dequantize, random_rotation_almost_hadamard, print_test_results

class QuantizedLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, weight_val=None, bias_val=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        self.rotation, rotation_inv = self.get_rotation_and_inv(in_features)
        
        weight_val = weight_val if weight_val is not None else self.weight
        weight, self.weight_scale = quantize(rotation_inv.type(torch.float64) @ weight_val.T.type(torch.float64))
        weight = weight.T
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        bias_val = bias_val if bias_val is not None else self.bias
        if bias:
            self.bias = torch.nn.Parameter(bias_val.type(torch.float16), requires_grad=False)
    def forward(self, input: Tensor) -> Tensor:
        # TODO: test rotations
        input_q, input_scale = quantize(input.type(torch.float16) @ self.rotation)
        c_q = input_q.type(torch.int16) @ self.weight.type(torch.int16).T
        c_q = dequantize(c_q, input_scale * self.weight_scale)
        if self.bias is not None:
            return c_q + self.bias
        else:
            return c_q
    def get_rotation_and_inv(self, size):
        return torch.eye(size, dtype=torch.float16), torch.eye(size, dtype=torch.float16)

class QuantizedHadRotLinear(QuantizedLinear):
    def get_rotation_and_inv(self, size):
        return random_rotation_almost_hadamard(size, use_hardcoded=True, run_full_orthogonality_tests=False, check_inv_max=False)
    
class QuantizedRandRotLinear(QuantizedLinear):
    def get_rotation_and_inv(self, size):
        return random_rotation_almost_hadamard(size, use_hardcoded=False, run_full_orthogonality_tests=False, check_inv_max=False)
    
class QuantizedRandRotInvTLinear(QuantizedLinear):
    def get_rotation_and_inv(self, size):
        q, _ = random_rotation_almost_hadamard(size, use_hardcoded=False, run_full_orthogonality_tests=False, check_inv_max=False)
        return q, q.T

m, k, n = 2048, 2048, 2048 # 16, 16, 16
use_bias = True

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

    w = torch.tensor(np.random.uniform(-0.1, 0.1, (k, n)), dtype=torch.float32)

    model = Linear(k, n, bias=use_bias)
    model.weight = torch.nn.Parameter(w.T)

    model_q = QuantizedLinear(k, n, weight_val=model.weight, bias_val=model.bias, bias=use_bias)
    model_qh = QuantizedHadRotLinear(k, n, weight_val=model.weight, bias_val=model.bias, bias=use_bias)
    model_qr = QuantizedRandRotLinear(k, n, weight_val=model.weight, bias_val=model.bias, bias=use_bias)
    model_qrt = QuantizedRandRotInvTLinear(k, n, weight_val=model.weight, bias_val=model.bias, bias=use_bias)

    truth = model(x.type(torch.float32))

    results_basic.append((truth, model_q(x)))
    results_rot_hard.append((truth, model_qh(x)))
    results_rot_rand.append((truth, model_qr(x)))
    results_rot_rand_transp.append((truth, model_qrt(x)))

print_test_results(results_basic, 'basic quantization')
print_test_results(results_rot_hard, 'rotated quantization (hadamard hardcoded)')
print_test_results(results_rot_rand, 'rotated quantization (random)')
print_test_results(results_rot_rand_transp, 'rotated quantization (random, transpose)')
