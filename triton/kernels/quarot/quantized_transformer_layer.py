import random
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from utils import random_rotation_almost_hadamard, print_test_results
import utils

class FFNLayer(Module):
    def __init__(self, hidden_size, intermediate_size, w_gate, w_up, w_down, scaling_factor, activation=None) -> None:
        super().__init__()
        if activation is None:
            activation = torch.nn.functional.silu
        self.activation = activation
        self.q, q_inv = self.get_rotation_and_inv(hidden_size)
        self.h, h_inv = self.get_rotation_and_inv(intermediate_size)
        self.w_up, self.w_up_s = self.quantize(q_inv.type(torch.float32) * scaling_factor.type(torch.float32) @ w_up.type(torch.float32))
        self.w_gate, self.w_gate_s = self.quantize(q_inv.type(torch.float32) * scaling_factor.type(torch.float32) @ w_gate.type(torch.float32))
        self.w_down, self.w_down_s = self.quantize(h_inv.type(torch.float32) @ w_down.type(torch.float32))

    def forward(self, input: Tensor) -> Tensor:
        input = input @ self.q
        input = input / input.square().sum(dim=1, keepdim=True).sqrt()
        input_q, input_q_s = self.quantize(input)

        gate_out, gate_out_s = input_q @ self.w_gate, input_q_s * self.w_gate_s
        gate_out = self.dequantize(gate_out, gate_out_s)
        gate_out = self.activation(gate_out)

        up_out, up_out_s = input_q @ self.w_up, input_q_s * self.w_up_s
        up_out = self.dequantize(up_out, up_out_s)

        temp = gate_out * up_out
        temp_r = temp @ self.h
        temp_r, temp_r_s = self.quantize(temp_r)

        down_out, down_out_s = temp_r @ self.w_down, temp_r_s * self.w_down_s
        down_out = self.dequantize(down_out, down_out_s)

        return down_out
    
    def quantize(self, x: torch.Tensor):
        return x.type(torch.float16), 1
    
    def dequantize(self, x, scale):
        return x

    def get_rotation_and_inv(self, size):
        return torch.eye(size, dtype=torch.float16), torch.eye(size, dtype=torch.float16)

class QuantizedFFNLayer(FFNLayer):
    def quantize(self, x):
        return utils.quantize(x)
    
    def dequantize(self, x, scale):
        return utils.dequantize(x, scale)

class QuantizedHadRotFFN(QuantizedFFNLayer):
    def get_rotation_and_inv(self, size):
        return random_rotation_almost_hadamard(size, use_hardcoded=True, run_full_orthogonality_tests=False, check_inv_max=False)
    
class QuantizedRandRotFFN(QuantizedFFNLayer):
    def get_rotation_and_inv(self, size):
        return random_rotation_almost_hadamard(size, use_hardcoded=False, run_full_orthogonality_tests=False, check_inv_max=False)
    
class QuantizedRandRotInvTFFN(QuantizedFFNLayer):
    def get_rotation_and_inv(self, size):
        q, _ = random_rotation_almost_hadamard(size, use_hardcoded=False, run_full_orthogonality_tests=False, check_inv_max=False)
        return q, q.T

context_size, hidden_size, intermediate_size = 512, 1024, 2048 # 2048, 4096, 8192

runs_per_test = 1

results_basic = []
results_rot_rand = []
results_rot_rand_transp = []
results_rot_hard = []

for i in range(0, runs_per_test):
    x = torch.randn((context_size, hidden_size), dtype=torch.float16).uniform_(-0.1, 0.1)
    for i in range(context_size * hidden_size // 100):
        i, j = random.randrange(0, context_size), random.randrange(0, hidden_size)
        x[i, j] = random.uniform(-0.4, 0.4)

    w_up = torch.tensor(np.random.uniform(-0.1, 0.1, (hidden_size, intermediate_size)), dtype=torch.float16)
    w_gate = torch.tensor(np.random.uniform(-0.1, 0.1, (hidden_size, intermediate_size)), dtype=torch.float16)
    w_down = torch.tensor(np.random.uniform(-0.1, 0.1, (intermediate_size, context_size)), dtype=torch.float16)
    scaling_factor = torch.tensor(np.random.uniform(-0.1, 0.1, (1, hidden_size)), dtype=torch.float16)

    model = FFNLayer(hidden_size, intermediate_size, w_gate, w_up, w_down, scaling_factor)
    model_q = QuantizedFFNLayer(hidden_size, intermediate_size, w_gate, w_up, w_down, scaling_factor)
    model_qh = QuantizedHadRotFFN(hidden_size, intermediate_size, w_gate, w_up, w_down, scaling_factor)
    model_qr = QuantizedRandRotFFN(hidden_size, intermediate_size, w_gate, w_up, w_down, scaling_factor)
    model_qrt = QuantizedRandRotInvTFFN(hidden_size, intermediate_size, w_gate, w_up, w_down, scaling_factor)

    truth = model(x)

    results_basic.append((truth, model_q(x)))
    results_rot_hard.append((truth, model_qh(x)))
    results_rot_rand.append((truth, model_qr(x)))
    results_rot_rand_transp.append((truth, model_qrt(x)))

print_test_results(results_basic, 'basic quantization')
print_test_results(results_rot_hard, 'rotated quantization (hadamard hardcoded)')
print_test_results(results_rot_rand, 'rotated quantization (random)')
print_test_results(results_rot_rand_transp, 'rotated quantization (random, transpose)')