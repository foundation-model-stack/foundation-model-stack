import math
import random
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from utils import random_rotation_almost_hadamard, print_test_results, diag_tile_block
import utils

class FFNLayer(Module):
    def __init__(self, hidden_size, intermediate_size, w_gate, w_up, w_down, scaling_factor, activation=None) -> None:
        super().__init__()
        if activation is None:
            activation = torch.nn.functional.silu
        self.activation = activation
        self.q, q_inv = self.get_rotation_and_inv(hidden_size)
        self.h, h_inv = self.get_rotation_and_inv(intermediate_size)
        self.w_up, self.w_up_s = self.quantize((q_inv.type(torch.float32) * scaling_factor.view(1, -1).type(torch.float32)) @ w_up.type(torch.float32))
        self.w_gate, self.w_gate_s = self.quantize((q_inv.type(torch.float32) * scaling_factor.view(1, -1).type(torch.float32)) @ w_gate.type(torch.float32))
        self.w_down, self.w_down_s = self.quantize(h_inv.type(torch.float32) @ w_down.type(torch.float32))

    def forward(self, input: Tensor) -> Tensor:
        input = input @ self.q
        input = utils.rms_norm(input)
        input_q, input_q_s = self.quantize(input)

        gate_out, gate_out_s = input_q.type(self.acc_t()) @ self.w_gate.type(self.acc_t()), input_q_s * self.w_gate_s
        gate_out = self.dequantize(gate_out, gate_out_s)
        gate_out = self.activation(gate_out)

        up_out, up_out_s = input_q.type(self.acc_t()) @ self.w_up.type(self.acc_t()), input_q_s * self.w_up_s
        up_out = self.dequantize(up_out, up_out_s)

        temp = gate_out * up_out
        temp_r = temp @ self.h
        temp_r, temp_r_s = self.quantize(temp_r)

        down_out, down_out_s = temp_r.type(self.acc_t()) @ self.w_down.type(self.acc_t()), temp_r_s * self.w_down_s
        down_out = self.dequantize(down_out, down_out_s)

        return down_out
    
    def quantize(self, x: torch.Tensor):
        return x.type(torch.float16), 1
    
    def dequantize(self, x, scale):
        return x.type(torch.float16)

    def get_rotation_and_inv(self, size):
        return torch.eye(size, dtype=torch.float16), torch.eye(size, dtype=torch.float16)
    
    def acc_t(self):
        return torch.float32

class QuantizedFFNLayer(FFNLayer):
    def quantize(self, x):
        return utils.quantize(x)
    
    def dequantize(self, x, scale):
        return utils.dequantize(x, scale)
    
    def acc_t(self):
        return torch.int32

class QuantizedHadRotFFN(QuantizedFFNLayer):
    def get_rotation_and_inv(self, size):
        return random_rotation_almost_hadamard(size, use_hardcoded=True, run_full_orthogonality_tests=False, check_inv_max=True)
    
class QuantizedRandRotFFN(QuantizedFFNLayer):
    def get_rotation_and_inv(self, size):
        return random_rotation_almost_hadamard(size, use_hardcoded=False, run_full_orthogonality_tests=False, check_inv_max=True)
    
class QuantizedRandRotInvTFFN(QuantizedFFNLayer):
    def get_rotation_and_inv(self, size):
        q, _ = random_rotation_almost_hadamard(size, use_hardcoded=False, run_full_orthogonality_tests=False, check_inv_max=True)
        return q, q.T

def rope(pos, size, base_theta=10000.0):
    rope_m = torch.zeros((size, size), dtype=torch.float16)
    for i in range(0, size // 2):
        theta = torch.tensor(pos / pow(base_theta, 2 * i / size), dtype=torch.float16)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rope_m[i * 2 + 0, i * 2 + 0] = cos_theta
        rope_m[i * 2 + 0, i * 2 + 1] = -sin_theta
        rope_m[i * 2 + 1, i * 2 + 0] = sin_theta
        rope_m[i * 2 + 1, i * 2 + 1] = cos_theta
    return rope_m



class ATTNLayer(Module):
    def __init__(self, embedding_size, d_v_all, d_k_all, w_q, w_k, w_v, w_out, scaling_factor, num_heads) -> None:
        super().__init__()
        self.q, q_inv = self.get_rotation_and_inv(embedding_size)
        self.w_q, self.w_q_s = self.quantize((q_inv.type(torch.float32) * scaling_factor.view(1, -1).type(torch.float32)) @ w_q.type(torch.float32))
        self.w_k, self.w_k_s = self.quantize((q_inv.type(torch.float32) * scaling_factor.view(1, -1).type(torch.float32)) @ w_k.type(torch.float32))

        self.d_v = d_v_all // num_heads
        self.rot_head, self.rot_head_inv = self.get_rotation_and_inv(self.d_v)
        self.d_k = d_k_all // num_heads
        self.rot_head_k, self.rot_head_k_inv = self.get_rotation_and_inv(self.d_k)

        # instead of computing all 32 heads' V_i separately, we'll tile h_head along a diagonal 32 times to achieve the same thing (the result is the concat of all V_i * h_head)
        self.rot_head_tiled = diag_tile_block(self.rot_head, num_heads)
        self.rot_head_k_tiled = diag_tile_block(self.rot_head_k, num_heads)
        
        self.w_v, self.w_v_s = self.quantize((q_inv.type(torch.float32) * scaling_factor.view(1, -1).type(torch.float32)) @ w_v.type(torch.float32) @ self.rot_head_tiled.type(torch.float32))

        self.w_out, self.w_out_s = self.quantize(self.rot_head_tiled.type(torch.float32).inverse() @ w_out.type(torch.float32))


    def forward(self, input: Tensor) -> Tensor:
        input = input @ self.q
        input = utils.rms_norm(input)
        input_q, input_q_s = self.quantize(input)

        query_vals, query_vals_s = input_q.type(self.acc_t()) @ self.w_q.type(self.acc_t()), input_q_s * self.w_q_s
        query_vals = self.dequantize(query_vals, query_vals_s)

        key_vals, key_vals_s = input_q.type(self.acc_t()) @ self.w_k.type(self.acc_t()), input_q_s * self.w_k_s
        key_vals = self.dequantize(key_vals, key_vals_s)

        val_vals, val_vals_s = input_q.type(self.acc_t()) @ self.w_v.type(self.acc_t()), input_q_s * self.w_v_s
        val_vals = self.dequantize(val_vals, val_vals_s)

        for i in range(input.shape[-2]):
            rope_m = rope(i, input.shape[-1]) # doesn't matter what it starts at as long as sequential?
            query_vals[i] = query_vals[i] @ rope_m
            key_vals[i] = key_vals[i] @ rope_m

        # TODO: kv cache would be quantized

        query_vals, query_vals_s = self.quantize(query_vals @ self.rot_head_k_tiled)
        key_vals, key_vals_s = self.quantize(key_vals @ self.rot_head_k_tiled.T)

        temp, temp_s = query_vals.type(self.acc_t()) @ key_vals.T.type(self.acc_t()), query_vals_s * key_vals_s
        temp = self.dequantize(temp, temp_s)
        temp /= math.sqrt(self.w_q.shape[-1])
        triu = torch.triu(torch.ones(temp.shape[0], temp.shape[0], dtype=torch.half), diagonal=1)
        temp[triu == 1] = -torch.inf
        temp = torch.softmax(temp, dim=1)

        # TODO: althout QuaRot doesn't seem to, we could quantize this by not dequantizing val_vals and by re-quantizing temp
        attn_vals = temp @ val_vals
        attn_vals_q, attn_vals_qs = self.quantize(attn_vals)

        out_vals_q, out_vals_qs = attn_vals_q.type(self.acc_t()) @ self.w_out.type(self.acc_t()), attn_vals_qs * self.w_out_s

        out_vals = self.dequantize(out_vals_q, out_vals_qs)

        return out_vals
    
    
    def quantize(self, x: torch.Tensor):
        return x.type(torch.float16), 1
    
    def dequantize(self, x, scale):
        return x.type(torch.float16)

    def get_rotation_and_inv(self, size):
        return torch.eye(size, dtype=torch.float16), torch.eye(size, dtype=torch.float16)
    
    def acc_t(self):
        return torch.float32


class QuantizedATTNLayer(ATTNLayer):
    def quantize(self, x):
        return utils.quantize(x)
    
    def dequantize(self, x, scale):
        return utils.dequantize(x, scale)
    
    def acc_t(self):
        return torch.int32

class QuantizedHadRotATTN(QuantizedATTNLayer):
    def get_rotation_and_inv(self, size):
        return random_rotation_almost_hadamard(size, use_hardcoded=True, run_full_orthogonality_tests=False, check_inv_max=True)
    
class QuantizedRandRotATTN(QuantizedATTNLayer):
    def get_rotation_and_inv(self, size):
        return random_rotation_almost_hadamard(size, use_hardcoded=False, run_full_orthogonality_tests=False, check_inv_max=True)
    
class QuantizedRandRotInvTATTN(QuantizedATTNLayer):
    def get_rotation_and_inv(self, size):
        q, _ = random_rotation_almost_hadamard(size, use_hardcoded=False, run_full_orthogonality_tests=False, check_inv_max=True)
        return q, q.T



class TransformerBlock(Module):
    def __init__(self, attn_layer: ATTNLayer, ffn_layer: FFNLayer) -> None:
        super().__init__()
        self.attn_layer = attn_layer
        self.ffn_layer = ffn_layer
    
    def forward(self, input: Tensor) -> Tensor:
        attn_out = input.type(torch.float16) + self.attn_layer(input).type(torch.float16)
        return attn_out.type(torch.float16) + self.ffn_layer(attn_out).type(torch.float16)

if __name__ == "__main__":
    # context_size, hidden_size, intermediate_size = 512, 1024, 2048 # 2048, 4096, 8192
    num_heads = 32
    d_v = 128
    d_k = 128
    context_size, embedding_size, intermediate_size = 1, 4096, 4096 # 512, 1024

    runs_per_test = 1

    method_names = ["basic quantization", "hadamard", "rand rot", "rand rot transp"]
    results = [[] for _ in method_names]
    attn_layer_type: list[type[ATTNLayer]] = [QuantizedATTNLayer, QuantizedHadRotATTN, QuantizedRandRotATTN, QuantizedRandRotInvTATTN]
    ffn_layer_types: list[type[FFNLayer]] = [QuantizedFFNLayer, QuantizedHadRotFFN, QuantizedRandRotFFN, QuantizedRandRotInvTFFN]

    for i in range(0, runs_per_test):
        x = torch.randn((context_size, embedding_size), dtype=torch.float16).normal_(0, 0.02 / 0.67)
        for i in range(math.ceil(context_size * embedding_size / 25000)):
            i, j = random.randrange(0, context_size), random.randrange(0, embedding_size)
            x[i, j] = random.uniform(0.3, 0.7) * (-1 + 2 * random.randint(0, 1))
        
        w_q = torch.tensor(np.random.uniform(-0.1, 0.1, (embedding_size, d_k * num_heads)), dtype=torch.float16)
        w_k = torch.tensor(np.random.uniform(-0.1, 0.1, (embedding_size, d_k * num_heads)), dtype=torch.float16)
        w_v = torch.tensor(np.random.uniform(-0.1, 0.1, (embedding_size, d_v * num_heads)), dtype=torch.float16)
        w_out = torch.tensor(np.random.uniform(-0.1, 0.1, (d_v * num_heads, embedding_size)), dtype=torch.float16)
        scaling_factor_attn = torch.tensor(np.random.uniform(-0.1, 0.1, (1, embedding_size)), dtype=torch.float16)

        w_up =   torch.tensor(np.random.uniform(-0.1, 0.1, (embedding_size, intermediate_size)), dtype=torch.float16)
        w_gate = torch.tensor(np.random.uniform(-0.1, 0.1, (embedding_size, intermediate_size)), dtype=torch.float16)
        w_down = torch.tensor(np.random.uniform(-0.1, 0.1, (intermediate_size, embedding_size)), dtype=torch.float16)
        scaling_factor_ffn = torch.tensor(np.random.uniform(-0.1, 0.1, (1, embedding_size)), dtype=torch.float16)

        truth_attn = ATTNLayer(embedding_size, d_v * num_heads, d_k * num_heads, w_q, w_k, w_v, w_out, scaling_factor_attn, num_heads)
        truth_ffn = FFNLayer(embedding_size, intermediate_size, w_gate, w_up, w_down, scaling_factor_ffn)
        truth_model = TransformerBlock(truth_attn, truth_ffn)
        truth = truth_model(x)

        for i, (attn_type, ffn_type) in enumerate(zip(attn_layer_type, ffn_layer_types)):
            model_attn = attn_type(embedding_size, d_v * num_heads, d_k * num_heads, w_q, w_k, w_v, w_out, scaling_factor_attn, num_heads)
            model_ffn = ffn_type(embedding_size, intermediate_size, w_gate, w_up, w_down, scaling_factor_ffn)
            model = TransformerBlock(model_attn, model_ffn)
            results[i].append((truth, model(x)))

    for method_name, result_list in zip(method_names, results):
        print_test_results(result_list, method_name)

    print("done")