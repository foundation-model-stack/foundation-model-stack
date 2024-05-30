import json
import math
import random
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from quantized_transformer_layer import ATTNLayer, FFNLayer, QuantizedATTNLayer, QuantizedHadRotATTN, QuantizedRandRotATTN, QuantizedRandRotInvTATTN, QuantizedFFNLayer, QuantizedHadRotFFN, QuantizedRandRotFFN, QuantizedRandRotInvTFFN, TransformerBlock
from utils import print_test_results
from safetensors.torch import load_file
import sentencepiece as spm

num_heads = 32
d_v = 128
d_k = 128
context_size, embedding_size, intermediate_size = 1, 4096, 4096 # 512, 1024
ffn_intermediate_size = 11008

class Transformer(Module):
    def __init__(self, weights, attn_type: type[ATTNLayer], ffn_type: type[FFNLayer], num_blocks=32) -> None:
        super().__init__()

        self.blocks: list[TransformerBlock] = []
        for i in range(num_blocks):
            w_q = weights[f"model.layers.{i}.self_attn.q_proj.weight"].T.type(torch.float16)
            w_k = weights[f"model.layers.{i}.self_attn.k_proj.weight"].T.type(torch.float16)
            w_v = weights[f"model.layers.{i}.self_attn.v_proj.weight"].T.type(torch.float16)
            w_o = weights[f"model.layers.{i}.self_attn.o_proj.weight"].T.type(torch.float16)
            input_norm_weights = weights[f"model.layers.{i}.input_layernorm.weight"].type(torch.float16)
            ffn_norm_weights = weights[f"model.layers.{i}.post_attention_layernorm.weight"].type(torch.float16)
            w_up = weights[f"model.layers.{i}.mlp.up_proj.weight"].T.type(torch.float16)
            w_down = weights[f"model.layers.{i}.mlp.down_proj.weight"].T.type(torch.float16)
            w_gate = weights[f"model.layers.{i}.mlp.gate_proj.weight"].T.type(torch.float16)
            self.blocks.append(TransformerBlock(
                attn_type(embedding_size, d_v * num_heads, d_k * num_heads, w_q, w_k, w_v, w_o, input_norm_weights, num_heads=num_heads),
                ffn_type(embedding_size, ffn_intermediate_size, w_gate, w_up, w_down, ffn_norm_weights)
            ))

    def forward(self, input: Tensor) -> Tensor:
        print("running forward...")
        x = input
        for i, block in enumerate(self.blocks):
            x = block.forward(x)
            print(f"ran layer {i}", flush=True)
        return x


runs_per_test = 1

method_names = ["basic quantization", "hadamard", "rand rot", "rand rot transp"]
results = [[] for _ in method_names]
attn_layer_type: list[type[ATTNLayer]] = [QuantizedATTNLayer, QuantizedHadRotATTN, QuantizedRandRotATTN, QuantizedRandRotInvTATTN]
ffn_layer_types: list[type[FFNLayer]] = [QuantizedFFNLayer, QuantizedHadRotFFN, QuantizedRandRotFFN, QuantizedRandRotInvTFFN]

weights_map_file = 'triton/kernels/quarot/test_models/granite-7b-lab/model.safetensors.index.json'
with open(weights_map_file, 'r') as f:
    weights_map: dict = json.load(f)['weight_map']
weights = {}
for value in set(weights_map.values()):
    temp = load_file(f"triton/kernels/quarot/test_models/granite-7b-lab/{value}")
    weights.update(temp)

tokenizer_model_file = 'triton/kernels/quarot/test_models/granite-7b-lab/tokenizer.model'
sp = spm.SentencePieceProcessor(model_file=tokenizer_model_file)
hello = sp.encode('hello') # 22172
assert len(hello) == 1

embedding_weights = weights[f"model.embed_tokens.weight"].type(torch.float16)
test_val = embedding_weights[hello].type(torch.float16)

for i in range(0, runs_per_test):
    truth_model = Transformer(weights, ATTNLayer, FFNLayer)
    x = test_val
    truth = truth_model(x)

    for i, (attn_type, ffn_type) in enumerate(zip(attn_layer_type, ffn_layer_types)):
        model = Transformer(weights, attn_type, ffn_type)
        results[i].append((truth, model(x)))

for method_name, result_list in zip(method_names, results):
    print_test_results(result_list, method_name, embedding_weights, sp)

print("done")