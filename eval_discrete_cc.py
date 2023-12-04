import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pyarrow as pa
from typing import Callable, Union

from fms.models.llama import LLaMAConfig, LLaMA

print("Job start!")

modelc = LLaMAConfig(32000, 4096, 1e-6, 32, 0, 32)
model = LLaMA(modelc)
model.eval()
model.cuda()

d = torch.load("/lustre/dwertheimer/llama_7b_ckp.pth", map_location='cpu')['model_state']
keylist = list(d.keys())
for key in keylist:
    if "dec_process" in key:
        value = d.pop(key)
        fields = key.split(".")
        fields[0] = "layers"
        d[".".join(fields)] = value

model.load_state_dict(d, strict=False)
model.cuda()

print("Model loaded!")

# data = torch.load("/lustre/dwertheimer/sap-v2-test_2_encode.pth")

data = []
datapath = "/lustre/bluepile-processing/rel0_5/tokens_llama2/lang=en/dataset=commoncrawl/part-00000-0ad865cb-a6d4-4037-bc29-79b5c6097d0b-c000-attempt_202306281109048020644254530093061_0204_m_000000_158265.arrow"
with pa.ipc.open_file(pa.memory_map(datapath)) as reader:
    for i in range(200):
        test = reader.get_batch(i)['tokens']
        line = test.tolist()
        # Take first half
        line = line[:len(line)//2]
        # Shorten if necessary
        line = line[:3096]
        data.append(line)


from fms.modules.layernorm import LayerNormParameterized

class Speculator(nn.Module):
    def __init__(self, emb_dim=4096, vocab_size=32000, n_heads=4):
        super().__init__()
        self.nheads = n_heads
        self.emb_dim = emb_dim
        self.vsize = vocab_size
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.w_in = nn.Parameter(torch.empty(emb_dim * 2, int((emb_dim * 2.6875) // 256) * 256 * 2))  # d 2z
        self.a = nn.GELU()
        self.w_out = nn.Parameter(torch.empty(int((emb_dim * 2.6875) // 256) * 256, emb_dim * n_heads))  # z hd
        self.ln = LayerNormParameterized(emb_dim, elementwise_shift=False, elementwise_scale=True)
        self.head = nn.Parameter(torch.empty(n_heads, emb_dim, vocab_size))  # h d v
        self.reset_params()

    def reset_params(self):
        nn.init.trunc_normal_(self.w_in, 0, (1 / 2.6875) ** (1 / 6) / self.emb_dim**0.5)
        nn.init.trunc_normal_(self.w_out, 0, (1 / 2.6875) ** (1 / 6) / self.emb_dim**0.5)
        nn.init.trunc_normal_(self.head, 0, 1 / (self.vsize * self.emb_dim) ** 0.25)
        nn.init.trunc_normal_(self.emb.weight, 0, 1 / self.emb_dim**0.5)

    def forward(self, x, i):
        # x: b n d
        z = torch.cat([x, self.emb(i)], dim=2)
        z, g = z.matmul(self.w_in).chunk(2, dim=2)
        z = z * self.a(g)
        z = z.matmul(self.w_out).view(x.size(0), x.size(1), self.nheads, self.emb_dim)  # b n h d
        z = z + x.unsqueeze(2)
        z = self.ln(z)
        z = torch.einsum("bnhd,hdv->bnhv", z, self.head)
        return z  # b n h v
    
    
def get_topk_tree(logits, k, thresh):
    # probs: b h v
    n_adds = logits.size(1)
    probs = logits.chunk(n_adds, dim=1) # h of b v
    probs = [probs[i][:,0,:thresh[i]].softmax(1) for i in range(n_adds)] # h of b t'
    # Generate probabilities for all combos of predictions
    probtable = [
        probs[i].view(
            *([-1] + [1]*i + [thresh[i]] + [1]*(n_adds-i-1))
        ).expand(
            *([-1] + thresh)
        )
        for i in range(n_adds)
    ]
    probtable = torch.stack(probtable, 0).prod(0) # b v v v...
    psize = probtable.size()
    probtable = probtable.view(psize[0],-1)
    # Fetch top-k most probable tree nodes
    v,i = probtable.topk(k, dim=1) # b k
    i = torch.stack(torch.unravel_index(i, psize[1:]), 1)
    # v: b k
    # i: b h k
    return v,i


def speculative_generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: torch.LongTensor,
    smallmodel: torch.nn.Module,
    max_seq_len: int = 2048,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 25,
    num_beams: int = 1,
):
    threshes = [7,5,3]
    do_sample = False
    batched = False
    if num_beams != 1:
        raise NotImplementedError("generate() does yet not support beam search")
    if type(input_ids) == torch.Tensor:
        if input_ids.dim() != 1:
            batched = True
    else:
        raise RuntimeError("generate() requires a tensor of token ids as the prefix")

    if not batched:
        input_ids = input_ids.unsqueeze(0)

    
    # cudastats = torch.cuda.memory_summary(device=torch.cuda.current_device(), abbreviated=True)
    # print(cudastats)

    result = input_ids
    next_input = input_ids
    kwargs = dict()
    kwargs["past_key_value_states"] = None
    kwargs["use_cache"] = True

    output = model(input_ids[:,:-1], include_embeds=True, **kwargs)
    _, past_key_value_states, embeds = output
    embeds = embeds[:,-1:]
    kwargs["past_key_value_states"] = past_key_value_states
    next_input = next_input[:,-1:]
    del output
    torch.cuda.empty_cache()
    # cudastats = torch.cuda.memory_summary(device=torch.cuda.current_device(), abbreviated=True)
    # print("post_first_forward", cudastats)
    
    n_gen = 0
    n_steps = 0
    n_kv_s = past_key_value_states
    while n_gen < max_new_tokens:
        # print("n_gen:", n_gen)
        # print(result.shape)
        
        n_steps += 1
        input_ids = next_input[:, -max_seq_len:]
        
        probs = smallmodel(embeds, input_ids).squeeze(1) # b h v
        probs, topk = probs.topk(max(threshes), dim=2) # b h 5
        n_adds = smallmodel.nheads
        # for i in range(n_adds):
        #     print(f"Topk@{i+1}:", decode_obo(topk[0,i]))
        
        # Build probability table
        topk_v, topk_i = get_topk_tree(probs, top_k, threshes)
        
        # Assemble batch of tree branches
        adds = topk.gather(2, topk_i).transpose(1,2) # b k h
        adds = adds[0] # For now, non-batching and take only first b entry
        input_ids = torch.cat([input_ids.expand(top_k,1), adds], dim=-1) 
#         print("Speculations:")
#         for i in range(top_k):
#             print(decode_obo(input_ids[i]))
        
        mask = torch.ones(input_ids.size(1),input_ids.size(1)+n_kv_s[0][0].size(2), device=input_ids.device)
        mask = mask.tril(diagonal=mask.size(1)-mask.size(0))
        mask = mask.unsqueeze(0).unsqueeze(0).log()
        
#         input_ids = input_ids[0].unsqueeze(0).expand(25,-1)
        
        output = model.forward(input_ids, include_embeds=True, mask=mask, **kwargs)
        
        logits, past_key_value_states, embeds = output
        logits = logits[:, -n_adds-1:, :]

        if do_sample:
            # get logits from last value in sequence and scale
#             logits = logits / temperature
#             if top_k:
#                 v, _ = torch.topk(logits, top_k)
#                 logits[logits < v[:, [-1]]] = -float("inf")

#             probs = F.softmax(logits, dim=-1)
#             next_val = torch.multinomial(probs, num_samples=1)
            assert False
        else:
            next_vals = torch.argmax(logits, dim=-1)
        
        # Check correctness of smallmodel predictions
        test = input_ids.roll(-1, 1).eq(next_vals).cumprod(1)
        
        n_correct = test.sum(1).clamp(0,n_adds)
        best_guess = n_correct.argmax()
        
#         for i in range(top_k):
#             print(decode_obo(input_ids[i]), decode_obo(next_vals[i]), test[i].tolist(), n_correct[i].item())
        
        next_vals = next_vals[best_guess].unsqueeze(0)
        n_correct = n_correct[best_guess]
        embeds = embeds[best_guess].unsqueeze(0)
        
        # print("Verification:", decode_obo(input_ids[best_guess]), decode_obo(next_vals), n_correct.item())
        
        # Toss any wrong smallmodel outputs
        next_vals = next_vals[:,:n_correct+1]
        n_gen += n_correct.item()+1
        embeds = embeds[:,n_correct].unsqueeze(1)
            
        n_wrong = n_adds - n_correct
        # kv updates are required for torch.compile with
        # mode='reduce-overhead'
        n_kv_s = []
        for layer_idx in range(len(past_key_value_states)):
            n_kv_s.append([])
            for tensor_idx in range(2):
                base = past_key_value_states[layer_idx][tensor_idx]
                new = past_key_value_states[layer_idx][tensor_idx+2][best_guess].unsqueeze(0)
                if n_wrong > 0:
                    new = new[:,:,:-n_wrong]
                base = torch.cat([base, new], dim=2)
                n_kv_s[layer_idx].append(
                    base.clone(memory_format=torch.contiguous_format).detach().cuda()
                )
                # torch._dynamo.mark_dynamic(n_kv_s[layer_idx][tensor_idx], 2)
        kwargs["past_key_value_states"] = n_kv_s

        result = torch.cat((result, next_vals), dim=-1)
        # print("Updated output:", decode_obo(result))
        # print()

        next_input = next_vals[:,-1].unsqueeze(-1)

    if not batched:
        result = result[0]
    return result, n_steps

test = Speculator(n_heads=3)
test.load_state_dict(torch.load("/lustre/dwertheimer/results/llama-speculator/gen3/discrete-n2_PAhsdp_ws8_mbs8_sl4096_pr0_vFMS4e2972ae_jid2389_sysAwsEfa0/checkpoints/step_40000_ckp.pth", map_location="cpu")["model_state"])
test.cuda()

print("Speculator ready!")

torch.cuda.empty_cache()
steps = {}
outs = []
for k in [2, 5, 10, 25]:
    steps[k] = []
    for j,seq in enumerate(data):
        inp = torch.IntTensor(seq).cuda()
        with torch.no_grad():
            out, nsteps = speculative_generate(model, inp, test, 4096, 100, top_k=k)
        if k==5:
            outs.append(out.squeeze().tolist())
        steps[k].append(nsteps)
        print(f"Ex {j}, topk={k}: 100 tokens in {nsteps} steps.")
        # print("    ", out.squeeze().tolist()[-100:])

torch.save(steps, "/lustre/dwertheimer/results/llama-speculator/gen3/discrete-n2_PAhsdp_ws8_mbs8_sl4096_pr0_vFMS4e2972ae_jid2389_sysAwsEfa0/steps_for_100_at_k.pth")
torch.save(outs, "/lustre/dwertheimer/results/sandbox/llama_7b_sap_outputs.pth")