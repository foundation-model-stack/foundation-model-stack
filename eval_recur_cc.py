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
        self.emb = nn.ModuleList([nn.Embedding(vocab_size, emb_dim) for _ in range(n_heads)])
        self.proj = nn.ModuleList([nn.Linear(emb_dim * 2, emb_dim, bias=False) for _ in range(n_heads)])
        self.head = nn.ModuleList([nn.Linear(emb_dim, vocab_size, bias=False) for _ in range(n_heads)])
        self.ln = nn.ModuleList(
            [LayerNormParameterized(emb_dim, elementwise_shift=True, elementwise_scale=True) for _ in range(n_heads)]
        )
        self.a = nn.GELU()
        self.reset_params()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding) or isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, 0, 1 / self.emb_dim**0.5)
            elif isinstance(m, LayerNormParameterized):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def generate_tree(self, state, ind, topk=[5,4,3], k=25):
        # state: b 1 d
        # ind: b 1
        b = state.size(0)
        out = torch.LongTensor(b,1,0) # b k h
        log_probs = torch.zeros(b,1) # b k
        assert len(topk)==self.nheads
        for i in range(self.nheads):
            z = self.emb[i](ind) # b k d
            z = torch.cat([state, z], dim=2) # b k 2d
            state = self.a(self.ln[i](self.proj[i](z))) # b k d
            probs = F.log_softmax(self.head[i](state), dim=2) # b k v
            probs, preds = probs.topk(topk[i], dim=2) # b k k'
            out = out.unsqueeze(2).expand(-1,-1,topk[i],-1) # b k k' h
            out = torch.cat([out, preds.unsqueeze(3)], dim=3) # b k k' h+1
            
            # Prep for next round
            out = out.view(b, -1, i+1) # b kk' h+1
            state = state.unsqueeze(2).expand(-1,-1,topk[i],-1) # b k k' d
            state = state.reshape(b, -1, state.size(3)) # b kk' d
            ind = preds.view(b, -1) # b kk'
            log_probs = log_probs.unsqueeze(2).expand(b,-1,topk[i]) # b k k'
            log_probs = log_probs.add(probs).reshape(b, -1) # b kk'
            
        best_guesses = log_probs.topk(k, dim=1)[1] # b k
        
        return out.gather(1, best_guesses.unsqueeze(2).expand(-1,-1,self.nheads)) # b k h
            

    def forward(self, state, inds):
        # state: b n d
        # inds: b n+2 (..., pred token, n+2, n+3)
        out = []
        for i in range(self.nheads):
            h_inds = inds[:, i : i + state.size(1)]
            z = self.emb[i](h_inds)  # b n d
            z = torch.cat([state, z], dim=2)  # b n 2d
            state = self.a(self.ln[i](self.proj[i](z)))  # b n d
            out.append(self.head[i](state))  # b n v
        return torch.stack(out, dim=2)  # b n h v
    
    
def speculative_generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: torch.LongTensor,
    smallmodel: torch.nn.Module,
    max_seq_len: int = 2048,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 24,
    num_beams: int = 1,
    threshes = [4,3,2]
):
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
    
    n_gen = 0
    n_steps = 0
    n_kv_s = past_key_value_states
    while n_gen < max_new_tokens:
        n_steps += 1
        input_ids = next_input[:, -max_seq_len:]
        
        adds = smallmodel.generate_tree(embeds, input_ids, threshes, top_k)
#         for i in range(adds.size(1)):
#             print(decode_obo(adds[0,i]))
        
        n_adds = smallmodel.nheads
#         probs = smallmodel(embeds, input_ids).squeeze(1) # b h v
#         probs, topk = probs.topk(max(threshes), dim=2) # b h 5
#         for i in range(n_adds):
#             print(f"Topk@{i+1}:", decode_obo(topk[0,i]))
        
#         # Build probability table
#         topk_v, topk_i = get_topk_tree(probs, top_k, threshes)
        
#         # Assemble batch of tree branches
#         adds = topk.gather(2, topk_i).transpose(1,2) # b k h
        adds = adds[0] # For now, non-batching and take only first b entry
        input_ids = torch.cat([input_ids.expand(top_k,1), adds], dim=-1) 
#         print("Speculations:")
#         for i in range(top_k):
#             print(decode_obo(input_ids[i]))
        
        mask = torch.ones(input_ids.size(1),input_ids.size(1)+n_kv_s[0][0].size(2))
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
        n_gen += n_correct+1
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
                    base.clone(memory_format=torch.contiguous_format).detach()
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
test.load_state_dict(torch.load("/lustre/dwertheimer/results/llama-speculator/gen3/recurrent-n2_PAhsdp_ws8_mbs8_sl4096_pr0_vFMS9fa84965_jid2395_sysAwsEfa0/checkpoints/step_40000_ckp.pth", map_location="cpu")["model_state"])
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
            out, nsteps = speculative_generate(model, inp, test, 4096, 100, top_k=k, threshes=[5,3,2])
        if k==5:
            outs.append(out.squeeze().tolist())
        steps[k].append(nsteps)
        print(f"Ex {j}, topk={k}: 100 tokens in {nsteps} steps.")
        # print("    ", out.squeeze().tolist()[-100:])

torch.save(steps, "/lustre/dwertheimer/results/llama-speculator/gen3/recurrent-n2_PAhsdp_ws8_mbs8_sl4096_pr0_vFMS9fa84965_jid2395_sysAwsEfa0/steps_for_100_at_k.pth")
# torch.save(outs, "/lustre/dwertheimer/results/sandbox/llama_7b_sap_outputs.pth")