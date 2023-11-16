import torch
from torch import distributed as dist

from fms.models import get_model

dist.init_process_group()
torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
local_rank = dist.get_rank()

torch.manual_seed(2023)
torch.cuda.manual_seed_all(2023)

model = get_model("llama",
                  "micro",
                  device_type="cuda",
                  distributed_strategy="tp",
                  group=dist.group.WORLD
                  ).half().eval()

cmodel = torch.compile(model)
cmodel_cg = torch.compile(model, mode="reduce-overhead")

device = torch.device("cuda", local_rank)

input = torch.randint(0, 255, (1, 16), dtype=torch.int64, device=device)
with torch.no_grad():
    out_e, kv_cache_e = model(input, only_last_token=True, use_cache=True, past_key_value_states=None)
    out_e = torch.argmax(out_e).unsqueeze(0).unsqueeze(0)

    out_c, kv_cache_c = cmodel(input, only_last_token=True, use_cache=True, past_key_value_states=None)
    out_c = torch.argmax(out_c).unsqueeze(0).unsqueeze(0)

    out_cg, kv_cache_cg = cmodel_cg(input, only_last_token=True, use_cache=True, past_key_value_states=None)
    out_cg = torch.argmax(out_cg).unsqueeze(0).unsqueeze(0)

    # Compare kv caches between c and cg
    for layer_idx in range(len(kv_cache_c)):
        keys_diff = torch.max(torch.abs(kv_cache_c[layer_idx][0] - kv_cache_cg[layer_idx][0]))
        vals_diff = torch.max(torch.abs(kv_cache_c[layer_idx][1] - kv_cache_cg[layer_idx][1]))
        if keys_diff > 0 or vals_diff > 0:
            print(f"[rank {local_rank}] Max diff in layer {layer_idx} pre-clone: (k {keys_diff}, v {vals_diff})")

    for layer_idx in range(len(kv_cache_cg)):
        kv_cache_cg[layer_idx] = (kv_cache_cg[layer_idx][0].clone(), kv_cache_cg[layer_idx][1].clone())
    
    for layer_idx in range(len(kv_cache_c)):
        keys_diff = torch.max(torch.abs(kv_cache_c[layer_idx][0] - kv_cache_cg[layer_idx][0]))
        vals_diff = torch.max(torch.abs(kv_cache_c[layer_idx][1] - kv_cache_cg[layer_idx][1]))
        if keys_diff > 0 or vals_diff > 0:
            print(f"[rank {local_rank}] Max diff in layer {layer_idx} post-clone: (k {keys_diff}, v {vals_diff})")

    torch.compiler.cudagraph_mark_step_begin()
    for layer_idx in range(len(kv_cache_c)):
        keys_diff = torch.max(torch.abs(kv_cache_c[layer_idx][0] - kv_cache_cg[layer_idx][0]))
        vals_diff = torch.max(torch.abs(kv_cache_c[layer_idx][1] - kv_cache_cg[layer_idx][1]))
        if keys_diff > 0 or vals_diff > 0:
            print(f"[rank {local_rank}] Max diff in layer {layer_idx} post-step: (k {keys_diff}, v {vals_diff})")
    dist.barrier()
    out_e, _ = model(out_e, past_key_value_states=kv_cache_e, only_last_token=True, use_cache=True)
    out_c, _ = cmodel(out_c, past_key_value_states=kv_cache_c, only_last_token=True, use_cache=True)
    out_cg, _ = cmodel_cg(out_cg, past_key_value_states=kv_cache_cg, only_last_token=True, use_cache=True)

print(f"For rank {local_rank}: eager {out_e} {torch.argmax(out_e)}, compile: {out_c} {torch.argmax(out_c)}, cudagraphs: {out_cg} {torch.argmax(out_cg)}")