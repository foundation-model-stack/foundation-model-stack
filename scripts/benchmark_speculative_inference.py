import argparse
import os
import time

import torch

from fms.models import llama, get_model
from fms.modules.speculator import Speculator
from fms.utils.cache.paged import PagedKVCacheManager
from fms.utils.generation import speculative_generate, generate
from torch import distributed as dist


parser = argparse.ArgumentParser(description="Script to run inference on a LLaMA model")
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the directory containing HF LLaMa weights",
)
parser.add_argument(
    "--speculator_path",
    type=str,
    required=True,
    help="Path to the checkpoint containing speculator weights (single .pth file, not HF weights)",
)
parser.add_argument(
    "--model_path_source",
    type=str,
    default="meta",
    help="The source format of the model weights. E.g. meta, hf",
)
parser.add_argument(
    "--architecture",
    type=str,
    default="llama",
    help="The model architecture to benchmark",
)
parser.add_argument(
    "--variant",
    type=str,
    default="7b",
    help="The model variant (configuration) to benchmark. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--checkpoint_sharding",
    type=str,
    default=None,
    help="type of weight sharding. E.g. tensor-parallel (tp), None",
)
parser.add_argument(
    "--data_path",
    type=str,
    required=True,
    help="Path to the set of prompts (single .pth file)",
)
parser.add_argument(
    "--bsize", type=int, default=1, help="Number of sequences per batch"
)
parser.add_argument(
    "--output_path",
    type=str,
    default="",
    help="Path to save the final scores, if needed",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument("--device_type", type=str, default="cuda")

args = parser.parse_args()

def pad_prompt(prompt, pad_len):
    to_pad = pad_len - len(prompt)
    if to_pad == 0:
        return prompt

    pad_id = 0
    pad_ids = [pad_id] * to_pad
    pads = torch.tensor(pad_ids, dtype=prompt.dtype, device=device)
    return torch.cat((pads, prompt))


print("Job start!")
local_rank = int(os.getenv("LOCAL_RANK", 0))
device = torch.device(args.device_type, local_rank)
torch.cuda.set_device(device)

torch.set_default_device(device)
torch.set_default_dtype(torch.half)

if args.distributed:
    dist.init_process_group()

model = get_model(
    args.architecture,
    args.variant,
    model_path=args.model_path,
    source=args.model_path_source,
    device_type=args.device_type,
    checkpoint_sharding=args.checkpoint_sharding,
    norm_eps=1e-6,
)
model.eval()

print("Model loaded!")

print("initiating cache")
kv_cache = PagedKVCacheManager(
    model.config.nlayers,
    model.config.nheads,
    model.config.emb_dim,
    total_num_gpu_blocks=2000,
    tensor_parallel_size=dist.get_world_size() if args.distributed else 1,
    dtype=torch.get_default_dtype(),
)

data = torch.load(args.data_path)
print("Data prepared!")

test = Speculator(n_predict=3, emb_dim=model.config.emb_dim)
test.load_state_dict(
    torch.load(
        args.speculator_path,
        map_location="cpu",
    )["model_state"]
)
test.cuda()

print("Speculator ready!")

torch.cuda.empty_cache()
bsize = args.bsize
steps = {}
outs = []
for k in [0, 2, 5, 10, 25]:
    steps[k] = []
    for j in range(len(data) // bsize):
        seqs = data[j * bsize : j * bsize + bsize]
        max_seq = max(len(line) for line in seqs)
        inp = [torch.IntTensor(line).cuda() for line in seqs]
        with torch.no_grad():

            if k == 0:
                inp = torch.stack([pad_prompt(t, max_seq) for t in inp], dim=0)
                start_time = time.time()
                out, nsteps, generation_time = generate(
                    model,
                    inp,
                    max_new_tokens=100,
                    max_seq_len=4096,
                    do_sample=False,
                    use_cache=True,
                    kv_cache_manager=kv_cache,
                )
            else:
                start_time = time.time()
                out, nsteps, generation_time = speculative_generate(
                    model,
                    inp,
                    test,
                    new_tokens=100,
                    max_seq_len=4096,
                    top_k=k,
                    kv_cache_manager=kv_cache,
                )
            end_time = time.time()
            total_time = end_time - start_time
        if k == 5:
            outs += [line.squeeze().tolist() for line in out]

        for i in range(bsize):
            num_generated = len(out[i]) - len(inp[i])
            print(f"Ex {j*bsize+i}, topk={k}: {num_generated} tokens in {nsteps} steps.")
            print(f"--- avg per token: {total_time / len(out[i])}, avg per new token: {generation_time / num_generated}")
            steps[k].append([nsteps * 100 / num_generated, total_time / len(out[i]), generation_time / num_generated])

if len(args.output_path) > 0:
    torch.save(steps, os.path.join(args.output_path, "steps_for_100_at_k.pth"))
