import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2,5"

import torch
import torch.distributed as dist
import json

from fms.models import get_model


# ======================================================================================
# PART I: load GPTQ model with FMS
# ======================================================================================

ARCHITECTURE="gpt_bigcode"
VARIANT="ibm.20b"
GPTQ_MODEL_PATH = "/home/afasoli/storage/granite_gptq/granite-20b-code-instruct-v1/gptq_model-4bit-128g"
# GPTQ_MODEL_PATH = "/home/afasoli/LLM_GPTQ/storage_granite_gptq/granite-20b-code-instruct-v1/base_1600btok_1013"
GPTQ_CONFIG = {"linear_type": "gptq", "group_size": 128, "use_marlin": False, "disable_exllama": True, "disable_exllamav2": False}
# GPTQ_CONFIG = {"linear_type": "torch_linear",}
IS_DISTRIBUTED = int(os.environ["GPTQ_DISTRIBUTED"]) == 1


# get env variables
world_rank = int(os.getenv("RANK", 0))
local_rank = world_rank
world_size = int(os.getenv("WORLD_SIZE", 1))
local_size = int(os.getenv("LOCAL_WORLD_SIZE", 1))

device = torch.device(f'cuda:{world_rank}')
dist_backend = 'nccl' # gloo for cpu, nccl for gpu
torch.set_default_device(device)

if IS_DISTRIBUTED:
    # initialize the process group
    torch.distributed.init_process_group(backend=dist_backend, rank=world_rank, world_size=world_size)

    print(f"Rank {world_rank} | Start run")
    group = dist.new_group(list(range(world_size)))
    dist.barrier()

torch.set_default_dtype(torch.float16)
model = get_model(
    architecture=ARCHITECTURE,
    variant=VARIANT,
    model_path=GPTQ_MODEL_PATH,
    device_type=device,
    source="hf",
    distributed_strategy="tp" if IS_DISTRIBUTED else None,
    group=dist.group.WORLD if IS_DISTRIBUTED else None,
    linear_config=GPTQ_CONFIG,
)

dist.barrier()
print(f"Rank {world_rank} | after get_model")

if dist.get_rank() == 0:
    print(model)