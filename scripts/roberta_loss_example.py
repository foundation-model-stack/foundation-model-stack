import torch
import torch.distributed as dist

from fms.models import get_model


torch.set_default_dtype(torch.float16)


print("loading model")

model = get_model(
    "roberta",
    "base",
    # model_path=args.model_path,
    device_type="cuda",
    group=dist.group.WORLD,
)

model = torch.compile(model)

input = torch.randint(0, 100, (3, 512), dtype=torch.int64, device="cuda")

output = model(input)

loss_fn = torch.nn.CrossEntropyLoss()

loss = loss_fn(output, torch.zeros((3, 50265), device="cuda", dtype=torch.int64))

loss.backward()
