from fms.models import get_model
import torch

roberta_model = get_model(
    architecture="roberta",
    variant="base",
    device_type="cuda",
)

compiled_model = torch.compile(roberta_model)

model_input = torch.randint(0, 50_000, (3, 256), dtype=torch.int32, device="cuda")
model_output = compiled_model(model_input)
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(model_output, torch.zeros((3, 50265), device="cuda", dtype=torch.int64))
loss.backward()