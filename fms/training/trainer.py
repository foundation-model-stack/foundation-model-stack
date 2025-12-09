from contextlib import nullcontext
from typing import List

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from fms.training.plugins import TrainerPlugin
from fms.utils import print0


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model: nn.Module, loss_fn: nn.Module) -> None:
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor, label: torch.Tensor, **kwargs):
        output = self.model(x, **kwargs)
        return self.loss_fn(output, label)


def __one_step(
    loss_model: ModelWithLoss,
    input: torch.Tensor,
    label: torch.Tensor,
    grad_scaler,
    **kwargs,
):
    device_type = input.device.type
    autocast = (
        torch.autocast(device_type=device_type)
        if grad_scaler is not None
        else nullcontext()
    )
    with autocast:
        loss = loss_model(input, label, **kwargs)

    if grad_scaler is not None:
        grad_scaler.scale(loss).backward()
    else:
        loss.backward()
    return loss


def __optimize(model, optimizer, grad_scaler):
    if grad_scaler is not None:
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    optimizer.zero_grad()


def __one_epoch(
    loss_model: ModelWithLoss,
    optimizer: Optimizer,
    data: DataLoader,
    device,
    epoch: int,
    prev_step: int,
    plugins: List[TrainerPlugin],
    accum_iters: int = 1,
):
    print0("Epoch", epoch)
    loss_model.model.train()

    grad_scaler = None
    # grad_scaler = torch.cuda.amp.GradScaler()

    if data.sampler is not None and isinstance(data.sampler, DistributedSampler):
        data.sampler.set_epoch(epoch)

    optimized = False
    optimizer.zero_grad()

    highest_step = prev_step
    batch_size = -1
    input_length = -1
    for step, (input, label, kwargs) in enumerate(data):
        step = prev_step + step + 1
        highest_step = step

        batch_size = input.shape[0]
        input_length = input.shape[1]

        input = input.to(device)
        label = label.to(device)
        kwargs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }

        loss = __one_step(
            loss_model,
            input,
            label,
            grad_scaler,
            **kwargs,
        )
        if (step + 1) % accum_iters == 0:
            __optimize(loss_model.model, optimizer, grad_scaler)
            optimized = True
        else:
            optimized = False

        metrics = {
            "loss": loss,
            "batch_size": batch_size,
            "input_length": input_length,
        }
        for plugin in plugins:
            plugin.step(epoch, step, metrics)
    if not optimized:
        __optimize(loss_model.model, optimizer, grad_scaler)
    metrics = {
        "batch_size": batch_size,
        "input_length": input_length,
    }
    for plugin in plugins:
        plugin.step(epoch, step=highest_step, metrics=metrics, end_of_epoch=True)


def train(
    model,
    optimizer,
    dataloader: DataLoader,
    device,
    loss_fn: nn.Module,
    start_epoch: int = 0,
    epochs: int = 1,
    prev_step: int = -1,
    trainer_plugins: List[TrainerPlugin] = [],
    grad_accum_iters: int = 1,
    compile_loss: bool = False,
    compile_backend: str = "inductor",
):
    loss_model = ModelWithLoss(model, loss_fn)
    if compile_loss:
        loss_model.compile(backend=compile_backend)

    for epoch in range(start_epoch, start_epoch + epochs):
        __one_epoch(
            loss_model,
            optimizer,
            dataloader,
            device,
            epoch,
            prev_step,
            trainer_plugins,
            accum_iters=grad_accum_iters,
        )
