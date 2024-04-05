from contextlib import nullcontext
from typing import List, Optional

import torch
from torch import nn
from torch.cuda import amp
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from fms.training.plugins import TrainerPlugin
from fms.utils import print0


def __one_step(
    model: nn.Module,
    input: torch.Tensor,
    label: torch.Tensor,
    loss_fn: nn.Module,
    grad_scaler: Optional[amp.GradScaler],
):
    autocast = amp.autocast if grad_scaler is not None else nullcontext
    with autocast():
        output = model(input)
        loss = loss_fn(output, label)

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
    model: nn.Module,
    optimizer: Optimizer,
    data: DataLoader,
    device,
    loss_fn,
    epoch: int,
    prev_step: int,
    plugins: List[TrainerPlugin],
    accum_iters: int = 1,
):
    print0("Epoch", epoch)
    model.train()

    grad_scaler = None
    # grad_scaler = torch.cuda.amp.GradScaler()

    if data.sampler is not None and isinstance(data.sampler, DistributedSampler):
        data.sampler.set_epoch(epoch)

    optimized = False
    optimizer.zero_grad()

    highest_step = prev_step
    for step, (input, label) in enumerate(data):
        step = prev_step + step + 1
        highest_step = step

        batch_size = input.shape[0]
        input_length = input.shape[1]

        input = input.to(device)
        label = label.to(device)

        loss = __one_step(model, input, label, loss_fn, grad_scaler)
        if (step + 1) % accum_iters == 0:
            __optimize(model, optimizer, grad_scaler)
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
        __optimize(model, optimizer, grad_scaler)
    for plugin in plugins:
        plugin.step(epoch, step=highest_step, end_of_epoch=True)


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
):
    for epoch in range(start_epoch, start_epoch + epochs):
        __one_epoch(
            model,
            optimizer,
            dataloader,
            device,
            loss_fn,
            epoch,
            prev_step,
            trainer_plugins,
            accum_iters=grad_accum_iters,
        )
