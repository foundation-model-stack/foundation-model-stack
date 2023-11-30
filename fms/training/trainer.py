from contextlib import nullcontext
from typing import List, Optional
import torch
from torch.cuda import amp
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Optimizer
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
    return loss.item()


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
    loss_fn,
    epoch: int,
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
    for step, (input, label) in enumerate(data):
        loss = __one_step(model, input, label, loss_fn, grad_scaler)
        if (step + 1) % accum_iters == 0:
            __optimize(model, optimizer, grad_scaler)
            optimized = True
        else:
            optimized = False

        metrics = {
            "loss": loss,
            "batch_size": input.shape[0],
            "input_length": input.shape[1],
        }
        for plugin in plugins:
            plugin.step(model, optimizer, epoch, metrics, step)
    if not optimized:
        __optimize(model, optimizer, grad_scaler)
    for plugin in plugins:
        plugin.step(model, optimizer, epoch)


def train(
    model,
    optimizer,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    start_epoch=0,
    epochs: int = 1,
    trainer_plugins: List[TrainerPlugin] = [],
    grad_accum_iters: int = 1,
):
    for epoch in range(start_epoch, start_epoch + epochs):
        __one_epoch(
            model,
            optimizer,
            dataloader,
            loss_fn,
            epoch,
            trainer_plugins,
            accum_iters=grad_accum_iters,
        )
