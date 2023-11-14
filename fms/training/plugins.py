from abc import abstractmethod
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch import distributed as dist
from torch import nn
from fms import utils
from fms.utils import print0, generation


class TrainerPlugin:
    """
    A TrainerPlugin runs once every epoch, and possibly every `steps` steps.
    It is passed relevant objects that can be used for checkpointing, logging,
    or validation.
    """

    def __init__(self, steps=None):
        self.steps = steps

    def run(self, step):
        """
        Whether or not to run this plugin on the current step.
        """
        # if step is None, we're at an epoch end not an intermediate step.
        # By default we always run for epoch ends.
        if step is None:
            return True
        # If self.steps is None, we're only recording epoch ends and this isn't one.
        if self.steps is None:
            return False
        # record every `step` steps
        if (step + 1) % self.steps == 0:
            return True
        return False

    @abstractmethod
    def step(
        self,
        model: nn.Module,
        optimizer,
        epoch: int,
        metrics: Dict = {},
        step: Optional[int] = None,
    ):
        """
        This method is called on every step of training, or with step=None
        at the end of each epoch. Implementations can use the passed in
        parameters for validation, checkpointing, logging, etc.

        Args:
        model: The model being trained.
        step: The step in training, re-starting from zero each epoch. None at
                 epoch end.
        metrics: a dictionary of metrics that might be useful for
                logging/reporting. E.g. 'loss'. Specific metrics subject
                to change.
        """
        pass


class InferenceValidator(TrainerPlugin):
    """
    A training plugin to print the results of running inference on a given prompt.
    """

    def __init__(
        self, prompt_tokens: List[str], tokenizer, device, steps=None, eos_token=None
    ):
        super().__init__(steps)
        self.tokenizer = tokenizer
        input_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
        self.input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        self.eos_token_id = (
            None
            if eos_token is None
            else self.tokenizer.convert_tokens_to_ids([eos_token])[0]
        )

    def step(
        self,
        model: nn.Module,
        optimizer,
        epoch: int,
        metrics: Dict = {},
        step: Optional[int] = None,
    ):
        if not self.run(step):
            return
        training = model.training
        model.eval()
        with torch.no_grad():
            curtime = datetime.now().strftime("%H:%M.%S")
            prefix = f"{curtime}:{epoch:02d}"
            if step is not None:
                prefix = prefix + f":{step:04d}"

            result = generation.generate(model, self.input_ids, use_cache=True)
            result = generation.truncate_after_eos(result, self.eos_token_id)
            result = self.tokenizer.convert_ids_to_tokens(result)
            result = self.tokenizer.convert_tokens_to_string(result)
            print0("generated result:")
            print0(result)
        model.train(training)


class MetricReporter(TrainerPlugin):
    """
    A training plugin to periodically log metrics. Logs every `seconds`
    seconds by calling `writer` with the log message. A custom writer
    should accept `*args` similar to `print`.
    """

    # TODO: add optional validation dataloader and validation loss.
    # TODO: add `writer` functions that handles logging metrics to experiment
    # tracking tools such as aimstack/wandb/neptune (or add alternate plugin?)
    def __init__(self, seconds=10, writer=print0):
        super().__init__(1)
        self.seconds = seconds
        self.last_reported_time = datetime.now()
        self.tokens_seen = 0
        self.last_reported_step = -1
        self.cum_loss = 0
        self.last_step = -1
        self.writer = writer

    def step(
        self,
        model: nn.Module,
        optimizer,
        epoch: int,
        metrics: Dict = {},
        step: Optional[int] = None,
    ):
        if "batch_size" in metrics and "input_length" in metrics:
            self.tokens_seen += metrics["batch_size"] * metrics["input_length"]
        if "loss" in metrics:
            self.cum_loss += metrics["loss"]

        current_time = datetime.now()
        elapsed = (current_time - self.last_reported_time).total_seconds()

        if step is None:
            step = self.last_step + 1
            self.last_step = 0
            steps = step - self.last_reported_step
            self.last_reported_step = 0
        else:
            self.last_step = step
            if elapsed < self.seconds or step == self.last_reported_step:
                return
            steps = step - self.last_reported_step
            self.last_reported_step = step

        # TODO: aggregate these per-rank statistics when training with
        # distributed. e.g.: dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        # for loss, tokens_seen, etc.
        to_report = {}
        if "loss" in metrics:
            to_report["loss"] = f"{metrics['loss']:.4f}"
        to_report["avg_loss"] = f"{self.cum_loss / steps:.4f}"

        more_metrics = {
            "tok/stp": f"{self.tokens_seen / steps:,.1f}",
            "tok/s": f"{self.tokens_seen / elapsed:,.1f}",
        }
        if torch.cuda.is_available() and utils.has_package("pynvml"):
            nvidia_metrics = {
                "gpu_mem_use": f"{torch.cuda.memory_usage()}%",
                "gpu_utzn": f"{torch.cuda.utilization()}%",
            }
            more_metrics.update(nvidia_metrics)

        self.last_reported_time = current_time
        self.tokens_seen = 0
        self.cum_loss = 0
        to_report.update(more_metrics)
        self.writer(epoch, step, current_time, to_report)


class Checkpointer(TrainerPlugin):
    """
    A training plugin to write checkpoints.
    TODO: This will require changes to handle distributed checkpoints.
    """

    def __init__(
        self,
        save_dir: str | Path = Path("./checkpoints"),
        steps: Optional[int] = None,
        name: Optional[str] = None,
        group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__(steps)
        os.makedirs(save_dir, exist_ok=True)
        save_dir = os.path.expanduser(save_dir)
        self.save_dir = Path(save_dir)
        self.group = group
        self.name = name

    # TODO: this probably also needs to accept a dataset since we want to
    # support checkpointable datasets.
    # TODO: end of epoch checkpoints should probably consolidate to one
    # rank when using FSDP.
    def step(
        self,
        model: nn.Module,
        optimizer,
        epoch: int,
        metrics: Dict = {},
        step: Optional[int] = None,
    ):
        if not self.run(step):
            return

        model_name = (
            model.__class__.__name__.lower() if self.name is None else self.name
        )
        model_dict = model.state_dict()
        if step is not None:
            file = f"{model_name}_{epoch:03d}_{step+1:05d}"
        else:
            file = f"{model_name}_{epoch:03d}_final"
        save_dir = self.save_dir

        if self.group is None:
            path = save_dir / f"{file}.pth"
            train_file = f"{file}.train"
        else:
            path = save_dir / file
            os.makedirs(path, exist_ok=True)
            train_file = f"rank_{self.group.rank():02d}.train"
            path = path / f"rank_{self.group.rank():02d}.pth"
        print0("Writing checkpoint", path)
        torch.save(model_dict, path)

        optim_dict = optimizer.state_dict()
        train_dict = {"optimizer": optim_dict, "epoch": epoch}
        torch.save(train_dict, train_file)
