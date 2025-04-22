import os
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch import distributed as dist
from torch import nn
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.optim import Optimizer

from fms import utils
from fms.datasets.util import SavableDataset
from fms.utils import generation, print0
from fms.utils.tokenizers import BaseTokenizer


class TrainerPlugin:
    """
    A TrainerPlugin runs once every epoch, and possibly every `steps` steps.
    It is passed relevant objects that can be used for checkpointing, logging,
    or validation.
    """

    def __init__(self, steps: Optional[int] = None):
        self.steps = steps

    def run(self, step: int, end_of_epoch: bool):
        """
        Whether or not to run this plugin on the current step.
        """
        # By default we always run for epoch ends.
        if end_of_epoch:
            return True
        # If self.steps is None, we're only recording epoch ends and this isn't one.
        if self.steps is None:
            return False
        # record every `step` steps, starting from step `step`
        if step != 0 and (step + 1) % self.steps == 0:
            return True
        return False

    @abstractmethod
    def step(
        self, epoch: int, step: int, metrics: Dict = {}, end_of_epoch: bool = False
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
        self,
        model: nn.Module,
        prompt_tokens: List[str],
        tokenizer: BaseTokenizer,
        device: Union[torch.device, str],
        steps: Optional[int] = None,
        eos_token: Optional[str] = None,
    ):
        super().__init__(steps)
        self.model = model
        self.tokenizer = tokenizer
        input_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
        self.input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        self.eos_token_id = (
            None
            if eos_token is None
            else self.tokenizer.convert_tokens_to_ids([eos_token])[0]
        )

    def step(
        self, epoch: int, step: int, metrics: Dict = {}, end_of_epoch: bool = False
    ):
        if not self.run(step, end_of_epoch):
            return
        training = self.model.training
        self.model.eval()
        with torch.no_grad():
            curtime = datetime.now().strftime("%H:%M.%S")
            prefix = f"{curtime}:{epoch:02d}"
            if step is not None:
                prefix = prefix + f":{step:04d}"

            result = generation.generate(self.model, self.input_ids, use_cache=True)
            result = generation.truncate_after_eos(result, self.eos_token_id)
            result = self.tokenizer.convert_ids_to_tokens(result)
            result = self.tokenizer.convert_tokens_to_string(result)
            print0("generated result:")
            print0(result)
        self.model.train(training)


class MetricReporter(TrainerPlugin):
    """
    A training plugin to periodically log metrics. Logs every `seconds`
    seconds by calling `writer` with the log message. A custom writer
    should accept `*args` similar to `print`.
    """

    # TODO: add optional validation dataloader and validation loss.
    # TODO: add `writer` functions that handles logging metrics to experiment
    # tracking tools such as aimstack/wandb/neptune (or add alternate plugin?)
    def __init__(
        self,
        seconds=10,
        group: Optional[dist.ProcessGroup] = None,
        prev_step: int = -1,
        cumulative_tokens: int = 0,
        device="cpu",
        writer=print0,
    ):
        super().__init__(1)
        self.seconds = seconds
        self.last_reported_time = datetime.now()
        self.tokens_seen = torch.tensor(0, device=device)
        self.cumulative_tokens = torch.tensor(cumulative_tokens, device=device)
        self.last_reported_step = prev_step
        self.cum_loss = torch.tensor(0.0, device=device)
        self.time_per_step = torch.tensor(1.0, device=device)
        self.last_step = -1
        self.group = group
        self.writer = writer

    def step(
        self, epoch: int, step: int, metrics: Dict = {}, end_of_epoch: bool = False
    ):
        if "batch_size" in metrics and "input_length" in metrics:
            self.tokens_seen += metrics["batch_size"] * metrics["input_length"]
        if "loss" in metrics:
            self.cum_loss += metrics["loss"]

        current_time = datetime.now()
        elapsed = (current_time - self.last_reported_time).total_seconds()

        if end_of_epoch:
            step = self.last_step + 1
            self.last_step = 0
            steps = step - self.last_reported_step
            self.last_reported_step = 0
        else:
            self.last_step = step
            if step == self.last_reported_step:
                return
            steps_taken = step - self.last_reported_step
            if steps_taken * self.time_per_step < self.seconds:
                return
            time_per_step = elapsed / (step - self.last_reported_step)
            self.time_per_step.fill_(time_per_step)
            steps = step - self.last_reported_step
            self.last_reported_step = step

        # group and global world are only different for HSDP
        group_world = 1 if self.group is None else self.group.size()
        global_world = 1 if not dist.is_initialized() else dist.get_world_size()
        if global_world > 1 and self.tokens_seen.device.type == "cuda":
            # these use the default process group (all nodes even if HSDP)
            dist.all_reduce(self.tokens_seen, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.cum_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.time_per_step, op=dist.ReduceOp.AVG)

        self.cumulative_tokens += self.tokens_seen

        to_report = {}
        if "loss" in metrics:
            to_report["loss"] = f"{metrics['loss']:.4f}"
        to_report["avg_loss"] = f"{self.cum_loss.item() / steps / group_world:.4f}"

        more_metrics = {
            "tok/stp": f"{self.tokens_seen.item() / steps:,.1f}",
            "s/stp": f"{self.time_per_step.item():,.3f}",
            "tok/gpu/s": f"{self.tokens_seen.item() / elapsed / global_world:,.1f}",
            "cum_tokens": f"{self.cumulative_tokens.item():,}",
        }
        if torch.cuda.is_available() and utils.has_package("pynvml"):
            nvidia_metrics = {
                "gpu_mem_use": f"{torch.cuda.memory_usage()}%",
                "gpu_utzn": f"{torch.cuda.utilization()}%",
            }
            more_metrics.update(nvidia_metrics)

        self.last_reported_time = current_time

        self.tokens_seen.fill_(0)
        self.cum_loss.fill_(0)

        to_report.update(more_metrics)
        self.writer(epoch, step, current_time, to_report)


class Checkpointer(TrainerPlugin):
    """
    A training plugin to write checkpoints.
    TODO: This will require changes to handle distributed checkpoints.

    Args:

    group: The group to checkpoint. i.e. if using hsdp, you would want to pass
            a subgroup for a single hsdp shard group.
    name: included in the file path to differentiate this particular checkpoint.
    save_dir: the base directory into which to save checkpoints.
    dataset: if set, save the state_dict of this dataset.
    steps: save a checkpoint every `steps` steps.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        dataset: Optional[SavableDataset] = None,
        save_dir: Union[str, Path] = Path("./checkpoints"),
        steps: Optional[int] = None,
        cumulative_tokens: int = 0,
        prev_step: int = -1,
        name: Optional[str] = None,
        group: Optional[dist.ProcessGroup] = None,
        device="cpu",
    ):
        super().__init__(steps)
        os.makedirs(save_dir, exist_ok=True)
        save_dir = os.path.expanduser(save_dir)
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.save_dir = Path(save_dir)
        self.cumulative_tokens = torch.tensor(cumulative_tokens, device=device)
        self.group = group
        self.name = name
        self.prev_step = prev_step
        self.device = device

    # TODO: this probably also needs to accept a dataset since we want to
    # support checkpointable datasets.
    def step(
        self, epoch: int, step: int, metrics: Dict = {}, end_of_epoch: bool = False
    ):
        if not self.run(step, end_of_epoch):
            return

        steps_since_last = step - self.prev_step
        if end_of_epoch:
            self.prev_step = -1
        else:
            self.prev_step = step

        # other metrics are per-rank so we need to aggregate cumulative tokens
        # manually
        if "batch_size" in metrics and "input_length" in metrics:
            tokens = metrics["batch_size"] * metrics["input_length"] * steps_since_last
            tokens = torch.tensor(tokens, device=self.device)
            global_world = 1 if not dist.is_initialized() else dist.get_world_size()
            if global_world > 1 and tokens.device.type == "cuda":
                # these use the default process group (all nodes even if HSDP)
                dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
            self.cumulative_tokens += tokens

        model_name = (
            self.model.__class__.__name__.lower() if self.name is None else self.name
        )

        # For FSDP, consolidate checkpointable data to rank0.
        # TODO: may also want to support distcp checkpoints which should be faster
        # to save and load but are harder to use for inference.

        is_fsdp = isinstance(self.model, FSDP)
        # For HSDP, self.group is only set for the first shard group. We only
        # need to save checkpoints for one shard group.
        if is_fsdp and self.group is None:
            return

        if is_fsdp:
            dict_type = StateDictType.FULL_STATE_DICT
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with dist.fsdp.FullyShardedDataParallel.state_dict_type(
                self.model, dict_type, cfg
            ):
                print0("Aggregating FSDP model statedict")
                model_dict = self.model.state_dict()
                print0("Aggregating optim statedict")
                optim_dict = FSDP.optim_state_dict(
                    self.model, self.optimizer, group=self.group
                )
        else:
            model_dict = self.model.state_dict()
            optim_dict = self.optimizer.state_dict()

        if step is not None:
            file = f"{model_name}_{epoch:03d}_{step + 1:05d}"
        else:
            file = f"{model_name}_{epoch:03d}_final"
        save_dir = self.save_dir

        if self.group is None:
            path = save_dir / f"{file}.pth"
            train_file = save_dir / f"{file}.train"
        else:
            path = save_dir / file
            os.makedirs(path, exist_ok=True)
            train_file = path / f"rank_{self.group.rank():02d}.train"
            path = path / f"rank_{self.group.rank():02d}.pth"

        print0("Writing model checkpoint", path)
        if is_fsdp:
            if self.group is not None and self.group.rank() == 0:
                torch.save(model_dict, path)
        else:
            torch.save(model_dict, path)

        print0("Writing training state", train_file)
        train_dict = {
            "optimizer": optim_dict,
            "epoch": epoch,
            "cumulative_tokens": self.cumulative_tokens.item(),
        }

        if step:
            train_dict |= {"step": step}

        if self.dataset is not None and hasattr(self.dataset, "state_dict"):
            dataset_sd = self.dataset.state_dict()
            train_dict |= {"dataset": dataset_sd}

        if not dist.is_initialized() or (
            self.group is not None and self.group.rank() == 0
        ):
            torch.save(train_dict, train_file)
