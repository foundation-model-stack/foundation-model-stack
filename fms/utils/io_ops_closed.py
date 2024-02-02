import json
import os
import shutil
import time
from pathlib import Path

import torch
from torch.distributed import barrier
from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from fms.utils.from_closed import get_local_rank, get_rank, get_world_size, run_rank_n, get_latest, get_oldest, human_readable, human_readable_time


@run_rank_n
def print_rank_n(*args, **kwargs) -> None:
    """
    wrap print method to run on only rank 0
    """
    print(*args, **kwargs)


@run_rank_n
def report(*args, **kwargs):
    print(*args)
    for key in kwargs:
        print(key, "=", kwargs[key])


@run_rank_n
def log(logdir, *args, **kwargs):
    """
    Drop args, write kwargs to log json in logdir
    """
    if len(kwargs) > 0:
        log_path = os.path.join(logdir, "log_main.json")
        if not os.path.exists(log_path):
            print("Starting new log file at", log_path)
            with open(log_path, "w") as f:
                json.dump([kwargs], f)
        else:
            with open(log_path, "r+") as f:
                loaded_results = json.load(f)
                f.seek(0)
                json.dump(loaded_results + [kwargs], f)


@run_rank_n
def report_and_log(logdir, *args, **kwargs):
    report(*args, **kwargs)
    log(logdir, *args, **kwargs)

 
@run_rank_n
def human_readable_report_and_log(logdir, *args, **kwargs):
    """
    Report and log, but massage numerical report values to be more human-readable. Relies on field name heuristics.
    """
    new_kwargs = {}
    for key in kwargs:
        if isinstance(kwargs[key], str):
            new_kwargs[key] = kwargs[key]
        elif "speed" in key or "time" in key:
            new_kwargs[key] = human_readable_time(kwargs[key])
        elif "tok" in key or "param" in key:
            new_kwargs[key] = human_readable(kwargs[key], 2)
        elif "loss" in key or "norm" in key:
            new_kwargs[key] = human_readable(kwargs[key], 3)
        else:
            new_kwargs[key] = kwargs[key]
    report(*args, **new_kwargs)
    if logdir is not None:
        log(logdir, *args, **kwargs)


class Checkpointer:
    """
    Manages the checkpoint directory. Saves new checkpoints and deletes old ones after the specified number are written.
    ...
    Args
    ----
    ckpdir : str
        Absolute path to desired save location. Creates a new 'checkpoints/' subfolder at that location.
    n_to_save : int
        Number of volatile checkpoints to maintain at any given time.
    parallel_mode : str
        Write sharded folder ckps (when sharded: 'fsdp' or 'hsdp') or unsharded file ckps (when sharded: 'ddp')
    verbose : bool
        Print progress update(s) when saving?

    Methods
    -------
    save : keyword args -> str | None
        Saves dictionary of keyword arg key/value pairs to specified checkpoint directory, deleting old checkpoints
        as necessary. If a checkpoint is deleted, returns the filename of that checkpoint.
    """

    def __init__(
        self,
        ckpdir,
        n_to_save,
        parallel_mode,
        verbose=False,
        process_group=None,
        replicate_group=None,
    ):
        self.max_ckps = n_to_save
        self.ckp_path = os.path.join(ckpdir, "checkpoints/")
        os.makedirs(self.ckp_path, exist_ok=True)
        self.p_mode = parallel_mode
        self.verbose = verbose
        self.process_group = process_group
        self.replicate_group = replicate_group

    def _print(self, x):
        if self.verbose:
            print(x)

    def do_save(self, rank, local_rank, shard_group, replicate_group):
        # TODO: Distributed writing contingent upon the following fix: https://github.com/pytorch/pytorch/issues/104081
        # if not is_dist:
        #     return (rank == local_rank)
        # else:
        #     a = rank % shard_group.size()
        #     b = rank // shard_group.size()
        #     return True if a % replicate_group.size() == b else False
        return rank == local_rank

    def write_dcp(self, state_dict, loader_state, process_group, save_name, rank):
        os.makedirs(save_name, exist_ok=True)
        writer = FileSystemWriter(save_name, single_file_per_rank=True)

        if state_dict is not None:
            self._print(f"Writing state dict on rank={rank}")
            save_state_dict(
                state_dict=state_dict, storage_writer=writer, process_group=process_group, planner=DefaultSavePlanner()
            )
            self._print(f"Finished writing state dict on rank={rank}")
        if loader_state is not None:
            self._print(f"Writing data loader state on rank={rank}...")
            loader_state.save_to_path(save_name)
            self._print(f"Finished writing data loader state on rank={rank}...")

    def save_dcp(
        self,
        step,
        model_state,
        optimizer_state,
        loader_state,
        final=False,
        permanent=False,
        **kwargs,
    ):
        # Note: metadata kwargs cannot contain any of:
        # (step, model_state, optimizer_state, loader_state, final, permanent)
        is_sharded_dp = self.p_mode == "fsdp" or self.p_mode == "hsdp"
        is_ddp = self.p_mode == "ddp"
        is_distributed = is_ddp or is_sharded_dp

        suffix = "_final" if final else ""
        c = "_" if is_sharded_dp else "."
        filetype = c + "ckp" if final or permanent else c + "tmp"
        save_name = os.path.join(self.ckp_path, "step_" + str(step) + suffix + filetype)
        rank = get_rank()
        local_rank = get_local_rank()

        self._print(f"Calling save_dcp on rank={rank}, local_rank = {local_rank}, path = {save_name}")
        state_dict = {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
        }
        if self.p_mode == "fsdp":
            self.write_dcp(state_dict, loader_state, self.process_group, save_name, rank)
            if rank == 0:
                metadata = kwargs
                metadata["step"] = step
                torch.save(metadata, os.path.join(save_name, "metadata.pth"))

        elif self.p_mode == "hsdp":
            if self.do_save(
                rank,
                local_rank,
                shard_group=self.process_group,
                replicate_group=self.replicate_group,
            ):
                self.write_dcp(state_dict, loader_state, self.process_group, save_name, rank)
            else:
                self.write_dcp(None, loader_state, None, save_name, rank)
            if rank == 0:
                self._print(f"Saving metadata file on rank={rank}")
                metadata = kwargs
                metadata["step"] = step
                torch.save(metadata, os.path.join(save_name, "metadata.pth"))
                self._print(f"Done saving metadata file on rank={rank}")

        else:
            if not is_ddp or rank == 0:
                metadata = kwargs
                metadata["step"] = step
                metadata["model_state"] = model_state
                if optimizer_state is not None:
                    metadata["optimizer_state"] = optimizer_state
                if loader_state is not None:
                    metadata["loader_state"] = loader_state
                torch.save(metadata, save_name)

        # Clean old checkpoints. Barrier to keep synchronization correct.
        if is_distributed:
            barrier()
        file_to_remove = None
        removing_node = not is_distributed or rank == 0
        if removing_node and len([x for x in os.listdir(self.ckp_path) if c + "tmp" in x]) > self.max_ckps:
            file_to_remove = Path(get_oldest(self.ckp_path, qualifier=lambda x: c + "tmp" in x))
            if is_sharded_dp:
                shutil.rmtree(file_to_remove)
            else:
                file_to_remove.unlink()
        if is_distributed:
            barrier()
        return file_to_remove


class Llama_Checkpointer:
    """
    Manages the checkpoint directory. Saves new checkpoints and deletes old ones after the specified number are written.
    Also handles loading and saving of checkpoints in sharded and unsharded formats.
    Assumes model and optimizer inputs are in FSDP.
    ...
    Args
    ----
    ckpdir : str
        Absolute path to desired save location. Creates a new 'checkpoints/' subfolder at that location.
    n_to_save : int
        Number of volatile checkpoints to maintain at any given time.
    parallel_mode : str
        Write sharded folder ckps (when sharded: 'fsdp' or 'hsdp') or unsharded file ckps (when sharded: 'ddp')
    report_fn : Callable or None
        Optional function for reporting or logging status updates. Expected to handle arbitrary *args, **kwargs.

    Methods
    -------
    save : keyword args -> str | None
        Saves dictionary of keyword arg key/value pairs to specified checkpoint directory, deleting old checkpoints
        as necessary. If a checkpoint is deleted, returns the filename of that checkpoint.
    """

    def __init__(
        self,
        ckpdir,
        n_to_save,
        parallel_mode,
        report_fn=None,
    ):
        self.max_ckps = n_to_save
        self.ckp_path = os.path.join(ckpdir, "checkpoints/")
        os.makedirs(self.ckp_path, exist_ok=True)
        self.p_mode = parallel_mode
        assert parallel_mode in ["fsdp", "hsdp", "ddp"]
        self.report = self._dummy_report if report_fn is None else report_fn

    def _dummy_report(self, *args, **kwargs):
        pass

    def _cleanup(self):
        # Clean old checkpoints. Barrier to keep synchronization correct.
        file_to_remove = None
        if get_rank() == 0 and len([x for x in os.listdir(self.ckp_path) if "tmp" in x]) > self.max_ckps:
            ckp_to_remove = Path(get_oldest(self.ckp_path, qualifier=lambda x: "tmp" in x))
            if os.path.is_file(ckp_to_remove):
                ckp_to_remove.unlink()
            else:
                shutil.rmtree(ckp_to_remove)
        return file_to_remove

    def _do_save(self, rank, local_rank):  # , shard_group, replicate_group):
        if self.p_mode == "hsdp":
            return rank == local_rank
        else:
            return True
        # TODO: Distributed writing contingent upon the following fix: https://github.com/pytorch/pytorch/issues/104081
        # if not is_dist:
        #     return (rank == local_rank)
        # else:
        #     a = rank % shard_group.size()
        #     b = rank // shard_group.size()
        #     return True if a % replicate_group.size() == b else False
        # shard_group = model.process_group
        # replicate_group = model.__inter_node_state.process_group

    def _write(self, state_dict, loader_state, process_group, save_name, rank):
        os.makedirs(save_name, exist_ok=True)
        writer = FileSystemWriter(save_name, single_file_per_rank=True)

        if state_dict is not None:
            self.report(f"Writing state dict on rank={rank}")
            save_state_dict(
                state_dict=state_dict, storage_writer=writer, process_group=process_group, planner=DefaultSavePlanner()
            )
            self.report(f"Finished writing state dict on rank={rank}")
        if loader_state is not None:
            self.report(f"Writing data loader state on rank={rank}...")
            loader_state.save_to_path(save_name)
            self.report(f"Finished writing data loader state on rank={rank}...")

    def _validate_ckp_path(self, path):
        """Interpret path to appropriate checkpoint. If found, return modified path. If not found, return None."""
        # Does path exist and is it non-empty?
        if os.path.exists(path):
            # Is this a file?
            if os.path.isfile(path):
                return path
            # Is this a sharded directory?
            elif "metadata.pth" in os.listdir(path):
                return path
            # Is this a path to a set of checkpoints?
            elif len(os.listdir(path)) > 0:
                latest = get_latest(path)
                if os.path.isfile(latest):
                    return latest
                elif "metadata.pth" in os.listdir(latest):
                    return latest
        return None

    def load(self, model, optimizer, dataloader, path="", reset_stepcount=False, strict=True):
        """
        Handle checkpoint loading for model/optimizer/dataloader from given path, according to arguments.
        Defaults to save path for locating an appropriate checkpoint. If a path is provided, will use
        it only if no appropriate checkpoint is found in the save path (in which case it's a job restart).
        Reset_stepcount manually resets optimizer and dataloader states, and stat tracking.
        Strict determines whether to use strict loading or not FOR SINGLEFILE LOADING ONLY.
        Returns model, optimizer, dataloader, current step, and current tokens seen.
        """
        if self._validate_ckp_path(self.ckp_path) is not None:
            path = self.ckp_path
            reset_stepcount = False
        load_path = self._validate_ckp_path(path)
        if load_path is None:
            self.report(f"No valid checkpoint detected at {path}, starting from scratch.")
            return model, optimizer, dataloader, 0, 0
        else:
            self.report(f"Prior checkpoint {load_path} detected.")
            model_load_time = time.time()
            if os.path.isfile(load_path):
                checkpoint_data = torch.load(load_path, map_location="cpu")
                model.load_state_dict(checkpoint_data.get("model_state"), strict=strict)
                model.to(get_local_rank())
                self.report(
                    f"Checkpoint {load_path} is a single-file checkpoint containing only a model. Optimizer and dataloader are from scratch.",
                    model_load_time=human_readable(time.time() - model_load_time),
                )
                return model, optimizer, dataloader, 0, 0
            else:
                # Load model
                with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                    state_dict = model.state_dict()
                    model_ckp = {"model_state": state_dict}
                    load_state_dict(
                        state_dict=model_ckp, storage_reader=FileSystemReader(load_path), planner=DefaultLoadPlanner()
                    )
                    model.load_state_dict(model_ckp["model_state"])
                model.to(get_local_rank())
                self.report(model_load_time=human_readable(time.time() - model_load_time))
                step = 0
                ntok = 0
                # Load metadata
                if not reset_stepcount:
                    metadata = torch.load(os.path.join(load_path, "metadata.pth"))
                    step = metadata.get("step", 0)
                    ntok = metadata.get("tokens_seen", 0)
                    self.report("Metadata loaded", start_step=step, n_tokens_seen=human_readable(ntok))
                # Load optimizer
                if optimizer is not None:
                    optim_load_time = time.time()
                    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                        optim_state = load_sharded_optimizer_state_dict(
                            model_state_dict=model.state_dict(),
                            optimizer_key="optimizer_state",
                            storage_reader=FileSystemReader(load_path),
                        )
                    flattened_osd = FSDP.optim_state_dict_to_load(model, optimizer, optim_state["optimizer_state"])
                    optimizer.load_state_dict(flattened_osd)
                    self.report(optimizer_load_time=human_readable(time.time() - optim_load_time))
                else:
                    self.report("Skipping optimizer load, no optimizer provided.")
                # Load dataset
                if dataloader is not None:
                    data_load_time = time.time()
                    dataloader._dataset.load_from_path(load_path)
                    self.report(dataset_load_time=human_readable(time.time() - data_load_time))
                else:
                    self.report("Skipping dataset load, no dataloader provided.")
                return model, optimizer, dataloader, step, ntok

    def save(
        self,
        step,
        model,
        optimizer,
        dataloader,
        **kwargs,
    ):
        # Note: metadata kwargs cannot contain any of:
        # (step, model, optimizer, dataloader)
        rank = get_rank()

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model_state = model.state_dict()
            optim_state = FSDP.sharded_optim_state_dict(model, optimizer, group=model.process_group)
        dataloader_state = dataloader._dataset

        save_name = os.path.join(self.ckp_path, "step_" + str(step) + "_ckp")
        state_dict = {"model_state": model_state, "optimizer_state": optim_state}
        if self._do_save(rank, get_local_rank()):
            self._write(state_dict, dataloader_state, model.process_group, save_name, rank)
        else:
            self._write(None, dataloader_state, None, save_name, rank)
        if rank == 0:
            metadata = kwargs
            metadata["step"] = step
            torch.save(metadata, os.path.join(save_name, "metadata.pth"))

        return self._cleanup()

    def save_single_file(
        self,
        step,
        model,
        **kwargs,
    ):
        # Note: metadata kwargs cannot contain any of:
        # (step, model)
        save_name = os.path.join(self.ckp_path, "step_" + str(step) + "_ckp.pth")

        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            model_state = model.state_dict()
        if get_rank() == 0:
            metadata = kwargs
            metadata["step"] = step
            metadata["model_state"] = model_state
            torch.save(metadata, save_name)

        return self._cleanup()
