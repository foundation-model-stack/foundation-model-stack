import argparse
import functools
import logging
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from fms.models.llama import LLaMA, LLaMABlock, LLaMAConfig
from fms.training.args import (
    add_ckp_args,
    add_config_args,
    add_profiler_args,
    add_training_args,
    add_vocab_args,
    add_llama_args,
    validate_arg_tokens,
    validate_args,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.profiler import ProfilerAction, ProfilerActivity
from torch.profiler import schedule as prof_schedule

from fms.datasets import dataset as fmdata 
from fms.utils.from_closed import (
    get_datasets_and_weights,
    get_local_rank,
    get_rank,
    get_world_size,
    run_rank_n,
    maybe_profile,
    trace_handler,
    setup_distributed,
    pcount,
    human_readable,
)
from fms.utils.io_ops_closed import(
    human_readable_report_and_log,
    Llama_Checkpointer,
)


loglevel = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=loglevel)


parser = argparse.ArgumentParser(description="Llama training on IBM NLP data")

# Config args
config_arg_group = parser.add_argument_group("General Config")
add_config_args(config_arg_group)

# Profiler args
prof_arg_group = parser.add_argument_group("Profiler Config")
add_profiler_args(prof_arg_group)

# Checkpoint args
ckp_arg_group = parser.add_argument_group("Checkpoint Config")
add_ckp_args(ckp_arg_group)
parser.add_argument(
    "--flexible_load", default=False, action="store_true", help="Disable strict loading for single-file checkpoints?"
)

# Vocab args
vocab_arg_group = parser.add_argument_group("Vocab Config")
add_vocab_args(vocab_arg_group)

# Model args
model_arg_group = parser.add_argument_group("Model Config")
add_llama_args(model_arg_group)

# Training args
training_arg_group = parser.add_argument_group("Training Config")
add_training_args(training_arg_group)
parser.add_argument(
    "--prompt_len",
    type=int,
    default=1,
    help="Number of words to ingest before making trainable predictions",
)
training_arg_group.add_argument("--decay_interval", type=int, default=0, help="Number of steps for cosine annealing")

args = parser.parse_args()

# Define reporting fns

run_rank_n(os.makedirs)(args.log_path, exist_ok=True)
report = functools.partial(human_readable_report_and_log, args.log_path)


def sync_report(*args, **kwargs):
    dist.barrier()
    report(*args, **kwargs)


# Handle args
rank = get_rank()
local_rank = get_local_rank()
world_size = get_world_size()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_device(local_rank)

args = validate_args(args, world_size)
validate_arg_tokens(args, "pad,sep", allow_no_pad=True)

if args.decay_interval == 0:
    args.decay_interval = args.num_steps

datasets, weights = get_datasets_and_weights(args.datasets, args.dataset_weights)
is_testrun = args.testrun_data_index != -100

report(
    "Starting training run:",
    world_size=world_size,
    torch_version=torch.__version__,
    **vars(args),
)

# Model setup policies
mp_policy = (
    None
    if not args.bf16
    else MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        _module_classes_to_ignore=[],
    )
)
wrapping_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={LLaMABlock})
model_sharding_strategies = {
    "fsdp": ShardingStrategy.FULL_SHARD,
    "hsdp": ShardingStrategy.HYBRID_SHARD,
    "ddp": ShardingStrategy.NO_SHARD,
}
model_sharding_strategy = model_sharding_strategies[args.parallel_mode]

# reset stats
torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())


# Training loop
def train_func(args):
    emu_factor = args.simulated_gpus // world_size
    if emu_factor > 1:
        report(
            WARNING="Grad accumulation has not been fully tested with FSDP. Correctness for this run is NOT guaranteed!"
        )
    report("Setting up NCCL...")
    setup_distributed()

    bsize = args.b_size
    effective_bsize = args.b_size * args.simulated_gpus
    start_step = 0
    tokens_seen = 0

    # Model
    report("Constructing model...")
    model = LLaMAConfig(
        src_vocab_size=args.vocab,
        emb_dim=args.emb_dim,
        norm_eps=args.norm_eps,
        nheads=args.n_heads,
        kvheads=args.kv_heads,
        nlayers=args.n_layers,
        pad_id=args.pad_token,
        hidden_grow_factor=args.hidden_grow_factor,
        multiple_of=args.multiple_of,
        activation_fn=args.act_fn,
        p_dropout=args.dropout,
        max_expected_seq_len=args.seq_len,
    )  # Change up your model architecture here!
    model = LLaMA(model)

    for m in model.modules():
        if args.recompute_policy == "ffn" and isinstance(m, LLaMABlock):
            m.ff_sub_layer.recompute = True
        elif args.recompute_policy == "all" and isinstance(m, LLaMABlock):
            m.recompute = True

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    num_params = pcount(model)

    # Wrap model
    report(f"Applying wrapper for parallelization mode={args.parallel_mode}...")
    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mp_policy,
        sharding_strategy=model_sharding_strategy,
        device_id=local_rank,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    model.to(local_rank)
    signature = None
    sync_report("Model created!", num_params=num_params, signature=signature)

    # Apply activation checkpointing
    if args.recompute_policy != "none":
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        check_fn = lambda submodule: hasattr(submodule, "recompute") and submodule.recompute
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
        sync_report(msg="Activation checkpointing enabled!")

    # Optimizers
    report(lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1, lr=args.lr, betas=(args.beta1, args.beta2))

    schedule = lambda x: min(
        1 - (1 - min(x, args.warmup_interval) / args.warmup_interval) ** 2,  # parabolic anneal
        0.1 + 0.5 * (1 - 0.1) * (1 + math.cos(min(x, args.decay_interval) / args.decay_interval * math.pi)),
    )  # cos anneal to 10% @ 250k

    updated_schedule = lambda x: schedule(x + start_step)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, updated_schedule)

    def causal_lm(data_seq, prompt_len=1):
        """
        Perform causal language modeling by right-shifting the input sequence.
        Sets first prompt_len tokens to be ignored by the loss. Assumes inputs start with BOS.
        """
        data_seq = torch.IntTensor(data_seq)
        targ = data_seq.clone()[1:]
        data_seq = data_seq[:-1]
        targ[:prompt_len] = -100
        return data_seq, targ

    # Dataloaders
    base_scalable = fmdata.Scalable_Shard_Dataset if args.oversample_weights else fmdata.Scalable_Sampling_Dataset
    data = base_scalable(
        args.data_path,
        fmdata.Shard_Doc_Dataset if args.disable_streaming else fmdata.Streaming_Doc_Dataset,
        rank,
        world_size,
        args.sep_token,
        trainsplit=1 if not is_testrun else 0.05,
        is_val=False,
        min_length=3,
        datasets=datasets,
        weights=weights if not is_testrun else None,
        testrun_data_index=args.testrun_data_index,
        seed=args.seed,
        verbose=(rank == 0),
        n_logical_shards=args.logical_shards,
    )
    data = fmdata.Buffer_Dataset(
        data,
        [args.seq_len + 1],
        bos_token=args.sep_token,
        pack_hard=True,
    )
    data = fmdata.Preload_Buffer_Dataset(data, 10000 if not is_testrun else 100)
    data = fmdata.Preprocess_Dataset(data, causal_lm)

    report("Constructing datasets...", effective_bsize=effective_bsize)
    train_data = data
    train_loader = iter(torch.utils.data.DataLoader(train_data, num_workers=0, batch_size=bsize))
    sync_report("Datasets constructed!")

    # Open checkpointer
    checkpointer = Llama_Checkpointer(
        args.log_path,
        args.num_ckps,
        args.parallel_mode,
        report_fn=report,
    )

    # Load from checkpoint
    report("Loading checkpoint...")
    model, optimizer_, train_loader_, start_step, tokens_seen = checkpointer.load(
        model,
        None if args.drop_optimizer else optimizer,
        None if args.drop_dataset else train_loader,
        path=args.ckp_path,
        reset_stepcount=args.reset_stepcount,
        strict=not args.flexible_load,
    )
    if not args.drop_optimizer:
        optimizer = optimizer_
    if not args.drop_dataset:
        train_loader = train_loader_
    signature = None
    report("Checkpoint loaded!", signature=signature)

    # Override loaded optim hyperparams with the current values
    for g in optimizer.param_groups:
        g["initial_lr"] = args.lr
        g["betas"] = (args.beta1, args.beta2)

    # Clear memory
    torch.cuda.empty_cache()

    # Train
    report(
        "Beginning training! Depending on dataset size and config, initial few steps may be slow.",
        num_steps=args.num_steps - start_step,
    )
    model.train()
    losstracker = torch.zeros(1).to(local_rank)
    cliptracker = 0
    trackertracker = 0
    valloss = -1
    start = time.time()
    loop_start = time.time()
    step = start_step - 1
    with maybe_profile(
        use_profiler=args.profile,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule(
            skip_first=args.profile_skip,
            wait=args.profile_wait,
            warmup=args.profile_warmup,
            active=args.profile_active,
            repeat=1,
        ),
        on_trace_ready=functools.partial(trace_handler, output_path=args.log_path, extra_name=str(rank)),
        with_stack=args.profile_remove_stack,
        profile_memory=args.profile_remove_memory,
        record_shapes=args.profile_remove_shapes,
    ) as prof:
        for step in range(start_step, args.num_steps):
            optimizer.zero_grad()
            verbose = step == start_step or step % args.save_interval == 0
            if verbose:
                sync_report("Workers synchronized!")
            for ministep in range(emu_factor):
                inp, labels = next(train_loader)
                if verbose and ministep == 0:
                    sync_report("Collected data")
                inp = inp.to(local_rank)
                labels = labels.to(local_rank).long()  # long needed for loss fn (6/2/23)
                dist.barrier()
                pred = model(inp)
                loss = loss_fn(pred.view(-1, args.vocab), labels.view(-1)).div(emu_factor)
                #             z_loss = .001*pred.pow(2).mean() # Naive implementation of PaLM Z-loss
                losstracker += loss.item()
                dist.barrier()

                if verbose and ministep == 0:
                    sync_report("Got through forward pass", ntok=inp.size(1))
                if (ministep + 1) == emu_factor:
                    loss.backward()
                    track_norm = model.clip_grad_norm_(args.clip_th).item()
                    if track_norm > args.clip_th:
                        cliptracker += 1
                else:
                    with nullcontext():
                        loss.backward()
                if verbose and ministep == 0:
                    sync_report("Got through backward pass")

                # deallocating GPU memory for the pred tensor
                del pred

            optimizer.step()
            scheduler.step()
            if verbose:
                sync_report("Got through one step")
            tokens_seen += effective_bsize * (args.seq_len + 1)
            trackertracker += 1

            # Report training loss and speed
            if (step + 1) % args.report_interval == 0:
                dist.all_reduce(losstracker, op=dist.ReduceOp.SUM)
                trainloss = losstracker.item() / trackertracker / world_size
                elapsed_time = time.time() - loop_start
                elapsed_tokens = (step - start_step) * effective_bsize * (args.seq_len + 1)
                sync_report(
                    step=step + 1,
                    lr=optimizer.param_groups[0]["lr"],
                    gnorm=track_norm,
                    tokens_seen=tokens_seen,
                    percentclip=cliptracker / trackertracker,
                    trainloss=trainloss,
                    speed=(time.time() - start) / trackertracker,
                    elapsed_time=elapsed_time,
                    tok_per_sec_per_gpu=int((elapsed_tokens) / world_size / elapsed_time),
                    tok_per_day=int((elapsed_tokens / elapsed_time) * 3600 * 24),
                )
                losstracker.zero_()
                cliptracker = 0
                trackertracker = 0
                start = time.time()

            # Checkpoint model
            if (step + 1) % args.save_interval == 0:
                if args.profile and prof.current_action == ProfilerAction.RECORD_AND_SAVE:
                    report("You are profiling a checkpointing step, be careful about it!")
                signature = None
                cudastats = torch.cuda.memory_summary(device=torch.cuda.current_device(), abbreviated=True)
                report("Starting distributed checkpoint save...", signature=signature, mem_summary=cudastats)
                overwritten = checkpointer.save(
                    step + 1,
                    model,
                    optimizer,
                    train_loader,
                    tokens_seen=tokens_seen,
                    loss=trainloss,
                    signature=signature,
                )
                sync_report("Checkpoint written", ckp_time=human_readable(time.time() - start))
                if overwritten:
                    report("Checkpoint", overwritten, "dumped")
                torch.cuda.empty_cache()

                model.train()
                start = time.time()

            # Profiler step
            if args.profile:
                prof.step()

    signature = None
    sync_report(msg="Writing final checkpoint", step=step + 1, signature=signature)
    checkpointer.save_single_file(
        step + 1,
        model,
        tokens_seen=tokens_seen,
        loss=trainloss,
        signature=signature,
    )
    report("Final checkpoint written!")
    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


start = time.time()
train_func(args)
report("Job Complete!", total_hours=human_readable((time.time() - start) / 3600))
