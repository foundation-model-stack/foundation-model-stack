import argparse
import math
import os
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
from torch import distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fms import datasets, models
from fms.training import plugins as trainplugins
from fms.training import trainer
from fms.utils import fusion, print0, tokenizers

#
# This is a fairly minimal training/tuning script for causal language models.
#
# Example usage for fine tuning llama 7b on the alpaca dataset on slurm:
# srun --gres=gpu:2 --cpus-per-task=24 --mem=512G --unbuffered --gres-flags=enforce-binding \
#       torchrun --nproc_per_node=2 scripts/train_causal.py --architecture=llama --variant=7b \
#       --tokenizer=~/models/tokenizer.model --model_path=~/models/7B/ --output_path=./tuned/ \
#       --report_steps=10 --checkpoint_format=meta --distributed=fsdp
#
# Simple example of pre-training on tokens stored in arrow files (pre-processed to length 4096):
#
# export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda:/usr/local/cuda/targets/x86_64-linux/lib/:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:$LD_LIBRARY_PATH
# export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
# srun --gres=gpu:8 --cpus-per-task=96 -N 8 --mem=1T --unbuffered --gres-flags=enforce-binding \
#       --exclusive bash -c 'torchrun --nnodes=$SLURM_NTASKS --nproc_per_node=8 --node_rank=$SLURM_NODEID \
#       --master_addr=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1` \
#       scripts/train_causal.py --variant=7b --tokenizer=~/models/tokenizer.model \
#       --device_type=cuda --distributed=hsdp  --dataset_style=arrow \
#       --dataset_path=file:///lustre/users/bvaughan/data/'
#
# Logs output like:
# 0 19 2024-03-11 19:58:35.773642 {'loss': '7.6250', 'avg_loss': '8.3703', 'tok/stp': '524,288.0', 's/stp': '2.227', 'tok/gpu/s': '3,679.1', 'gpu_mem_use': '0%', 'gpu_utzn': '0%'}
# 0 28 2024-03-11 19:58:57.736138 {'loss': '7.6250', 'avg_loss': '7.6424', 'tok/stp': '524,288.0', 's/stp': '2.439', 'tok/gpu/s': '3,357.0', 'gpu_mem_use': '46%', 'gpu_utzn': '100%'}
# 0 37 2024-03-11 19:59:17.431584 {'loss': '7.4688', 'avg_loss': '7.5139', 'tok/stp': '524,288.0', 's/stp': '2.189', 'tok/gpu/s': '3,743.4', 'gpu_mem_use': '37%', 'gpu_utzn': '100%'}
#
# use sbatch for longer running training jobs.
#

parser = argparse.ArgumentParser(description="Script to train or tune a model")

# parameters for model and tokenizer initialization and loading
parser.add_argument(
    "--architecture",
    type=str,
    default="llama",
    help="The model architecture to tune",
)
parser.add_argument(
    "--variant",
    type=str,
    default="micro",
    help="The model variant (configuration) to tune. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--num_classes",
    type=str,
    default="micro",
    help="The model variant (configuration) to tune. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--checkpoint_format",
    type=str,
    default=None,
    help="E.g. meta, hf, or None. Resuming from a checkpoint will be `None` but fine tuning may initially load from another source",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Start/resume from a checkpoint a the given path",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    default="char_tokenizer",
    help="Name of or path to the tokenizer (e.g. ~/tokenizer.model)",
)
parser.add_argument(
    "--device_type",
    type=str,
    default=None,
    help="Device type. If not specified check for availability of cuda, mps, then cpu",
)
parser.add_argument(
    "--default_dtype",
    type=str,
    default=None,
    choices=["bf16", "fp16", "fp32"],
    help="If set to one of the choices, overrides the model checkpoint weight format by setting the default pytorch format",
)
parser.add_argument(
    "--distributed",
    type=str,
    default=None,
    help="The strategy used for distributing the model. E.g. fsdp, ddp, tp, mp. Default None",
)
parser.add_argument(
    "--unfuse_weights",
    action="store_true",
    help="If set to True, this will unfuse any fused weight modules that support the unfuse_weights method",
)

# Dataset arguments
parser.add_argument(
    "--dataset_style",
    type=str,
    default="instruction",
    help="'instruction' uses alpaca-formatted json. 'text' points to a raw text file. See `--dataset_path` to specify a file or URL",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
    help="The path or URI refering to data to use in tuning or training",
)

# Metrics/reporting/output
parser.add_argument(
    "--report_steps",
    type=int,
    default=500,
    help="Run the reporting function every report_steps steps",
)
parser.add_argument(
    "--checkpoint_steps",
    type=int,
    default=None,
    help="If > 0, Checkpoint every checkpoint_steps steps within a single epoch.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="./checkpoints",
    help="Output directory to save trained model checkpoints",
)
parser.add_argument(
    "--compile",
    action="store_true",
    help="Whether to compile the model and optimizer.",
)
parser.add_argument(
    "--compile_backend",
    type=str,
    default="inductor",
    help="What backend to use for compilation.",
)
parser.add_argument(
    "--head_only",
    action="store_true",
    help="Whether to only tune the head or the whole model.",
)

# Training/tuning parameters
parser.add_argument(
    "--epochs", type=int, default=2, help="Number of epochs to train/tune"
)
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument(
    "--grad_accum_steps",
    type=int,
    default=1,
    help="Number of steps to accumulate gradients before applying",
)


args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
rank = int(os.getenv("RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))

# default search for what's available
device_type = args.device_type
if device_type is None:
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"

if device_type == "cuda":
    device = torch.device(device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(device_type)

default_dtype = None
dtypes_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
if args.default_dtype is not None:
    default_dtype = dtypes_map[args.default_dtype]

group = None

if args.distributed is not None:
    # gathering optimizer state takes more than 10 minutes, so need a longer timeout.
    dist.init_process_group(backend="nccl")
    group = dist.GroupMember.WORLD
    # Fix until PT 2.3
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)


def get_loss_fn():
    ce_loss = torch.nn.CrossEntropyLoss()

    def loss_fn(output, target):
        return ce_loss(output, target)

    return loss_fn


def training_state(model_path, model, rank):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    is_fsdp = isinstance(model, FSDP)
    dataset_sd = {}
    epoch = 0
    prev_step = -1
    cumulative_tokens = 0

    if model_path is not None:
        path = Path(args.model_path).expanduser()
        if path.exists():
            if path.is_dir():
                files = sorted(path.glob("*.train"))
                if len(files) == 1:
                    training = files[0]
                elif (
                    len(files) > 1
                    and is_fsdp
                    and model.sharding_strategy == ShardingStrategy.FULL_SHARD
                ):
                    training = files[rank]
                elif (
                    len(files) > 1
                    and is_fsdp
                    and model.sharding_strategy == ShardingStrategy.HYBRID_SHARD
                ):
                    training = files[local_rank]
                else:
                    training = None
            else:
                training = path.parent / (path.stem + ".train")
            if training is not None and training.exists():
                sd = torch.load(training)
                optim_sd = sd["optimizer"]
                epoch = sd["epoch"]
                prev_step = sd["step"]
                cumulative_tokens = sd["cumulative_tokens"]
                if "dataset" in sd:
                    dataset_sd = sd["dataset"]
                if isinstance(model, FSDP):
                    optim_sd = model.optim_state_dict_to_load(
                        model, optimizer, optim_sd
                    )
                optimizer.load_state_dict(optim_sd)

                return (optimizer, dataset_sd, epoch, prev_step, cumulative_tokens)
    return (optimizer, dataset_sd, epoch, prev_step, cumulative_tokens)


def main():
    if args.default_dtype:
        torch.set_default_dtype(default_dtype)

    print0("Loading model...")
    model = models.get_model(
        args.architecture,
        args.variant,
        args.model_path,
        source=args.checkpoint_format,
        device_type=device_type,
        data_type=default_dtype,
        distributed_strategy=args.distributed,
        group=group,
        p_dropout=0.0,
    )
    if args.unfuse_weights:
        model = fusion.apply_unfuse_weights(model)
    # model.base_model.rot_emb.compute_freqs_cis(device, 512)
    if args.head_only:
        # Initialize the classification head so we don't get NaNs
        model.classification_head.head.weight.data.normal_(
            0,
            1
            / math.sqrt(math.sqrt(model.config.emb_dim * model.config.src_vocab_size)),
        )
        for param in model.parameters():
            param.requires_grad = False
        model.classification_head.head.weight.requires_grad = True
    model.base_model.embedding.weight.requires_grad = False
    model.base_model.position_embedding.weight.requires_grad = False
    optimizer, dataset_sd, epoch, prev_step, cum_tokens = training_state(
        args.model_path, model, rank
    )
    print("model loaded on worker", rank)
    print0(
        "starting from epoch", epoch, "prior step", prev_step, "cum tokens", cum_tokens
    )
    print0("dataset state", dataset_sd)

    if args.compile:
        # Handled by trainer to include loss
        # model = torch.compile(model, backend=args.compile_backend)
        optimizer.step = torch.compile(optimizer.step, backend=args.compile_backend)

    tokenizer = tokenizers.get_tokenizer(args.tokenizer)

    bos_token_id = 101
    eos_token_id = 102
    bos_token = tokenizer.convert_ids_to_tokens([bos_token_id])[0]
    eos_token = tokenizer.convert_ids_to_tokens([eos_token_id])[0]

    # TODO: split a validation dataset
    dataset = datasets.get_dataset(
        args.dataset_style,
        tokenizer,
        args.dataset_path,
        pad_token=model.config.pad_id,
        max_len=512,
    )
    if len(dataset_sd):
        dataset.load_state_dict(dataset_sd)

    sampler = None
    # if the dataset is iterable, we can't shuffle it, and it should handle
    # sharding internally
    shuffle = not isinstance(dataset, datasets.IterableDataset)

    if args.distributed == "fsdp" and not isinstance(dataset, datasets.IterableDataset):
        sampler = DistributedSampler(
            dataset, rank=rank, num_replicas=world_size, shuffle=True
        )
        # if we shuffle the sampler then we don't shuffle the dataloader
        shuffle = False

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, shuffle=shuffle
    )

    loss_fn = get_loss_fn()

    # TODO, should batch these.
    if args.dataset_style == "instruction":
        sample_prompt = {"instruction": "Explain the meaning of life."}
        sample_prompt = dataset.make_prompt(sample_prompt)
        sample_prompt2 = {"instruction": "Please provide a recipe for chicken soup."}
        sample_prompt2 = dataset.make_prompt(sample_prompt2)
    else:
        sample_prompt = "O God! O God!"
        sample_prompt2 = "Romeo O Romeo,"

    sample_prompt = [bos_token] + tokenizer.tokenize(sample_prompt)
    sample_prompt2 = [bos_token] + tokenizer.tokenize(sample_prompt2)

    validator = trainplugins.InferenceValidator(
        model,
        sample_prompt,
        tokenizer,
        device,
        steps=args.report_steps,
        eos_token=eos_token,
    )
    validator2 = trainplugins.InferenceValidator(
        model,
        sample_prompt2,
        tokenizer,
        device,
        steps=args.report_steps,
        eos_token=eos_token,
    )
    if args.distributed == "hsdp":
        ckp_group = dist.new_group(list(range(torch.cuda.device_count())))
        # if current shard group isn't part of the new group, `new_group` returns an int (-100)
        if not isinstance(ckp_group, dist.ProcessGroup):
            ckp_group = None
    else:
        ckp_group = group
    checkpointing = trainplugins.Checkpointer(
        model,
        optimizer,
        dataset=dataset,
        save_dir=args.output_path,
        steps=args.checkpoint_steps,
        cumulative_tokens=cum_tokens,
        prev_step=prev_step,
        group=ckp_group,
        device=device,
    )
    reporting = trainplugins.MetricReporter(
        seconds=3,
        prev_step=prev_step,
        cumulative_tokens=cum_tokens,
        group=group,
        device=device,
    )

    plugins = [reporting, checkpointing]  #  validator, validator2
    print0("training...")
    with torch.cuda.device(local_rank) if device.type == "cuda" else nullcontext():
        trainer.train(
            model,
            optimizer,
            dataloader,
            device,
            loss_fn,
            start_epoch=epoch,
            epochs=args.epochs,
            prev_step=prev_step,
            trainer_plugins=plugins,
            grad_accum_iters=args.grad_accum_steps,
            compile_loss=args.compile,
            compile_backend=args.compile_backend,
        )


if __name__ == "__main__":
    main()
