import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fms import datasets, models
from fms.models.hf.utils import to_hf_api
from fms.training import plugins as trainplugins
from fms.training import trainer
from fms.utils import print0, tokenizers


#
# This is a fairly minimal training/tuning script for causal language models.
#
# Example usage for fine tuning llama 7b on the alpaca dataset on slurm:
# srun --gres=gpu:2 --cpus-per-task=24 --mem=512G --unbuffered --gres-flags=enforce-binding \
#       torchrun --nproc_per_node=2 scripts/train_causal.py --architecture=llama --variant=7b \
#       --tokenizer=~/models/tokenizer.model --model_path=~/models/7B/ --output_path=./tuned/ \
#       --report_steps=10 --checkpoint_format=meta --distributed=fsdp
#


parser = argparse.ArgumentParser(description="Script to train or tune a model")

# parameters for model and tokenizer initialization and loading
parser.add_argument(
    "--architecture",
    type=str,
    default="llama",
    help="The model architecture to benchmark",
)
parser.add_argument(
    "--variant",
    type=str,
    default="micro",
    help="The model variant (configuration) to benchmark. E.g. 7b, 13b, 70b.",
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
    "--distributed",
    type=str,
    default=None,
    help="The strategy used for distributing the model. E.g. fsdp, ddp, tp, mp. Default None",
)
parser.add_argument(
    "--peft_method",
    type=str,
    default=None,
    help="Peft method (lora, ...). Default None if not using peft",
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

group = None

if args.distributed is not None:
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    group = dist.GroupMember.WORLD
    # Fix until PT 2.3
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)


def get_loss_fn():
    ce_loss = torch.nn.CrossEntropyLoss()

    def loss_fn(output, target):
        # model output is batch x seq_len x vocab_size.
        # ce expects batch x vocab x seq_len
        output = output.transpose(-1, -2)
        return ce_loss(output, target)

    return loss_fn


def _hf_to_fms_model(model):
    model.orig_forward_f = model.forward

    def new_forward(self, *args, **kwargs):
        hf_out = model.orig_forward_f(self, *args, **kwargs)
        logits = hf_out.logits
        kv_cache = hf_out.past_key_values
        if "use_cache" in kwargs and kwargs["use_cache"]:
            return logits, kv_cache
        else:
            return logits

    model.forward = new_forward
    return model


def peft_model(model):
    # TODO: consider using loralib directly instead:
    # https://github.com/microsoft/LoRA/tree/main
    # This would simplify checkpoint handling.
    """
    Converts an fms model to an PEFT and HF adapted (wrapped) model, while
    preserving the original `forward` function.

    If we call state_dict() on one of these models we'll get keys prefixed with
    `base_model.model.*`, so correctly saving and re-loading one of these
    requires some care. The state_dict will also contain all paramters, not
    just the adapter, though we plan to use merged tuned models for now.
    """
    from fms.models.hf.llama.modeling_llama_hf import HFAdaptedLLaMAForCausalLM

    model = to_hf_api(model)
    from peft import LoraConfig, get_peft_config, get_peft_model
    from peft.mapping import PeftModelForCausalLM

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "key"],
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        inference_mode=False,
    )

    if args.peft_method == "lora":
        model = PeftModelForCausalLM(model, lora_config, adapter_name="None")
    else:
        # TODO: add others
        raise ValueError("unsupported peft method", args.peft_method)

    model = _hf_to_fms_model(model)
    return model


def training_state(model_path, model, rank):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    epoch = 0

    if model_path is not None:
        path = Path(args.model_path).expanduser()
        if path.exists():
            if path.is_dir():
                files = sorted(path.glob("*.train"))
                if len(files):
                    training = files[rank]
                else:
                    training = None
            else:
                training = path.parent / (path.stem + ".train")
            if training is not None and training.exists():
                sd = torch.load(training)
                optim = sd["optimizer"]
                epoch = sd["epoch"]
                optimizer.load_state_dict(optim)
                epoch = epoch + 1

                return (optimizer, epoch)
    return (optimizer, epoch)


def main():
    print0("Loading model...")
    model = models.get_model(
        args.architecture,
        args.variant,
        args.model_path,
        source=args.checkpoint_format,
        device_type=device_type,
        distributed_strategy=args.distributed,
        group=group,
    )
    # model.to(torch.half)
    optimizer, epoch = training_state(args.model_path, model, rank)

    print("model loaded", rank)

    tokenizer = tokenizers.get_tokenizer(args.tokenizer)

    if args.peft_method is not None:
        model = peft_model(model)

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    bos_token = tokenizer.convert_ids_to_tokens([bos_token_id])[0]
    eos_token = tokenizer.convert_ids_to_tokens([eos_token_id])[0]

    # TODO: split a validation dataset
    dataset = datasets.get_dataset(args.dataset_style, tokenizer, args.dataset_path)

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
        sample_prompt, tokenizer, device, steps=args.report_steps, eos_token=eos_token
    )
    validator2 = trainplugins.InferenceValidator(
        sample_prompt2, tokenizer, device, steps=args.report_steps, eos_token=eos_token
    )
    checkpointing = trainplugins.Checkpointer(
        steps=args.checkpoint_steps, group=group, save_dir=args.output_path
    )
    reporting = trainplugins.MetricReporter(seconds=20, group=group, device=device)

    plugins = [reporting, validator, validator2, checkpointing]
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
            trainer_plugins=plugins,
            grad_accum_iters=args.grad_accum_steps,
        )


if __name__ == "__main__":
    main()
