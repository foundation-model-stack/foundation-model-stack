import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import torch
from torch import distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from fms import models
from fms.datasets import instructions, text
from fms.models.llama import LLaMA, LLaMAConfig
from fms.utils import print0, tokenizers
from fms.training import trainer, plugins as trainplugins

# e.g.:
# srun --gres=gpu:2 --cpus-per-task=24 --mem=512G --unbuffered --gres-flags=enforce-binding \
#       torchrun --nproc_per_node=2 scripts/train_causal.py --architecture=llama --variant=7b \
#       --tokenizer=~/models/tokenizer.model --model_path=~/models/7B/ --output_path=./tuned/ \
#       --report_steps=10 --checkpoint_format=meta --distributed=fsdp
#


parser = argparse.ArgumentParser(description="Script to train or tune a model")
parser.add_argument("--epochs", type=int, default=1)
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
    "--device_type",
    type=str,
    default=None,
    help="Device type. If not specified check for availability of cuda, mps, then cpu",
)
parser.add_argument(
    "--distributed",
    type=str,
    default=None,
    help="The strategy used for distributed the model. E.g. fsdp, ddp, tp, mp. Default None",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="./checkpoints",
    help="Output directory to save trained model checkpoints",
)
parser.add_argument(
    "--architecture",
    type=str,
    default="llama",
    help="The model architecture to benchmark",
)
parser.add_argument(
    "--variant",
    type=str,
    default="7b",
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
    "--peft_method",
    type=str,
    default=None,
    help="Peft method (lora, ...). Default None if not using peft",
)

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
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

device = torch.device(device_type, local_rank)

group = None

if args.distributed is not None:
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    group = dist.GroupMember.WORLD


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

    model = HFAdaptedLLaMAForCausalLM.from_fms_model(model)
    from peft import get_peft_config, get_peft_model, LoraConfig
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
    optimizer, epoch = training_state(args.model_path, model, local_rank)

    print("model loaded", local_rank)

    tokenizer = tokenizers.get_tokenizer(args.tokenizer)

    if args.peft_method is not None:
        model = peft_model(model)

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    bos_token = tokenizer.convert_ids_to_tokens(bos_token_id)
    eos_token = tokenizer.convert_ids_to_tokens(eos_token_id)

    # TODO: provide a way to fetch other datasets by type and path (or url).
    # e.g. `text.shakespeare(pad_token=' ')` or the below with an alternate
    # json file.
    dataset = instructions.JsonInstructions(
        "/home/bvaughan/alpaca_data.json",
        tokenizer=tokenizer,
        max_len=2048,
        device=device,
    )

    sampler = None
    shuffle = True
    if args.distributed == "fsdp":
        sampler = DistributedSampler(
            dataset, rank=local_rank, num_replicas=world_size, shuffle=True
        )
        # if we shuffle the sampler then we don't shuffle the dataloader
        shuffle = False

    dataloader = DataLoader(dataset, sampler=sampler, shuffle=shuffle)

    loss_fn = get_loss_fn()

    # todo, should batch these.
    sample_prompt = {"instruction": "Explain the meaning of life."}
    sample_prompt = dataset.make_prompt(sample_prompt)
    sample_prompt2 = {"instruction": "Please provide a recipe for chicken soup."}
    sample_prompt2 = dataset.make_prompt(sample_prompt2)

    sample_prompt = [bos_token] + tokenizer.tokenize(sample_prompt)
    sample_prompt2 = [bos_token] + tokenizer.tokenize(sample_prompt2)

    # sample_prompt = args.validation_prompt
    validator = trainplugins.InferenceValidator(
        sample_prompt, tokenizer, device, steps=args.report_steps, eos_token=eos_token
    )
    validator2 = trainplugins.InferenceValidator(
        sample_prompt2, tokenizer, device, steps=args.report_steps, eos_token=eos_token
    )
    checkpointing = trainplugins.Checkpointer(
        steps=args.checkpoint_steps, group=group, save_dir=args.output_path
    )
    reporting = trainplugins.MetricReporter()

    plugins = [reporting, validator, validator2, checkpointing]
    print0("training...")
    with torch.cuda.device(local_rank) if device.type == "cuda" else nullcontext():
        trainer.train(
            model,
            optimizer,
            dataloader,
            loss_fn,
            start_epoch=epoch,
            epochs=args.epochs,
            trainer_plugins=plugins,
            grad_accum_iters=10,
        )


if __name__ == "__main__":
    main()
