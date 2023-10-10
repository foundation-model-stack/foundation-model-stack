import argparse
import os
from pathlib import Path
from typing import List
import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch import nn
from torch.utils.data import DataLoader
from fms import models
from fms.datasets import instructions, text
from fms.distributed.strategy import TensorParallelStrategy
from fms.models import llama
from fms.models.hf.llama.modeling_llama_hf import LLaMAHFForCausalLM
from fms.models.llama import LLaMA, LLaMAConfig
from fms.utils import print0, tokenizers
from fms.training import trainer, plugins as trainplugins

parser = argparse.ArgumentParser(
    description="Script to train or tune a model"
)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--report_steps", type=int, default=500, help="Run the reporting function every report_steps steps")
parser.add_argument("--checkpoint_steps", type=int, default=None, help="If > 0, Checkpoint every checkpoint_steps steps within a single epoch.")
parser.add_argument("--device_type", type=str, default=None, help="Device type. If not specified check for availability of cuda, mps, then cpu")
parser.add_argument(
    "--distributed",
    type=str,
    default=None,
    help="For a distributed job (multiple instances with RANK+WORLD_SIZE) specify fsdp or tp. Default None",
)
parser.add_argument("--output_path", type=str, default='./checkpoints', help="Output directory to save trained model checkpoints")
parser.add_argument("--model_path", type=str, default=None, help="Resume from a checkpoint a the given path")
parser.add_argument(
    "--tokenizer",
    type=str,
    default="char_tokenizer",
    help="Name of or path to the tokenizer (e.g. ~/tokenizer.model)",
)

parser.add_argument("--peft_method", type=str, default=None, help="Peft method (lora, ...). Default None if not using peft")

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))

# default search for what's available
device_type = args.device_type
if device_type is None:
    if torch.cuda.is_available():
        device_type = 'cuda'
    elif torch.backends.mps.is_available():
        device_type = 'mps'
    else:
        device_type = 'cpu'

device = torch.device(device_type, local_rank)

group = None

if args.distributed is not None:
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    group = dist.GroupMember.WORLD


def get_loss_fn():
    ce_loss = torch.nn.CrossEntropyLoss()
    def loss_fn(output, target):
        # model output is batch x seq_len x vocab_size.
        # ce expects batch x vocab x seq_len
        output = output.transpose(-1, -2)
        return ce_loss(output, target)
    return loss_fn

def get_micro_model() -> nn.Module:
    _micro_llama_config = LLaMAConfig(
        emb_dim=192, nheads=4, nlayers=5, max_expected_seq_len=1024, src_vocab_size=256
    )
    model = LLaMA(_micro_llama_config)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    epoch = 0
    if args.model_path is not None:
        # if we already started training, tune from where we left off.
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        # the saved epoch is the last completed one, we'll start at the next epoch
        epoch = state_dict['epoch'] + 1
    return model, optimizer, epoch

def get_llama():
    model = llama.load_fms_llama(args.model_path)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    epoch = 0
    print0("model loaded")
    return model, optimizer, epoch


def hf_to_fms_model(model):
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
    """
    Converts an fms model to an PEFT and HF adapted (wrapped) model, while
    preserving the original `forward` function.

    If we call state_dict() on one of these models we'll get keys prefixed with
    `base_model.model.*`, so correctly saving and re-loading one of these
    requires some care. The state_dict will also contain all paramters, not 
    just the adapter, though we plan to use merged tuned models for now.
    """
    model = LLaMAHFForCausalLM.from_fms_model(model)
    from peft import get_peft_config, get_peft_model, LoraConfig
    from peft.mapping import PeftModelForCausalLM
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "key"],
        bias= "none",
        task_type= "CAUSAL_LM",
        lora_dropout=0.05,
        inference_mode=False
    )

    if args.peft_method == 'lora':
        model = PeftModelForCausalLM(model, lora_config, adapter_name="None")
    else:
        # TODO: add others
        raise ValueError("unsupported peft method", args.peft_method)

    model = hf_to_fms_model(model)
    return model


def main():

    print0("Loading model...")
    torch.set_default_device(device)

    # micro llama model for faster testing
    #model, optimizer, epoch = get_micro_model()
    model, optimizer, epoch = get_llama()
    print("model loaded", local_rank)

    model = model.to(device)
    tokenizer = tokenizers.get_tokenizer(args.tokenizer)

    if args.peft_method is not None:
        model = peft_model(model)

    bos_token = '<s>' # for char tokenizer should be 2 and 3. TODO add to tokenizer?
    eos_token = '</s>'
    bos_token_id = tokenizer.convert_tokens_to_ids([bos_token])[0]
    eos_token_id = tokenizer.convert_tokens_to_ids([eos_token])[0]

    # for use with micro model
    # dataset = text.shakespeare(pad_token=' ').to(device)

    dataset = instructions.JsonInstructions(
        "/home/bvaughan/alpaca_data.json",
        tokenizer=tokenizer, max_len=2048, device=device,
        bos_tok_id=bos_token_id, eos_tok_id=eos_token_id
    )

    dataloader = DataLoader(dataset)

    #model.to(torch.bfloat16)

    loss_fn = get_loss_fn()

    # todo, should batch these.
    sample_prompt = {"instruction" : "Explain the meaning of life."}
    sample_prompt = dataset.make_prompt(sample_prompt)
    sample_prompt2 = {"instruction" : "Please provide a recipe for chicken soup."}
    sample_prompt2 = dataset.make_prompt(sample_prompt2)

    sample_prompt = [bos_token] + tokenizer.tokenize(sample_prompt)
    sample_prompt2 = [bos_token] + tokenizer.tokenize(sample_prompt2)

    # sample_prompt = args.validation_prompt
    validator = trainplugins.InferenceValidator(sample_prompt, tokenizer, device, steps=args.report_steps, eos_token=eos_token)
    validator2 = trainplugins.InferenceValidator(sample_prompt2, tokenizer, device, steps=args.report_steps, eos_token=eos_token)
    checkpointing = trainplugins.Checkpointer(steps=args.checkpoint_steps, group=group, save_dir=args.output_path)
    reporting = trainplugins.MetricReporter()

    plugins = [reporting, validator, validator2, checkpointing]
    print0("training...")
    trainer.train(model, optimizer, dataloader, loss_fn, start_epoch=epoch, epochs=args.epochs, trainer_plugins=plugins, grad_accum_iters=10)

if __name__ == '__main__':
    main()
