import argparse
import cmd
import os
import random
import re

import numpy as np
import torch
import torch._inductor.config
from torch import distributed as dist

from fms.models import get_model
from fms.utils import tokenizers
from fms.utils.generation import generate


# Simple interactive chat with LLaMa model.
#
# Example usage:
# (newfms) [bv-5976k 16th 19:45:01 ~/repos/newfms]$ python scripts/chat.py --device_type=cuda --model_path=/gpfs/models/llama/7B-F/consolidated.00.pth --model_source=meta --tokenizer=/gpfs/models/llama/tokenizer.model --system_message=
# loading model
# loading complete on rank 0
# Enter your prompts/questions or "/help" for usage instructions.

# $ hello!
#  Hello there! It's nice to meet you. Is there something I can help you with or would you like to chat? 😊
# $ just chat
#  Great! I'm happy to chat with you. How has your day been so far? 😊
# $ it's alright, how is yours?
#  I'm just an AI, I don't have a physical body or personal experiences, so my "day" is just a series of interactions with users like you through the internet. However, I'm here to help you with any questions or problems you might have, so feel free to ask me anything! 😊
# $ keyboard interrupt, exiting.

parser = argparse.ArgumentParser(
    description="Script to run inference on a causal model"
)
parser.add_argument("--device_type", type=str, default="cuda")
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
    "--model_path",
    type=str,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
parser.add_argument(
    "--model_source",
    type=str,
    help="Source of the checkpoint. E.g. 'meta', 'hf', None",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    required=True,
    help="Path to the tokenizer (e.g. ~/tokenizer.model)",
)
parser.add_argument(
    "--no_use_cache",
    action="store_false",
    help="Disable the kv-cache (on by default)",
)
parser.add_argument(
    "--compile",
    action="store_true",
    help="Use torch.compile (slow for first inference pass)",
)
parser.add_argument(
    "--compile_mode",
    type=str,
    help="Mode for compilation",
    default="default",
    choices=["default", "reduce-overhead"],
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Set torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=500,
    help="Maximum number of new tokens to generate for each prompt",
)

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""


parser.add_argument(
    "--system_message",
    type=str,
    default=DEFAULT_SYSTEM_PROMPT,
    help="Optional system message telling the chat how to behave. Default llama system message requests that the model to be safe and helpful. Empty string for no system message.",
)

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)

torch.set_default_dtype(torch.half)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)  # pytorch random seed
    np.random.seed(SEED)  # numpy random seed
    torch.use_deterministic_algorithms(True)

if args.distributed:
    dist.init_process_group()
    # Fix until PT 2.3
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

print("loading model")
if args.distributed:
    distr_param = "tp"
else:
    if torch.cuda.device_count() > 1 and world_size == 1:
        distr_param = "mp"
    else:
        distr_param = None

model = get_model(
    args.architecture,
    args.variant,
    model_path=args.model_path,
    device_type=args.device_type,
    source=args.model_source,
    distributed_strategy=distr_param,
    group=dist.group.WORLD,
)
tokenizer = tokenizers.get_tokenizer(args.tokenizer)
model.eval()
torch.set_grad_enabled(False)
print("loading complete on rank", local_rank)

if args.compile:
    print("compiling model")
    # compiling can make first inference pass slow
    model = torch.compile(model, mode=args.compile_mode)


"""
llama template for chat looks like:

<bos>[INST]<<SYS>>system message<</SYS>>Instruction prompt[/INST]Answer<eos><bos>[INST] Prompt [/INST] Answer <eos><bos>[INST] Prompt [/INST] Answer <eos>

specified here:
https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/llama/tokenization_llama.py#L438

and here:
https://github.com/meta-llama/llama/blob/556949fdfb72da27c2f4a40b7f0e4cf0b8153a28/llama/generation.py#L320-L362

TODO: add some way to specify alternate templates
"""


class ChatREPL(cmd.Cmd):
    def __init__(self, max_new_tokens, system_message=None):
        super().__init__()
        self.intro = 'Enter your prompts/questions or "/help" for usage instructions.\n'
        self.prompt = "$ "
        self.B_SYS = "<<SYS>>\n"
        self.E_SYS = "\n<<SYS>>\n\n"
        self.B_INST = "[INST]"
        self.E_INST = "[/INST]"
        self.max_new_tokens = max_new_tokens

        self.buffer = []
        self.buffer.append(tokenizer.bos_token_id)
        if system_message is not None and len(system_message):
            msg = self.B_SYS + system_message + self.E_SYS
            self.buffer.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(msg)))

    def onecmd(self, line):
        # Check if the line starts with a special command prefix
        if line.startswith("/"):
            command = line[1:].strip().split(" ")[0]
            args = line[len(command) + 2 :].strip()
            try:
                func = getattr(self, "do_" + command)
                return func(args)
            except AttributeError:
                print(f"Unknown command: {line}")
                return
        # handles ctrl-D
        elif line == "EOF":
            self.do_EOF()
        else:
            return self.default(line)

    def default(self, line):
        if not len(line.strip()):
            return

        self.buffer.extend(
            tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(self.B_INST + line + self.E_INST)
            )
        )
        ids = torch.tensor(self.buffer, dtype=torch.long, device=device)

        result = generate(
            model,
            ids,
            max_new_tokens=self.max_new_tokens,
            use_cache=args.no_use_cache,
            do_sample=not args.deterministic,
            eos_token_id=tokenizer.eos_token_id,
        )
        if local_rank != 0:
            return

        result = result[len(self.buffer) :].tolist()
        self.buffer = self.buffer + result + [tokenizer.eos_token_id]
        print(
            tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result))
        )

    def do_set(self, arg):
        args = re.split("=|\\s", arg)
        if len(args) != 2:
            print("usage: /set max_new_tokens=100")
        if args[0] == "max_new_tokens":
            self.max_new_tokens = int(args[1])
        else:
            print("Invalid attribute: " + args[0])

    def do_buffer(self, arg):
        print(f"The look-back buffer has {len(self.buffer)} tokens:")
        print(
            tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(self.buffer)
            )
        )

    def do_help(self, arg):
        print("Help on using the REPL:")
        print("/help - Show this help message.")
        print(
            "/set max_new_tokens <value> - set the maximum length of response. Responses may be truncated."
        )
        print("/buffer - show the full buffer of chat history being fed to the model.")
        print("Any other input is treated as text to be handled by the LLM.")
        print("ctrl-c or ctrl-d to exit.")

    def do_quit(self, line=""):
        self.do_EOF(line)

    def do_exit(self, line=""):
        self.do_EOF(line)

    def do_EOF(self, line=""):
        print("Exiting...")
        exit()


if __name__ == "__main__":
    try:
        ChatREPL(args.max_new_tokens, args.system_message).cmdloop()
    except KeyboardInterrupt:
        # ctrl-C isn't handled by cmd.Cmd
        print("keyboard interrupt, exiting.")
