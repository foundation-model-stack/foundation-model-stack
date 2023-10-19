import logging
import os
from argparse import _ArgumentGroup

from fms.utils.from_closed import str_to_activation


def add_config_args(arg_group: _ArgumentGroup):
    arg_group.add_argument("--simulated_gpus", type=int, default=0, help="Number of GPUs to simulate")
    arg_group.add_argument("--num_steps", type=int, default=110000, help="Number of SGD steps.")
    arg_group.add_argument("--b_size", type=int, default=64, help="Batch size per GPU")
    arg_group.add_argument(
        "--data_path",
        type=str,
        default="/data/input/t5_watson_1138/",
        help="Dataset directory (absolute path)",
    )
    arg_group.add_argument(
        "--log_path",
        type=str,
        default="/data/output/lego/TEMP/",
        help="Log/checkpoint directory (absolute path)",
    )
    arg_group.add_argument("--seed", type=int, default=42, help="Random seed, requires int (default 42)")
    arg_group.add_argument("--bf16", default=False, action="store_true", help="Use BF16 acceleration?")
    arg_group.add_argument(
        "--fast_ln", default=False, action="store_true", help="Use BF16 LayerNorm (3% speedup, but unstable)?"
    )
    arg_group.add_argument(
        "--parallel_mode",
        choices=("ddp", "fsdp", "hsdp"),
        help="Method for parallelizing (supports [ddp, fsdp, hsdp])",
    )
    arg_group.add_argument(
        "--recompute_policy",
        choices=("all", "none", "ffn"),
        help="Save memory be recomputing activations in modules: [all, none, ffn]",
    )
    arg_group.add_argument(
        "--testrun_data_index",
        type=int,
        default=-100,
        help="Dataset index (starting at 0) for test runs. Default -100 means not a testrun",
    )
    arg_group.add_argument(
        "--logical_shards", type=int, default=0, help="Number of logical data shards (upper bound on job scaling)"
    )


def add_profiler_args(arg_group: _ArgumentGroup):
    arg_group.add_argument(
        "--profile",
        default=False,
        action="store_true",
        help="Run profiler during training",
    )
    arg_group.add_argument(
        "--profile_skip",
        type=int,
        default=10,
        help="Skipped steps for profiler",
    )
    arg_group.add_argument(
        "--profile_wait",
        type=int,
        default=5,
        help="Waiting steps for profiler",
    )
    arg_group.add_argument(
        "--profile_warmup",
        type=int,
        default=1,
        help="Warm-up steps for profiler",
    )
    arg_group.add_argument(
        "--profile_active",
        type=int,
        default=4,
        help="Running steps for profiler",
    )
    arg_group.add_argument(
        "--profile_remove_stack",
        action="store_false",
        help="Not store stack traces in profiler",
    )
    arg_group.add_argument(
        "--profile_remove_memory",
        action="store_false",
        help="Not store memory usage in profiler",
    )
    arg_group.add_argument(
        "--profile_remove_shapes",
        action="store_false",
        help="Not store shapes in profiler",
    )


def add_ckp_args(arg_group: _ArgumentGroup):
    arg_group.add_argument(
        "--ckp_path",
        type=str,
        default="",
        help="Checkpoint file or directory for initial load. If directory, loads latest.",
    )
    arg_group.add_argument("--save_interval", type=int, default=1000, help="Permanent checkpoint interval")
    arg_group.add_argument("--report_interval", type=int, default=100, help="Reporting interval")
    arg_group.add_argument(
        "--reset_stepcount",
        action="store_true",
        help="If loading from checkpoint, force restart to step 0?",
    )
    arg_group.add_argument(
        "--drop_optimizer",
        action="store_true",
        help="If loading from checkpoint, skip loading the optimizer state?",
    )
    arg_group.add_argument(
        "--drop_dataset",
        action="store_true",
        help="If loading from checkpoint, skip loading the dataset state?",
    )


def add_vocab_args(arg_group: _ArgumentGroup):
    arg_group.add_argument("--vocab", type=int, default=50048, help="Size of tokenizer")
    arg_group.add_argument("--pad_token", type=int, default=-1, help="Pad token (must be nonnegative)")
    arg_group.add_argument("--unk_token", type=int, default=-1, help="Unk token (must be nonnegative)")
    arg_group.add_argument("--bos_token", type=int, default=-1, help="BOS token (must be nonnegative)")
    arg_group.add_argument("--eos_token", type=int, default=-1, help="EOS token (must be nonnegative)")
    arg_group.add_argument("--sep_token", type=int, default=-1, help="Separator token (must be nonnegative)")
    arg_group.add_argument(
        "--tie_head",
        action="store_true",
        default=False,
        help="Use embedding weights for prediction?",
    )


def add_training_args(arg_group: _ArgumentGroup):
    arg_group.add_argument("--lr", type=float, default=0, help="Max LR (default 0 autoscales)")
    arg_group.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    arg_group.add_argument("--dropout", type=float, default=0, help="Dropout value (default 0)")
    arg_group.add_argument(
        "--warmup_interval",
        type=int,
        default=2000,
        help="Number of steps for parabolic warmup",
    )
    arg_group.add_argument(
        "--clip_th",
        type=float,
        default=1.0,
        help="Clipping threshold for parameter gradients",
    )
    arg_group.add_argument("--label_smoothing", type=float, default=0, help="Amount of label smoothing")
    arg_group.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="First moment gradient tracker decay rate for AdamW",
    )
    arg_group.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Second moment gradient tracker decay rate for AdamW",
    )
    arg_group.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Absolute path to file (of comma separated dataset names) or comma separated str of dataset names.",
    )
    arg_group.add_argument(
        "--dataset_weights",
        type=str,
        default="",
        help="Absolute path to file (of comma separated weights) or comma separated str of weights.",
    )
    arg_group.add_argument(
        "--oversample_weights",
        default=False,
        action="store_true",
        help="Do weights indicate integer oversamping rates, as opposed to desired percentages of token makeup?",
    )
    arg_group.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Initialization gain",
    )


def add_llama_args(arg_group: _ArgumentGroup):
    # (default 7B architecture)
    arg_group.add_argument("--n_layers", type=int, default=32, help="Number of layers")
    arg_group.add_argument("--emb_dim", type=int, default=4096, help="emb dimension")
    arg_group.add_argument("--n_heads", type=int, default=32, help="Number of heads")
    arg_group.add_argument(
        "--kv_heads",
        type=int,
        default=0,
        help="Number of kv heads to use for GQA (default 0 produces MHA)",
    )
    arg_group.add_argument("--emb_kq", type=int, default=128, help="kq dim")
    arg_group.add_argument("--emb_v", type=int, default=128, help="v dim")
    arg_group.add_argument(
        "--hidden_grow_factor",
        type=float,
        default=2.6875,
        help="Growth factor for mlp sublayers",
    )
    arg_group.add_argument(
        "--act_fn",
        type=str,
        default="swish",
        help="Activation function (gelu,gelu-tanh,mish,relu,sigmoid,silu,[swish],tanh)",
    )
    arg_group.add_argument(
        "--multiple_of",
        type=int,
        default=256,
        help="Round grow_factor output to nearest multiple of this",
    )
    arg_group.add_argument(
        "--norm_eps",
        type=float,
        default=1e-6,
        help="Epsilon safety term for LayerNorm",
    )


def validate_args(args, world_size):

    # Verify that act_fn is valid
    str_to_activation(args.act_fn)

    # Handle any necessary default matching
    if args.simulated_gpus == 0:
        args.simulated_gpus = world_size
    if args.logical_shards == 0:
        args.logical_shards = world_size * 12

    # Run various asserts
    assert (
        args.simulated_gpus % world_size == 0
    ), f"Simulated gpus {args.simulated_gpus} must be an exact multiple of actual gpus {world_size}"
    assert (
        args.simulated_gpus >= world_size
    ), "Cannot simulate fewer gpus than are available. Please lower gpu count to match the desired target value."
    assert (
        args.save_interval % args.report_interval == 0
    ), f"For accurate timing metrics, reporting interval {args.report_interval} must divide checkpoint interval {args.save_interval} evenly"
    if args.fast_ln and not args.bf16:
        logging.warning("You have specified LN acceleration to bf16 in a non-bf16 model. This option will be ignored.")

    return args


def validate_arg_tokens(args, tokens, allow_no_pad=True):
    # tokens is a comma-separated string of three-character qualifiers (i.e. "sep,bos, unk")
    token_dict = {
        "pad": args.pad_token,
        "unk": args.unk_token,
        "bos": args.bos_token,
        "eos": args.eos_token,
        "sep": args.sep_token,
    }
    for t in [x.strip().lower() for x in tokens.split(",")]:
        assert t in token_dict.keys(), f"Dummy token {t} is not a recognized type (pad,unk,bos,eos,sep)"
        assert (token_dict[t] >= 0 and token_dict[t] < args.vocab) or (
            t == "pad" and token_dict[t] == -1 and allow_no_pad
        ), f"{t} token index {token_dict[t]} must be nonnegative and less than vocab size {args.vocab}, or -1 if pad token and allow_no_pad ({allow_no_pad}) is True"
