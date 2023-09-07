import json
import os
from pathlib import Path
import re

import torch
from fms.models.llama import LLaMA
from fms.utils.tokenizers import get_tokenizer

def _rename_weights_to_fms(orig_sd):
    replacements = [
        (r"^tok_embeddings", "shared.emb"),
        (r"^norm", "dec_norm"),
        (r"^output", "shared.head"),
        (r"^layers", "layers"),
        (r"\.attention\.", ".attn."),
        (r"attn\.wq", "attn.query"),
        (r"attn\.wk", "attn.key"),
        (r"attn\.wv", "attn.value"),
        (r"attn\.wo", "attn.dense"),
        (r"attention_norm", "ln"),
        (r"feed_forward\.w1", "ff_sub_layer.wg"),
        (r"feed_forward\.w2", "ff_sub_layer.w2"),
        (r"feed_forward\.w3", "ff_sub_layer.w1"),
        (r"ffn_norm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in orig_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

    return new_sd


def load_weights(model_path: str, tokenizer_path: str):
    # from llama.tokenizer import Tokenizer
    model_path = os.path.expanduser(model_path)
    tokenizer_path = os.path.expanduser(tokenizer_path)

    # Load tokenizer
    tokenizer = get_tokenizer(tokenizer_path)

    # Load Llama model from Meta's weights
    checkpoints = sorted(Path(model_path).glob("*.pth"))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"

    ckpt_path = checkpoints[local_rank]
    checkpoint_sd = torch.load(ckpt_path, map_location="cpu")
    with open(Path(model_path) / "params.json", "r") as f:
        params = json.loads(f.read())
    hidden_grow_factor = 8 / 3
    if "ffn_dim_multiplier" in params:
        hidden_grow_factor = hidden_grow_factor * params["ffn_dim_multiplier"]

    # IBM LLaMa
    fms_sd = _rename_weights_to_fms(checkpoint_sd)
    torch.set_default_dtype(torch.float16)
    model = LLaMA(
        src_vocab_size=tokenizer.vocab_size(),
        emb_dim=params["dim"],
        nheads=params["n_heads"],
        kvheads=params["n_kv_heads"] if "n_kv_heads" in params else 0,
        nlayers=params["n_layers"],
        pad_id=tokenizer.pad_id,
        hidden_grow_factor=hidden_grow_factor,
        multiple_of=params["multiple_of"],
        norm_eps=params["norm_eps"],
    )
    torch.set_default_dtype(torch.float32)
    model.load_state_dict(fms_sd, strict=False)  # the meta weights have some extra stuff

    return model, tokenizer
