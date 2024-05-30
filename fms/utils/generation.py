import functools
import statistics
import time
from typing import Any, Callable, List, MutableMapping, Optional, Union

import torch
import torch.distributed
import torch.nn.functional as F
import torch.profiler

from fms import distributed


def generate(
    prefill_model: Union[Callable, torch.nn.Module],
    decode_model: Union[Callable, torch.nn.Module],
    input_ids: torch.Tensor,
    max_seq_len: int = 4096,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 10,
    do_sample: bool = True,
    num_beams: int = 1,
    use_cache: bool = False,
    contiguous_cache: bool = False,
    eos_token_id: Optional[int] = None,
):
    """
    A trivial generate function that can be used for validation/testing in
    cases where HF is not available.
    We could add implementations for other types of generation, but this is
    enough for making sure a model is working.
    Does not implement batching nor beam search, but those could be added.

    Args:
        model: A function or nn.Module that takes a batch of input_ids and
            returns logits
        prefix: A tensor of token IDs.
        max_seq_len: the sequence length of the model
        max_new_tokens: max tokens to generate
        temperature: temperature of softmax when sampling
        top_k: only search among top k tokens
        do_sample: multinomial sampling. False for greedy.
        num_beams: TODO: support beam search
        use_cache: requires that the model accept use_cache and
            past_key_value_states args in forward method.
    """
    batched = False
    if num_beams != 1:
        raise NotImplementedError("generate() does yet not support beam search")
    if type(input_ids) == torch.Tensor:
        if input_ids.dim() != 1:
            batched = True
    else:
        raise RuntimeError("generate() requires a tensor of token ids as the prefix")

    if not batched:
        input_ids = input_ids.unsqueeze(0)

    eos_found = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
    )

    result = input_ids
    next_input = input_ids
    kwargs: MutableMapping[str, Any] = dict()
    kwargs["past_key_value_states"] = None
    kwargs["use_cache"] = use_cache
    kwargs["position_ids"] = torch.arange(
        0, input_ids.shape[1], device=input_ids.device, dtype=torch.int64
    ).repeat(input_ids.shape[0], 1)

    total_start = time.time()
    for i in range(max_new_tokens):
        # itl_start = time.time()
        input_ids = next_input[:, -max_seq_len:]
        if i == 0:
            output = prefill_model(input_ids, **kwargs)
        else:
            output = decode_model(input_ids, **kwargs)
        if use_cache:
            logits, past_key_value_states = output
            kwargs["past_key_value_states"] = past_key_value_states
        else:
            logits = output
        logits = logits[:, -1, :]

        if do_sample:
            # get logits from last value in sequence nad scale
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_val = torch.multinomial(probs, num_samples=1)
        else:
            next_val = torch.argmax(logits, dim=-1).unsqueeze(0).t()

        result = torch.cat((result, next_val), dim=-1)

        # avoid continuing to generate if all have reached EOS
        if eos_token_id is not None:
            eos_found = torch.logical_or(eos_found, next_val == eos_token_id)
            if torch.sum(eos_found) == input_ids.shape[0]:
                break

        if use_cache:
            next_input = next_val
        else:
            next_input = result
        kwargs["position_ids"] = kwargs["position_ids"][:, -1:] + 1

    torch.cuda.synchronize()

    # torch.distributed.barrier()
    total_time = time.time() - total_start

    print(f"Total time: {total_time}")
    if not batched:
        result = result[0]
    return result


def truncate_after_eos(result, eos_token_id):
    """
    Helper function to return a truncated sequence of token IDs stopping at
    (and including) the 'end of sentence' token.
    Currently only handles unbatched sequences.
    """
    if eos_token_id is None:
        return result

    eos_idx = torch.where(result == eos_token_id)
    eos_idx = eos_idx[0]
    if eos_idx.shape[0] >= 1:
        eos_idx = eos_idx[0].item()
        result = result[: eos_idx + 1]
    return result
