from typing import Any, Callable, List, MutableMapping, Union, Optional

import torch
import torch.nn.functional as F

from fms.modules.positions import compute_position_ids
from fms.utils.cache import ExpandableCacheData


def _make_cache_contiguous(past_key_value_states):
    # kv updates are required for torch.compile with
    # mode='reduce-overhead'
    n_kv_s: List[List[torch.Tensor]] = []
    for layer_idx in range(len(past_key_value_states)):
        n_kv_s.append([])
        for tensor_idx in range(len(past_key_value_states[layer_idx])):
            n_kv_s[layer_idx].append(
                past_key_value_states[layer_idx][tensor_idx]
                .clone(memory_format=torch.contiguous_format)
                .detach()
            )
            # torch._dynamo.mark_dynamic(n_kv_s[layer_idx][tensor_idx], 2)
    return n_kv_s


def generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: torch.Tensor,
    max_seq_len: int = 2048,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 10,
    do_sample: bool = True,
    num_beams: int = 1,
    use_cache: bool = False,
    contiguous_cache: bool = False,
    paged_kv_cache: Optional["PagedKVCache"] = None,  # type: ignore
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

    result = input_ids
    next_input = input_ids
    kwargs: MutableMapping[str, Any] = dict()
    kwargs["use_cache"] = use_cache

    if use_cache:
        kwargs["cache_data"] = None
        past_key_value_states = None

    for i in range(max_new_tokens):

        input_ids = next_input[:, -max_seq_len:]

        # compute the mask
        if not use_cache or i == 0:
            is_pad = input_ids == 0
            mask = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)
            kwargs["mask"] = mask.tril(diagonal=0)
        else:
            kwargs["mask"] = None

        # get the cache data and position ids if using cache
        # TODO: The context lengths which can determine the position ids can be handled by the cache manager, but since
        #  there is not yet an implementation for ExpandableKVCacheManager, for now we will do this management here
        if use_cache:
            if i == 0:
                context_lengths = [0 for _ in range(input_ids.size(0))]
                num_tokens_per_sequence = torch.count_nonzero(input_ids.T, dim=0).tolist()
                if paged_kv_cache:
                    cache_data = paged_kv_cache.allocate_prompt_tokens(num_tokens_per_sequence)
                    sequence_ids = cache_data.sequence_ids
                else:
                    cache_data = ExpandableCacheData(data=None)
            else:
                context_lengths = [l + n for l, n in zip(context_lengths, num_tokens_per_sequence)]
                num_tokens_per_sequence = [1 for _ in range(input_ids.size(0))]
                if paged_kv_cache:
                    cache_data = paged_kv_cache.allocate_generated_tokens(
                        sequence_ids, num_tokens_per_sequence
                    )
                else:
                    if contiguous_cache:
                        past_key_value_states = _make_cache_contiguous(past_key_value_states)

                    cache_data = ExpandableCacheData(data=past_key_value_states)

            kwargs["cache_data"] = cache_data
            kwargs["position_ids"] = torch.tensor(compute_position_ids(num_tokens_per_sequence, context_lengths), device=input_ids.device)

        output = model(input_ids, **kwargs)
        if use_cache:
            logits, past_key_value_states = output
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

        if use_cache:
            next_input = next_val
        else:
            next_input = result

    if not batched:
        result = result[0]

    if use_cache and paged_kv_cache:
        paged_kv_cache.free_sequences(sequence_ids)

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
