from typing import Any, Callable, List, MutableMapping, Optional, Union

import torch
import torch.nn.functional as F
from torch import distributed as dist

from fms.modules.positions import compute_position_ids
from fms.utils.cache import CacheDataWithMetadata, KVCacheManager
from fms.utils.cache.expandable import ExpandableKVCacheManager


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
        kv_cache_manager: Optional[KVCacheManager] = None,
        contiguous_cache: bool = False,
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
        sequence_ids: Optional[List[int]] = None
        if kv_cache_manager is None:
            # TODO: standardized way of getting nlayers, nheads, emb_dim
            kv_cache_manager = ExpandableKVCacheManager(
                model.config.nlayers,  # type: ignore
                model.config.nheads,  # type: ignore
                model.config.emb_dim,  # type: ignore
                tensor_parallel_size=dist.get_world_size()
                if dist.is_initialized()
                else 1,
                dtype=torch.get_default_dtype(),
                device=input_ids.device,
            )

    for i in range(max_new_tokens):

        input_ids = next_input[:, -max_seq_len:]

        # compute the mask
        if not use_cache or i == 0:
            is_pad = input_ids == 0
            mask = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)
            kwargs["mask"] = mask.tril(diagonal=0)
        else:
            is_not_pad = result != 0
            mask = is_not_pad.unsqueeze(-2)
            kwargs["mask"] = mask

        # get the cache data and position ids if using cache
        if use_cache and kv_cache_manager:
            if i == 0:
                num_tokens_per_sequence = torch.count_nonzero(
                    input_ids.T, dim=0
                ).tolist()
                cache_data: CacheDataWithMetadata = kv_cache_manager.allocate_tokens(
                    num_tokens_per_sequence
                )
            else:
                num_tokens_per_sequence = [1 for _ in range(input_ids.size(0))]
                cache_data = kv_cache_manager.allocate_tokens(
                    num_tokens_per_sequence, sequence_ids
                )

                # TODO: is this supported? is it necessary?
                # if contiguous_cache:
            sequence_ids = cache_data.sequence_ids

            kwargs["cache_data"] = cache_data
            # TODO: should we just have this as an attribute of CacheDataWithMetadata or provide computation
            kwargs["position_ids"] = cache_data.compute_position_ids(
                num_tokens_per_sequence
            )

        output = model(input_ids, **kwargs)
        if use_cache:
            logits, _ = output
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

    if use_cache:
        kv_cache_manager.free_sequences(sequence_ids)  # type: ignore

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
