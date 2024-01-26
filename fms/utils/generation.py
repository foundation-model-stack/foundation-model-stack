import time
from typing import Any, Callable, List, MutableMapping, Optional, Union

import torch
import torch.nn.functional as F
from torch import distributed as dist

from fms.modules.speculator import Speculator
from fms.utils.cache import CacheDataWithMetadata, KVCacheManager, flatten_batch, select_inflate_dim
from fms.utils.cache.expandable import ExpandableKVCacheManager
from fms.utils.cache.paged import PagedKVCacheManager


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
            if sequence_ids is None:
                num_tokens_per_sequence = torch.count_nonzero(
                    input_ids.T, dim=0
                ).tolist()
            else:
                num_tokens_per_sequence = [1 for _ in range(input_ids.size(0))]

            cache_data = kv_cache_manager.allocate_tokens(
                num_tokens_per_sequence, sequence_ids
            )

            # TODO: contiguous_cache -- is this supported? is it necessary?

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


def speculative_generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: Union[torch.Tensor, List[torch.Tensor]],
    speculator: Speculator,
    max_seq_len: int = 2048,
    new_tokens: int = 256,
    top_k: int = 5,
    threshes=[5, 3, 2],
    verbose_dict=None,
    kv_cache_manager: PagedKVCacheManager = None,
    flatting = True,
):
    """
    A reference implementation of speculative decoding generation.
    Returns at least the specified number of tokens - the speculator may return a
    few extra in the final step.
    If input is batched, continues generating until EVERY sequence has produced AT LEAST the required number of tokens.
    Input (and output) tokens beyond max_seq_len are simply dropped for a sliding-window approach.
    Currently reproduces behavior of greedy decoding only.
    Args:
        model: A function or nn.Module that takes a batch of input_ids and
            returns logits
        input_ids: A length n tensor of token IDs, or list of such tensors
        speculator: A function or nn.Module that takes a state vector and sampled token
            and returns a set of candidate suffixes
        max_seq_len: the sequence length of the base model
        new_tokens: number of tokens to generate
        top_k: only score the top k candidates from the speculator
        threshes: use top k predictions from each head to generate speculator candidate pool
        verbose_dict: Optional HF tokenizer vocab dict. If provided, runs verbosely and prints
            speculator behavior and scoring for each step
    Returns:
        result: List of id tensors, possibly different lengths if batching.
        n_steps: Number of foward passes used to generate provided tokens.
    """

    verbose = False
    if verbose_dict is not None:
        verbose = True
        vinv = {v: k for k, v in verbose_dict.items()}

    def decode_obo(x, vinv):
        return [vinv[z] for z in x.squeeze().tolist()]

    # Construct batch(es) and initial inputs
    bsize = len(input_ids)
    result = input_ids  # [b] n
    # Build padded batched input tensor
    max_len = max([seq.size(0) for seq in input_ids])
    n_pads_init = [max_len - seq.size(0) for seq in input_ids]
    n_pads = torch.Tensor(n_pads_init).to(device=input_ids[0].device, dtype=torch.int)
    inputs = torch.stack(
        [F.pad(input_ids[i], (n_pads_init[i], 0)) for i in range(bsize)]
    )
    num_tokens_per_sequence = torch.count_nonzero(
        inputs[:, :-1].T, dim=0
    ).tolist()
    cache_data = kv_cache_manager.allocate_tokens(num_tokens_per_sequence)
    parent_sequence_ids = cache_data.sequence_ids
    # Build padded causal mask
    mask = torch.ones(
        bsize,
        1,
        inputs.size(1) - 1,
        inputs.size(1) - 1,
        device=inputs.device,
    )
    mask = mask.tril()  # b 1 n-1 n-1
    # Mask off any left-pads
    pad_mask = torch.arange(mask.size(3), device=mask.device).view(
        1, 1, 1, -1
    )  # 1 1 1 n-1
    pad_mask = pad_mask.expand(bsize, 1, 1, -1)  # b 1 1 n-1
    pad_mask = pad_mask.sub(n_pads.sub(1).view(-1, 1, 1, 1)).clamp(0, 1)
    eye = torch.eye(mask.size(3), device=mask.device)[None, None, :, :]  # 1 1 n-1 n-1
    mask = mask.mul(pad_mask).logical_or(eye).log()  # b 1 n-1 n-1
    # Handle position_ids
    pos_ids = torch.arange(mask.size(3), device=inputs.device).repeat(bsize, 1)  # b n-1
    pos_ids -= n_pads[:, None]

    kwargs: MutableMapping[str, Any] = dict()
    kwargs["use_cache"] = True

    # Build kv cache and get initial state vector
    n_adds = speculator.n_predict + 1
    inputs = inputs[:, -max_seq_len + n_adds :]
    position_ids = cache_data.compute_position_ids(num_tokens_per_sequence)
    output = model(
        inputs[:, :-1],
        include_embeds=True,
        position_ids=position_ids,
        mask=mask,
        cache_data=cache_data,
        **kwargs
    )
    _, past_key_value_states, embeds = output
    embeds = embeds[:, -1:]

    n_gen = torch.zeros(bsize, device=inputs.device, dtype=torch.int)
    n_steps = 0
    inputs = inputs[:, -1:]
    start_time = time.time()
    while min(n_gen) < new_tokens:
        n_steps += 1

        # create candidate sequences
        child_sequence_ids_list = []
        child_sequence_ids_flattened = []
        num_tokens_per_sequence = [n_adds for _ in range(inputs.size(0) * top_k)]
        # each parent will have top_k child sequences
        for parent_sequence_id in parent_sequence_ids:
            child_sequence_ids = kv_cache_manager.add_child_sequences(parent_sequence_id, top_k)
            child_sequence_ids_list.append(child_sequence_ids)
            child_sequence_ids_flattened.extend(child_sequence_ids)

        # add n_adds tokens to each candidate
        cache_data = kv_cache_manager.allocate_tokens(num_tokens_per_sequence, child_sequence_ids_flattened)
        position_ids = cache_data.compute_position_ids(num_tokens_per_sequence)

        # Get candidate set of speculations
        adds = speculator.generate_suffixes(embeds, inputs, threshes, top_k)  # b k h
        inputs = torch.cat(
            [inputs.unsqueeze(1).expand(bsize, top_k, 1), adds], dim=-1
        ).int()  # b k 1+h

        this_flatting = False
        if flatting:
            flat_inputs, unflat_indices, flat_indices = flatten_batch(inputs) # b', b k 1+h, b'
            compression = flat_inputs.numel() / inputs.numel()
            if compression < .75:
                this_flatting = True
                flat_inputs = flat_inputs[None,] # 1 b'
                cache_data.unflatten_indices = unflat_indices
                cache_data.flatten_indices = flat_indices
                position_ids = select_inflate_dim(position_ids.view(-1), flat_indices)[None,]
        inputs = inputs.view(-1, n_adds)  # bk 1+h
        # Base model forward pass
        output = model(
            inputs if not this_flatting else flat_inputs, include_embeds=True, position_ids=position_ids, cache_data=cache_data, **kwargs
        ) # 1 b' v
        logits, _, embeds = output # 1 n' v, 1 n' d
        next_vals = torch.argmax(logits, dim=-1)  # 1 n'

        if this_flatting:
            unflat_indices = unflat_indices.view(-1, unflat_indices.size(2))
            next_vals = select_inflate_dim(next_vals[0], unflat_indices) # bk 1+h
            embeds = select_inflate_dim(embeds[0], unflat_indices) # bk 1+h d
            # TODO: make more efficient by best guessing out of unflat indices rather than from here directly

        # Check correctness of speculator predictions
        test = inputs.roll(-1, 1).eq(next_vals).cumprod(1)
        n_correct = (
            test.sum(1).clamp(0, n_adds - 1).view(bsize, top_k)
        )  # clamp in case pred[0]==targ[-1]
        best_guess = n_correct.argmax(1)  # b
        best_guess_unflat = (
            best_guess.unsqueeze(1).expand(bsize, n_adds).unsqueeze(1)
        )  # b 1 1+h

        # Set global values to those of best guess
        next_vals = next_vals.view(bsize, top_k, n_adds).gather(1, best_guess_unflat).squeeze(1)  # b 1+h
        n_correct = n_correct.gather(1, best_guess.unsqueeze(1)).squeeze(1)  # b
        embeds = embeds.view(bsize, top_k, *embeds.size()[1:]).gather(
            1, best_guess_unflat.unsqueeze(3).expand(-1, -1, -1, embeds.size(2))
        ).squeeze(1)  # b 1+h d

        if verbose:
            test = inputs.view(bsize, top_k, n_adds).gather(1, best_guess_unflat).squeeze(1)
            for i, line in enumerate(test):
                print(
                    "Speculation:",
                    decode_obo(line, vinv),
                    "n_correct:",
                    n_correct[i].item(),
                )

        # free all worst candidates and keep best candidates as parents
        parent_sequence_ids = []
        for parent_index, child_sequence_ids in enumerate(child_sequence_ids_list):
            best_index = best_guess[parent_index].item()

            # free all bad candidates
            kv_cache_manager.free_sequences(child_sequence_ids[:best_index] + child_sequence_ids[best_index + 1:])

            # decrease the context length of the sequence which used to be sequence length + n_adds by the number of incorrect tokens
            # for the correct candidate
            best_sequence_id = child_sequence_ids[best_index]
            parent_sequence_ids.append(best_sequence_id)
            kv_cache_manager.remove_tokens(best_sequence_id, n_adds - n_correct[parent_index].item() - 1)

        # Toss any wrong speculator tokens
        next_vals_split = list(next_vals)
        next_vals_split = [
            next_vals_split[i][: n_correct[i] + 1] for i in range(len(next_vals_split))
        ]  # [b] h'
        n_gen += n_correct + 1
        embeds = embeds.gather(
            1, n_correct.view(-1, 1, 1).expand(-1, -1, embeds.size(2))
        )  # Grab last correct embed

        # Update results
        result = [
            torch.cat((result[i], next_vals_split[i]), dim=0) for i in range(bsize)
        ]
        inputs = torch.stack([line[-1:] for line in next_vals_split], dim=0)  # b 1

        if verbose:
            for line in result:
                print("Updated output:", decode_obo(line, vinv))
            print()

    for parent_sequence_id in parent_sequence_ids:
        prefix = kv_cache_manager.cbg_map[parent_sequence_id].prefix
        kv_cache_manager.free(parent_sequence_id)
        while prefix is not None:
            kv_cache_manager.free(prefix.sequence_id)
            prefix = prefix.prefix

    end_time = time.time()
    return result, n_steps, (end_time - start_time)
