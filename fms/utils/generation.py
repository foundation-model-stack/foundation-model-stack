import inspect
import time
from typing import Any, Callable, List, MutableMapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def __prepare_list_input(
    input_ids_list: List[torch.Tensor], model: Union[Callable, torch.nn.Module]
) -> Tuple[torch.Tensor, MutableMapping[str, Any]]:
    """
    Convert the list of Tensors to a rectangular tensor. Return extra kwargs for the position_ids and mask, since this
    will be required to properly handle the rectangular tensor for certain models.
    """
    min_len = min([seq.size(0) for seq in input_ids_list])
    max_len = max([seq.size(0) for seq in input_ids_list])

    if isinstance(model, nn.Module):
        forward_function = model.forward
    else:
        forward_function = model

    params = inspect.signature(forward_function).parameters.keys()
    extra_kwargs = {}
    needs_mask = "mask" in params and min_len != max_len
    needs_position_ids = "position_ids" in params and min_len != max_len

    padded_input_ids_list = []
    mask_list = []
    position_ids_list = []
    for input_ids_i in input_ids_list:
        seq_len = input_ids_i.size(0)
        pads = torch.zeros(
            max_len - seq_len, dtype=torch.long, device=input_ids_i.device
        )
        non_pads = torch.ones(seq_len, dtype=torch.bool, device=input_ids_i.device)

        # Setting this to 0, however if 0 is the eos, we will end up truncating the output if using truncate_after_eos
        # once this workflow works for nested tensor, this can probably be removed
        padded_input_ids_list.append(torch.cat((pads, input_ids_i)))

        # computing this as it's lightweight but could potentially be skipped
        mask_list.append(torch.cat((pads.bool(), non_pads)))

        pos_ids_pads = pads
        pos_ids_seq = torch.arange(
            0, seq_len, dtype=torch.long, device=input_ids_i.device
        )
        position_ids_list.append(torch.cat((pos_ids_pads, pos_ids_seq)))

    input_ids = torch.stack(padded_input_ids_list)
    if needs_mask:
        mask = torch.stack(mask_list)
        # this is a causal mask for generation
        mask = (mask.unsqueeze(-1) == mask.unsqueeze(-2)).tril()
        extra_kwargs["mask"] = mask

    if needs_position_ids:
        position_ids = torch.stack(position_ids_list)
        extra_kwargs["position_ids"] = position_ids

    return input_ids, extra_kwargs


def __update_padding_kwargs(
    use_cache: bool, model_specific_kwargs: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Generic function to prepare any model specific keyword arguments"""
    # extend the attention mask
    mask = model_specific_kwargs.get("mask", None)
    if mask is not None:
        # get the last row of the 3d mask
        mask = mask[:, -1:, :]
        # extend the mask one slot
        mask = torch.cat(
            (
                mask,
                torch.ones(mask.size(0), 1, 1, dtype=torch.bool, device=mask.device),
            ),
            dim=2,
        )
        model_specific_kwargs["mask"] = mask

    # extend the position_ids
    position_ids = model_specific_kwargs.get("position_ids", None)
    if position_ids is not None:
        if use_cache:
            position_ids = position_ids[:, -1:] + 1
        else:
            position_ids = torch.cat(
                (position_ids, position_ids[:, -1:] + 1),
                dim=1,
            )
        model_specific_kwargs["position_ids"] = position_ids
    return model_specific_kwargs


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


past_key_value_states_g = None


def generate(
    prefill_model: Union[Callable, torch.nn.Module],
    decode_model: Union[Callable, torch.nn.Module],
    input_ids: Union[torch.Tensor, List[torch.Tensor]],
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
    if num_beams != 1:
        raise NotImplementedError("generate() does yet not support beam search")

    kwargs: MutableMapping[str, Any] = dict()
    # if the inputs are a tensor, we assume they are all non-pad ids and include entire context length
    if isinstance(input_ids, torch.Tensor):
        is_batch = len(input_ids.shape) > 1
        # our model requires batch dimension
        if not is_batch:
            input_ids = input_ids.unsqueeze(0)
    # if the inputs are a list, they may be made up of differently sized tensors
    # in the case where the tensors are of different sizes, proper position ids and pads will be created
    elif isinstance(input_ids, List):
        is_batch = len(input_ids) > 1
        input_ids, kwargs = __prepare_list_input(input_ids, prefill_model)
    else:
        raise TypeError("input_ids must be one of Tensor or List")

    eos_found = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
    )

    result = input_ids
    next_input = input_ids
    kwargs["past_key_value_states"] = None
    kwargs["use_cache"] = use_cache

    global past_key_value_states_g

    if past_key_value_states_g is not None:
        for layer_idx, cache_layer in enumerate(past_key_value_states_g):
            for tensor_idx, kv_tensor in enumerate(cache_layer):
                kv_tensor.fill_(0)
        kwargs["past_key_value_states"] = past_key_value_states_g
    
    total_start = time.time()

    for i in range(max_new_tokens):
        input_ids = next_input[:, -max_seq_len:]

        # prepare any padding keyword arguments
        # iteration 0 is the prefill step (cache has not been filled yet), so no need to extend the mask/position_ids
        if i > 0:
            kwargs = __update_padding_kwargs(use_cache, kwargs)
            output = decode_model(input_ids, **kwargs)
        else:
            output = prefill_model(input_ids, **kwargs)
        if use_cache:
            logits, past_key_value_states = output
            if i == 0:
                if past_key_value_states_g is None:
                    past_key_value_states_g = past_key_value_states
                    for cache_layer in past_key_value_states_g:
                        for kv_tensor in cache_layer:
                            torch._dynamo.mark_static_address(kv_tensor)
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
    total_time = time.time() - total_start

    print(f"Total time: {total_time}")
    if not is_batch:
        result = result[0]
    return result


def truncate_after_eos(
    result: torch.Tensor, eos_token_id: Union[int, "Any | None"]
) -> torch.Tensor:
    """
    Helper function to return a truncated sequence of token IDs stopping at
    (and including) the 'end of sentence' token.
    Currently only handles unbatched sequences.
    """
    if eos_token_id is None:
        return result
    eos_idx = torch.where(result == eos_token_id)
    eos_index = eos_idx[0]
    if eos_index.shape[0] >= 1:
        index = eos_index[0]
        result = result[: index + 1]
    return result
