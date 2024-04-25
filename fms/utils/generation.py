from typing import Any, Callable, List, MutableMapping, Union

import torch
import torch.nn.functional as F


def __create_prefill_mask(
    prompt_lengths: List[int], device: Union[str, torch.device]
) -> torch.Tensor:
    """Create a prefill mask based on

    Args:
        prompt_lengths: List[int]
            list of prompt lengths
        device: Union[str, torch.device]
            device to put mask on

    Returns:
    torch.Tensor
        a causal mask
    """
    max_tokens = max(prompt_lengths)

    is_pad_list = []
    for seq_len in prompt_lengths:
        pads = torch.zeros(max_tokens - seq_len, dtype=torch.bool, device=device)
        non_pads = torch.ones(seq_len, dtype=torch.bool, device=device)
        is_pad_list.append(torch.cat((pads, non_pads)))
    is_pad = torch.stack(is_pad_list)
    mask = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)
    return mask.tril(diagonal=0)


def __create_decode_mask(
    context_lengths: List[int], device: Union[str, torch.device]
) -> torch.Tensor:
    """create a decode mask

    Args:
        context_lengths: List[int]
            current context length of each sequence in the batch
        device: Union[str, torch.device]
            device to put mask on

    Returns:
    torch.Tensor
        a decode mask
    """
    max_context_length = max(context_lengths)

    is_not_pad_list = []
    for seq_len in context_lengths:
        ones = torch.ones(seq_len, dtype=torch.bool, device=device)
        zeroes = torch.zeros(
            max_context_length - seq_len, dtype=torch.bool, device=device
        )
        is_not_pad_list.append(torch.cat((zeroes, ones)))
    is_not_pad = torch.stack(is_not_pad_list)
    mask = is_not_pad.unsqueeze(-2)
    return mask


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
    input_ids: Union[torch.Tensor, List[torch.Tensor]],
    max_seq_len: int = 2048,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 10,
    do_sample: bool = True,
    num_beams: int = 1,
    use_cache: bool = False,
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
    if num_beams != 1:
        raise NotImplementedError("generate() does yet not support beam search")

    if isinstance(input_ids, torch.Tensor):
        if len(input_ids.shape) > 1:
            raise ValueError(
                "input ids should only have one dimension if given as a tensor"
            )
        model_input_lengths = [input_ids.size(0)]
        input_ids = input_ids.unsqueeze(0)
        is_batch = False
    else:
        bsize = len(input_ids)
        max_len = max([seq.size(0) for seq in input_ids])
        model_input_lengths = [seq.size(0) for seq in input_ids]
        input_ids = torch.stack(
            [
                F.pad(input_ids[i], (max_len - model_input_lengths[i], 0))
                for i in range(bsize)
            ]
        )
        is_batch = True

    result = input_ids
    next_input = input_ids
    kwargs: MutableMapping[str, Any] = dict()
    kwargs["past_key_value_states"] = None
    kwargs["use_cache"] = use_cache

    for i in range(max_new_tokens):
        input_ids = next_input[:, -max_seq_len:]

        # prefill
        if i == 0:
            kwargs["mask"] = __create_prefill_mask(
                model_input_lengths, input_ids.device
            )
        # decode
        else:
            # add 1 for generate
            model_input_lengths = [
                model_input_lengths + 1 for model_input_lengths in model_input_lengths
            ]
            kwargs["mask"] = __create_decode_mask(model_input_lengths, input_ids.device)

        output = model(input_ids, **kwargs)
        if use_cache:
            logits, past_key_value_states = output
            # TODO: this should go away when reduce-overhead issues are fixed, or
            # maybe could be moved into model code to be more portable.
            if contiguous_cache:
                kwargs["past_key_value_states"] = _make_cache_contiguous(
                    past_key_value_states
                )
            else:
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

        if use_cache:
            next_input = next_val
        else:
            next_input = result

    if not is_batch:
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
