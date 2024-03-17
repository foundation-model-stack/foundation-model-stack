from typing import Any, Callable, List, MutableMapping, Optional, Union

import torch
import torch.nn.functional as F


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


def left_pad(input_ids: torch.Tensor, max_length: int, pad_id: int = 0) -> torch.Tensor:
    """
    left pad an input_ids tensor

    Parameters
    ----------
    input_ids: torch.Tensor
        input ids corresponding to a single sequence in a batch
    max_length: int
        the max length to pad to
    pad_id: int
        the token to set as a pad in the resulting tensor

    Returns
    -------
    torch.Tensor
        a left padded tensor
    """
    pads_tensor = torch.tensor(
        [pad_id] * (max_length - input_ids.size(0)),
        device=input_ids.device,
        dtype=torch.long,
    )
    return torch.cat((pads_tensor, input_ids))


def __create_attention_mask(
    input_ids: torch.Tensor, use_cache: bool, is_prompt: bool, pad_id: int = -1
) -> Optional[torch.Tensor]:
    """
    Optionally create an attention mask if one is required

    Parameters
    ----------
    input_ids: torch.Tensor
        the input ids to the model's forward method
    use_cache: bool
        if True caching is being used, otherwise caching is not being used
    is_prompt: bool
        if True, input_ids correspond to the prompt, otherwise input_ids contain generated tokens
    pad_id: int
        the pad id used when padding. If the pad_id is -1, no padding is being used, otherwise assume some padding

    Returns
    -------
    Optional[torch.Tensor]
        if is_prompt, no use_cache, batch required padding => a bool Tensor is created for the mask
        otherwise None is returns and the mask will be handled internally as part of sdpa
    """
    # compute the attention mask
    # always need a mask if this is the prompt or if not using a cache
    if is_prompt or not use_cache:
        is_pad = input_ids == pad_id
        mask = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)
        mask = mask.tril(diagonal=0)
    # otherwise if the batch required padding, we need to account for the padding tokens when using cache
    elif pad_id != -1:
        is_not_pad = input_ids != pad_id
        mask = is_not_pad.unsqueeze(-2)
    else:
        mask = None
    return mask


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
    pad_id: int = 0,
):
    """
    A trivial generate function that can be used for validation/testing in
    cases where HF is not available.
    We could add implementations for other types of generation, but this is
    enough for making sure a model is working.
    Does not implement beam search, but could be added.

    Args:
        model: A function or nn.Module that takes a batch of input_ids and
            returns logits
        input_ids: torch.Tensor or list[torch.Tensor]
            the input ids to the model. If the input_ids are a tensor with dimensionality greater than 1, it will be
            assumed to be a batch where each sequence is the same length. If the input_ids are a list of tensors, this
            will make the assumption that batch generation is being done and if sequences are of different length, a
            mask will be used.
        prefix: A tensor of token IDs.
        max_seq_len: the sequence length of the model
        max_new_tokens: max tokens to generate
        temperature: temperature of softmax when sampling
        top_k: only search among top k tokens
        do_sample: multinomial sampling. False for greedy.
        num_beams: TODO: support beam search
        use_cache: requires that the model accept use_cache and
            past_key_value_states args in forward method.
        pad_id: int
            the pad token id to use in case of batch generation where sequences are of different lengths
    """
    batched = False
    if num_beams != 1:
        raise NotImplementedError("generate() does yet not support beam search")
    if isinstance(input_ids, torch.Tensor):
        # TODO: nested tensors
        if input_ids.dim() != 1:
            batched = True
    elif isinstance(input_ids, list):
        max_length = max(p.size(0) for p in input_ids)
        requires_padding = any(p.size(0) < max_length for p in input_ids)
        if requires_padding:
            input_ids = [left_pad(p, max_length, pad_id) for p in input_ids]
        else:
            pad_id = -1
        input_ids = torch.stack(input_ids, dim=0)
        batched = True
    else:
        raise RuntimeError("generate() requires a tensor of token ids as the prefix")

    if not batched:
        input_ids = input_ids.unsqueeze(0)

    result = input_ids
    next_input = input_ids
    kwargs: MutableMapping[str, Any] = dict()
    kwargs["past_key_value_states"] = None
    kwargs["use_cache"] = use_cache

    for i in range(max_new_tokens):
        input_ids = next_input[:, -max_seq_len:]

        # create the attention mask
        kwargs["mask"] = __create_attention_mask(
            input_ids=result, use_cache=use_cache, is_prompt=i == 0, pad_id=pad_id
        )

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
