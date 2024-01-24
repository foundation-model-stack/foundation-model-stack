import torch
from typing import Union, Callable, Tuple, Optional

from torch.nn.modules.loss import _Loss

from utils.generation import _make_cache_contiguous


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

def generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: torch.Tensor,
    pad_to_max_length: int = 2048,
    max_new_tokens: int = 256,
    use_cache: bool = False,
    contiguous_cache: bool = False,
    labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, _Loss]:
    """
    A trivial generate function that can be used for validation/testing in
    cases where HF is not available.
    We could add implementations for other types of generation, but this is
    enough for making sure a model is working.
    Does not implement batching nor beam search, but those could be added.

    Args:
        model: A function or nn.Module that takes a batch of input_ids and
            returns logits
        input_ids: torch.Tensor
            the input ids to the model
        pad_to_max_length: int
            all inputs will be padded to this length
        max_new_tokens: int
            total number of tokens to generate
        use_cache: requires that the model accept use_cache and
            past_key_value_states args in forward method.
        labels: torch.Tensor, optional
            the optional labels used to compute loss
    """

    if labels is not None:
        loss_fn = torch.nn.CrossEntropyLoss()


    sequence_length = input_ids.size(0)
    input_ids = left_pad(input_ids, pad_to_max_length)
    input_ids = input_ids.unsqueeze(0)

    result = input_ids
    next_input = input_ids
    past_key_values = None

    for i in range(max_new_tokens):
        input_ids = next_input[:, -pad_to_max_length:]

        # create mask
        if i == 0:
            is_pad = torch.tensor([1] * (pad_to_max_length - sequence_length) + [0] * sequence_length,
                                  device=input_ids.device).bool()
            mask = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)
            mask = mask.tril(diagonal=0)
        else:
            is_not_pad = torch.tensor([1] * input_ids.size(1), device=input_ids.device).unsqueeze(0).bool() # this will be 1 anyway for single token generation
            mask = is_not_pad.unsqueeze(-2)

        # model forward
        output = model(input_ids, mask=mask, use_cache=use_cache, past_key_values=past_key_values)

        # handle cache
        if use_cache:
            logits, past_key_values = output

            if contiguous_cache:
                past_key_values = _make_cache_contiguous(
                    past_key_values
                )
        else:
            logits = output
        logits = logits[:, -1, :]

        next_val = torch.argmax(logits, dim=-1).unsqueeze(0).t()
        result = torch.cat((result, next_val), dim=-1)

        if use_cache:
            next_input = next_val
        else:
            next_input = result

    if not batched:
        result = result[0]
    return result