import logging
import time
from typing import Any, Callable, List, Optional, Tuple, Union
from collections.abc import Iterable, MutableMapping

from fms.modules.attention import get_attention_type
import torch
import torch.nn.functional as F

from fms.modules.ssm import SSMCacheUnit


logger = logging.getLogger(__name__)


def pad_input_ids(
    input_ids_list: List[torch.Tensor],
    min_pad_length: int = 0,
    is_causal_mask=True,
    padding_side="left",
    position_ids_offset=0,
) -> Tuple[torch.Tensor, MutableMapping[str, Any]]:
    """
    Convert a list of Tensors to a rectangular tensor. Return extra padding kwargs for the position_ids and mask, since
    this will be required to properly handle the rectangular tensor for certain models.

    Parameters
    ----------
    input_ids_list: List[torch.Tensor]
        a list of Tensors of varied length
    min_pad_length: int
        pad to a min length provided. If the min_pad_length is less than the largest input_ids in the input_ids_list,
        padding will be determined based on the largest length input_ids.
    position_ids_offset: int
        some models are trained with position_ids that do not start at 0 but at pad_id + 1. The default parameter
        here will work for most models, but for example MPNet requires passing a real pad_id.

    Returns
    -------
    Tuple[torch.Tensor, MutableMapping[str, Any]]
        A rectangular 2d padded tensor and a mapping containing the mask and position_ids typically used in forward pass
        in fms models
        A mapping from mask to a 3d causal mask and from position_ids to a 2d rectangular position_ids tensor
    """
    max_len = max([min_pad_length] + [seq.size(0) for seq in input_ids_list])

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

        pos_ids_pads = pads
        pos_ids_seq = torch.arange(
            0, seq_len, dtype=torch.long, device=input_ids_i.device
        )
        if padding_side == "left":
            padded_input_ids_list.append(torch.cat((pads, input_ids_i)))
            mask_list.append(torch.cat((pads.bool(), non_pads)))
            position_ids_list.append(torch.cat((pos_ids_pads, pos_ids_seq)))
        elif padding_side == "right":
            padded_input_ids_list.append(torch.cat((input_ids_i, pads)))
            mask_list.append(torch.cat((non_pads, pads.bool())))
            position_ids_list.append(torch.cat((pos_ids_seq, pos_ids_pads)))
        else:
            raise NotImplementedError("padding_side must be 'right' or left'")

    input_ids = torch.stack(padded_input_ids_list)
    padding_kwargs = {}
    mask = torch.stack(mask_list)
    mask = mask.unsqueeze(-1) == mask.unsqueeze(-2)
    # this is a causal mask for generation
    if is_causal_mask:
        mask = mask.tril()
    mask = torch.where(mask.logical_not(), -torch.inf, 0.0)

    padding_kwargs["mask"] = mask
    # FIXME: this method should be per attn type (for now default it)

    position_ids = torch.stack(position_ids_list)
    position_ids += position_ids_offset
    padding_kwargs["position_ids"] = position_ids

    return input_ids, padding_kwargs


def __update_padding_kwargs(
    use_cache: bool, model_specific_kwargs: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Generic function to prepare any model specific keyword arguments"""
    # extend the attention mask
    attn_op = get_attention_type(**model_specific_kwargs)

    if "update_attn_kwargs" in attn_op:
        model_specific_kwargs = attn_op["update_attn_kwargs"](**model_specific_kwargs)

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


def _make_cache_contiguous(
    past_key_value_states: list[Iterable[torch.Tensor] | SSMCacheUnit],
) -> list[Iterable[torch.Tensor] | SSMCacheUnit]:
    # kv updates are required for torch.compile with
    # mode='reduce-overhead'
    n_kv_s: list[Iterable[torch.Tensor] | SSMCacheUnit] = []
    for layer_cache in past_key_value_states:
        if (
            isinstance(layer_cache, Iterable)
            and all(
                [
                    isinstance(cache_element, torch.Tensor)
                    for cache_element in layer_cache
                ]
            )
            and any(
                [not cache_element.is_contiguous() for cache_element in layer_cache]
            )
        ):
            n_kv_s.append(
                tuple(
                    [
                        cache_element.clone(
                            memory_format=torch.contiguous_format
                        ).detach()
                        for cache_element in layer_cache
                    ]
                )
            )
        else:
            n_kv_s.append(layer_cache)
    return n_kv_s


def _make_cache_dynamic(
    past_key_value_states: List[List[torch.Tensor]],
) -> List[List[torch.Tensor]]:
    # kv updates are required for torch.compile with
    # mode='reduce-overhead'
    for layer in past_key_value_states:
        if isinstance(layer, Iterable):
            for tensor in layer:
                torch._dynamo.mark_dynamic(tensor, 2)
    return past_key_value_states


def generate(
    model: Union[Callable, torch.nn.Module],
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
    timing: str = "",
    prepare_model_inputs_hook: Optional[
        Callable[
            [int, torch.Tensor, MutableMapping[str, Any]],
            Tuple[torch.Tensor, MutableMapping[str, Any]],
        ]
    ] = None,
    post_iteration_hook: Optional[
        Callable[
            [int, torch.Tensor, torch.Tensor, MutableMapping[str, Any]],
            Tuple[torch.Tensor, MutableMapping[str, Any]],
        ]
    ] = None,
    extra_kwargs: Optional[MutableMapping[str, Any]] = None,
):
    """
    A trivial generate function that can be used for validation/testing in
    cases where HF is not available.
    We could add implementations for other types of generation, but this is
    enough for making sure a model is working.
    Does not implement beam search, but this can be added.

    Args:
        model: A function or nn.Module that takes a batch of input_ids and
            returns logits
        input_ids: a rectangular tensor of input_ids (batch x seq)
        max_seq_len: the sequence length of the model
        max_new_tokens: max tokens to generate
        temperature: temperature of softmax when sampling
        top_k: only search among top k tokens
        do_sample: multinomial sampling. False for greedy.
        num_beams: TODO: support beam search
        use_cache: requires that the model accept use_cache and
            past_key_value_states args in forward method.
        contiguous_cache: ensures the cache is contiguous in device memory
        eos_token_id: the optional token id representing the end of sequence
        timing: whether to measure timings: "per-token" for each token generation time,
            "e2e" for full generation loop. Both options make `generate` return a tuple
            with the following information:
            - "per-token": Array with `max_new_tokens` time measurements (in s)
            - "e2e": Array with a single e2e generation loop time measurement (in s)
        prepare_model_inputs_hook: a function that will get called immediately before model forward.
            It must have the following signature: f(int generate_iteration, Tensor input_ids, Dict kwargs) ->
            Tuple[Tensor input_ids, Dict kwargs]. If it is defined, will replace input_ids
            and kwargs to next model forward based on the contents of the function.
        post_iteration_hook: a function that will get called after each iteration.
            It must have the following signature: f(int token_position, Tensor logits, Tensor next_val, Dict kwargs) ->
            Tuple[Tensor next_val, Dict kwargs]. If it is defined, will replace next_val
            and kwargs based on the contents of the function.
        extra_kwargs: an optional mapping of additional kwargs to pass to the model.
            For example: if extra_kwargs contains position_ids and mask keys, these
            model parameters will be updated as-appropriate for each token generated.
    """
    if num_beams != 1:
        raise NotImplementedError("generate() does yet not support beam search")

    kwargs: MutableMapping[str, Any] = dict()
    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)

    if isinstance(input_ids, torch.Tensor):
        is_batch = len(input_ids.shape) > 1
        # our model requires batch dimension
        if not is_batch:
            input_ids = input_ids.unsqueeze(0)
    else:
        raise TypeError("input_ids must be one of Tensor or List")

    eos_found = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
    )

    result = input_ids
    next_input = input_ids
    kwargs["past_key_value_states"] = None
    kwargs["use_cache"] = use_cache
    # if we didn't specify last_n_tokens and only_last_token is set to True, set last_n_tokens to 1, otherwise use default
    # we do this since the output shape of only_last_token is different and therefore would change the logic in generate
    if "last_n_tokens" not in kwargs and kwargs.get("only_last_token", False):
        kwargs["last_n_tokens"] = 1

    prompt_length = input_ids.shape[1]

    if timing != "":
        times: List[float] = []
        start_time = time.time()

    eos_reached: bool = False

    for i in range(max_new_tokens):
        input_ids = next_input[:, -max_seq_len:]

        # prepare any padding keyword arguments
        # iteration 0 is the prefill step (cache has not been filled yet), so no need to extend the mask/position_ids
        if i > 0:
            kwargs = __update_padding_kwargs(use_cache, kwargs)

        if prepare_model_inputs_hook is not None:
            input_ids, kwargs = prepare_model_inputs_hook(i, input_ids, kwargs)

        output = model(input_ids, **kwargs)
        if use_cache:
            logits, past_key_value_states = output
            # TODO: this should go away when reduce-overhead issues are fixed, or
            # maybe could be moved into model code to be more portable.
            kwargs["past_key_value_states"] = past_key_value_states
            if contiguous_cache:
                kwargs["past_key_value_states"] = _make_cache_contiguous(
                    kwargs["past_key_value_states"]
                )
            if torch._dynamo.config.dynamic_shapes:
                kwargs["past_key_value_states"] = _make_cache_dynamic(
                    kwargs["past_key_value_states"]
                )
        else:
            logits = output

        # always get last now since we still have this dim
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

        if post_iteration_hook is not None:
            next_val, kwargs = post_iteration_hook(
                i + prompt_length, logits, next_val, kwargs
            )

        result = torch.cat((result, next_val), dim=-1)

        # avoid continuing to generate if all have reached EOS
        if eos_token_id is not None:
            eos_found = torch.logical_or(eos_found, next_val == eos_token_id)
            if torch.sum(eos_found) == input_ids.shape[0]:
                eos_reached = True

        if use_cache:
            next_input = next_val
        else:
            next_input = result

        if timing == "per-token":
            if input_ids.device.type == "cuda":
                torch.cuda.synchronize()
            current_token_time = time.time() - start_time
            times.append(current_token_time)
            start_time = time.time()

        if eos_reached:
            break

    if timing == "e2e":
        if input_ids.device.type == "cuda":
            torch.cuda.synchronize()
        e2e_time = time.time() - start_time
        times.append(e2e_time)

    if not is_batch:
        result = result[0]

    if timing != "":
        return result, times
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


def trim_prefix(result: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Helper function to return a trimmed sequence of token IDs where
    all padding tokens (always 0 on our code) are removed.

    Examples:
    [0 0 0 0 1 2 3 4] with pad_token_id = 0 returns [1 2 3 4]
    [0 0 0 0 1 2 3 4] with pad_token_id = 5 returns [0 0 0 0 1 2 3 4]
    [1 2 3 4 0 1] with pad_token_id = 0 returns [1 2 3 4 0 1]

    Args:
    result: A 1D sequence of tokens
    pad_token_id: Token ID that will be trimmed from the start of the
        sequence
    """
    if result[0] != pad_token_id:
        return result
    output_diff = (result != pad_token_id).diff()
    first_real_token_idx = torch.where(output_diff > 0)
    if first_real_token_idx[0].numel() == 0:
        return result
    bos_index = first_real_token_idx[0][0]
    result = result[bos_index + 1 :]
    return result
