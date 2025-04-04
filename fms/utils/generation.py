import logging
import time
from typing import Any, Callable, List, MutableMapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_nested_block_mask


logger = logging.getLogger(__name__)


def pad_input_ids(
    input_ids_list: List[torch.Tensor], min_pad_length: int = 0
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
        padded_input_ids_list.append(torch.cat((pads, input_ids_i)))

        # computing this as it's lightweight but could potentially be skipped
        mask_list.append(torch.cat((pads.bool(), non_pads)))

        pos_ids_pads = pads
        pos_ids_seq = torch.arange(
            0, seq_len, dtype=torch.long, device=input_ids_i.device
        )
        position_ids_list.append(torch.cat((pos_ids_pads, pos_ids_seq)))

    input_ids = torch.stack(padded_input_ids_list)
    padding_kwargs = {}
    mask = torch.stack(mask_list)
    # this is a causal mask for generation
    mask = (mask.unsqueeze(-1) == mask.unsqueeze(-2)).tril()
    mask = torch.where(mask.logical_not(), -torch.inf, 0.0)
    padding_kwargs["mask"] = mask

    position_ids = torch.stack(position_ids_list)
    padding_kwargs["position_ids"] = position_ids

    return input_ids, padding_kwargs

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def true_mask(b, h, q_idx, kv_idx):
    return True

def __update_padding_kwargs(
    use_cache: bool, input_ids, past_key_value_states_orig, model_specific_kwargs: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Generic function to prepare any model specific keyword arguments"""
    # Extend the narrow view of the kv-cache
    kv_cache = model_specific_kwargs.get("past_key_value_states", None)
    for layer_i, layer in enumerate(kv_cache):
        batch_size = layer[0].lengths().size(0)
        # model_specific_kwargs["past_key_value_states"][layer_i] = [
        #     torch.nested.narrow(k_cache.view(batch_size, -1, *k_cache.size()[-2:]), 1, 0, layer[0].lengths() + 1, layout=torch.jagged),
        #     torch.nested.narrow(v_cache.view(batch_size, -1, *v_cache.size()[-2:]), 1, 0, layer[1].lengths() + 1, layout=torch.jagged),
        # ]
        kv_cache[layer_i] = [
            torch.nested.narrow(past_key_value_states_orig[layer_i][0], 1, 0, layer[0].lengths() + 1, layout=torch.jagged),
            torch.nested.narrow(past_key_value_states_orig[layer_i][1], 1, 0, layer[1].lengths() + 1, layout=torch.jagged),
        ]

    # extend the position_ids
    position_ids = model_specific_kwargs.get("position_ids", None)
    if position_ids is not None:
        if position_ids.is_nested:
            position_ids = torch.stack([t[-1].unsqueeze(0) for t in position_ids.unbind()]) + 1
            if not use_cache:
                position_ids = torch.cat(
                    (position_ids, position_ids[:, -1:] + 1),
                    dim=1,
                )
            position_ids = torch.nested.nested_tensor_from_jagged(
                values=position_ids.flatten(),
                offsets=torch.arange(0, position_ids.size(0)+1, device=position_ids.device, dtype=torch.int64),
                min_seqlen=1,
                max_seqlen=1,
            )
        else:
            if use_cache:
                position_ids = position_ids[:, -1:] + 1
            else:
                position_ids = torch.cat(
                    (position_ids, position_ids[:, -1:] + 1),
                    dim=1,
                )
        model_specific_kwargs["position_ids"] = position_ids

    # extend the attention mask
    mask = model_specific_kwargs.get("mask", None)
    # if mask is not None:
    #     # get the last row of the 3d mask
    #     mask = mask[:, -1:, :]
    #     # extend the mask one slot
    #     mask = torch.cat(
    #         (
    #             mask,
    #             torch.zeros(mask.size(0), 1, 1, device=mask.device),
    #         ),
    #         dim=2,
    #     )
    #     model_specific_kwargs["mask"] = mask
    if mask is not None:
        example_cache = kv_cache[0][0]
        # print(input_ids.device, example_cache.device)
        batch_size = past_key_value_states_orig[0][0].size(0)
        mask = create_nested_block_mask(true_mask, batch_size, 32, input_ids, kv_nt=example_cache, _compile=True)
        model_specific_kwargs["mask"] = mask
        # print(f"Mask: {mask[0][0].to_string(grid_size=(-1, -1), limit=1000)} {mask[1][0].to_string(grid_size=(-1, -1), limit=1000)}")
    return model_specific_kwargs


def _make_cache_contiguous(
    past_key_value_states: List[List[torch.Tensor]],
) -> List[List[torch.Tensor]]:
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
    return n_kv_s


def _make_cache_dynamic(
    past_key_value_states: List[List[torch.Tensor]],
) -> List[List[torch.Tensor]]:
    # kv updates are required for torch.compile with
    # mode='reduce-overhead'
    for layer in past_key_value_states:
        if isinstance(layer, tuple):
            for tensor in layer:
                torch._dynamo.mark_dynamic(tensor, 1)
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

    # Pre-allocate kv cache and use nested narrow
    past_key_value_states_orig = []
    kwargs["past_key_value_states"] = []
    head_dim = model.config.emb_dim // model.config.nheads
    kv_heads = model.config.nheads if model.config.kvheads == 0 else model.config.kvheads
    for layer_i in range(model.config.nlayers):
        past_key_value_states_orig.append([
            torch.zeros((input_ids.size(0), max_seq_len, kv_heads, head_dim), device=input_ids.device, dtype=torch.bfloat16),
            torch.zeros((input_ids.size(0), max_seq_len, kv_heads, head_dim), device=input_ids.device, dtype=torch.bfloat16)
        ])
        kwargs["past_key_value_states"].append([
            torch.nested.narrow(past_key_value_states_orig[-1][0], 1, 0, input_ids.offsets().diff(), layout=torch.jagged),
            torch.nested.narrow(past_key_value_states_orig[-1][1], 1, 0, input_ids.offsets().diff(), layout=torch.jagged)
        ])

    kwargs["mask"] = create_nested_block_mask(causal_mask, input_ids.size(0), model.config.nheads, q_nt=input_ids, kv_nt=kwargs["past_key_value_states"][0][0], _compile=True)
    kwargs["use_cache"] = use_cache

    prompt_length = input_ids.shape[1]

    if timing != "":
        times: List[float] = []
        start_time = time.time()

    for i in range(max_new_tokens):
        print(f"Iteration {i}")
        # input_ids = next_input[:, -max_seq_len:]
        input_ids = next_input

        # prepare any padding keyword arguments
        # iteration 0 is the prefill step (cache has not been filled yet), so no need to extend the mask/position_ids
        if i > 0:
            input_ids = torch.nested.nested_tensor_from_jagged(
                values=input_ids.flatten(),
                offsets=torch.arange(0, input_ids.size(0)+1, device=input_ids.device, dtype=torch.int64),
                min_seqlen=1,
                max_seqlen=1,
            )
            kwargs = __update_padding_kwargs(use_cache, input_ids, past_key_value_states_orig, kwargs)
        output = model(input_ids, **kwargs)
        if use_cache:
            logits, past_key_value_states = output
            # TODO: this should go away when reduce-overhead issues are fixed, or
            # maybe could be moved into model code to be more portable.
            # kwargs["past_key_value_states"] = past_key_value_states
            # if contiguous_cache:
            #     kwargs["past_key_value_states"] = _make_cache_contiguous(
            #         kwargs["past_key_value_states"]
            #     )
            # if torch._dynamo.config.dynamic_shapes:
            #     kwargs["past_key_value_states"] = _make_cache_dynamic(
            #         kwargs["past_key_value_states"]
            #     )
            # # Grow kv cache outside compiled region
            # if kwargs["past_key_value_states"][0][0].is_nested:
            #     example_kv_cache = kwargs["past_key_value_states"][0][0]
            #     kv_cache_size = example_kv_cache.size()
            #     dummy_tensor = torch.zeros(kv_cache_size[0], 1, *kv_cache_size[2:], device=example_kv_cache.device, dtype=example_kv_cache.dtype)
            #     for i, layer in enumerate(past_key_value_states):
            #         past_key_value_states[i] = (
            #             torch.cat([layer[0], dummy_tensor], dim=1),
            #             torch.cat([layer[1], dummy_tensor], dim=1)
            #         )

        else:
            logits = output

        if "only_last_token" not in kwargs:
            if logits.is_nested:
                logits = torch.stack([t[-1] for t in logits.unbind()])
            else:
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

        result = torch.cat((result, next_val), dim=1)

        # avoid continuing to generate if all have reached EOS
        if eos_token_id is not None:
            eos_found = torch.logical_or(eos_found, next_val == eos_token_id)
            if torch.sum(eos_found) == input_ids.shape[0]:
                break

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

        torch.cuda.memory._dump_snapshot(f"memory_dump_{i}.pickle")

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
