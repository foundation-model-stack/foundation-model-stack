import math
import random
import time
from typing import Any, Callable, List, MutableMapping, Optional, Tuple, Union
import torch

import torch.nn.functional as F

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

    if not hasattr(model, "config"):
        raise ValueError("model must have a config")

    eos_found = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
    )

    result = input_ids
    next_input = input_ids
    NUM_BLOCKS = 100
    BLOCK_SIZE = 64
    if hasattr(model, "head"):
        model_dtype = model.head.weight.dtype
    elif hasattr(model, "shared"):
        model_dtype = model.shared.head.weight.dtype
    else:
        model_dtype = torch.float32

    nheads = model.config.nheads
    if hasattr(model.config, "kvheads"):
        kvheads = model.config.kvheads
    elif hasattr(model.config, "multiquery_attn"):
        kvheads = 1 if model.config.multiquery_attn else model.config.nheads
    else:
        kvheads = nheads

    if hasattr(model, "distributed_strategy"):
        tensor_parallel_size = (
            model.distributed_strategy.group.size()
            if hasattr(model.distributed_strategy, "group")
            else 1
        )
    else:
        raise ValueError("model must have a distributed_strategy")

    kvheads = kvheads // tensor_parallel_size if kvheads > 1 else kvheads
    head_size = model.config.emb_dim // nheads
    # kwargs["attn_name"] = "spyre_paged_attn"
    kwargs["past_key_value_states"] = [
        (
            torch.zeros(NUM_BLOCKS, BLOCK_SIZE, kvheads, head_size, dtype=model_dtype),
            torch.zeros(NUM_BLOCKS, BLOCK_SIZE, kvheads, head_size, dtype=model_dtype),
        )
        for _ in range(model.config.nlayers)
    ]
    kwargs["block_table"] = None
    block_numbers = [i for i in range(NUM_BLOCKS)]
    random.seed(0)
    random.shuffle(block_numbers)
    left_padded_prompt_mask = (kwargs["position_ids"] == 0).sum(dim=1) - 1
    current_context_lengths = (kwargs["position_ids"] != 0).sum(dim=1) + 1
    current_tkv_mask = left_padded_prompt_mask + current_context_lengths
    slot_mapping = []
    block_table = []
    for seq_i in input_ids:
        block_table_i = []
        slot_mapping_i = []
        for pos_i in range(seq_i.size(0)):
            if pos_i % BLOCK_SIZE == 0:
                block_number = block_numbers.pop(0)
                block_table_i.append(block_number)
            block_offset = pos_i % BLOCK_SIZE
            slot = block_number * BLOCK_SIZE + block_offset
            slot_mapping_i.append(slot)
        slot_mapping.append(slot_mapping_i)
        block_table.append(block_table_i)
    kwargs["slot_mapping"] = torch.tensor(slot_mapping, dtype=torch.int64)
    kwargs["current_tkv_mask"] = None
    kwargs["left_padded_prompt_mask"] = None
    kwargs["use_cache"] = use_cache

    prompt_length = input_ids.shape[1]

    if timing != "":
        times: List[float] = []
        start_time = time.time()

    for i in range(max_new_tokens):
        input_ids = next_input[:, -max_seq_len:]

        # prepare any padding keyword arguments
        # iteration 0 is the prefill step (cache has not been filled yet), so no need to extend the mask/position_ids
        if i > 0:
            kwargs["mask"] = None
            kwargs["position_ids"] = kwargs["position_ids"][:, -1:] + 1
            pos_i = result.size(1) - 1
            if pos_i % BLOCK_SIZE == 0:
                for block_table_i in block_table:
                    block_number = block_numbers.pop(0)
                    block_table_i.append(block_number)
            block_offset = pos_i % BLOCK_SIZE

            slot_mapping = []
            for block_table_i in block_table:
                slot = block_table_i[-1] * BLOCK_SIZE + block_offset
                slot_mapping.append([slot])
            kwargs["block_table"] = torch.tensor(block_table, dtype=torch.int64)
            kwargs["slot_mapping"] = torch.tensor(slot_mapping, dtype=torch.int64)
            current_tkv_mask = current_tkv_mask + 1
            kwargs["current_tkv_mask"] = current_tkv_mask
            kwargs["left_padded_prompt_mask"] = left_padded_prompt_mask

        # prefill
        if i == 0:
            kwargs["mask"] = kwargs["mask"].unsqueeze(1)

            outputs_list = []
            current_kv_cache = kwargs["past_key_value_states"]
            for seq_i in range(input_ids.size(0)):
                input_ids_i = input_ids[seq_i].unsqueeze(0)
                slot_mapping_i = kwargs["slot_mapping"][seq_i].unsqueeze(0)
                position_ids_i = kwargs["position_ids"][seq_i].unsqueeze(0)
                mask_i = kwargs["mask"][seq_i].unsqueeze(0)

                # batch dynamic
                torch._dynamo.mark_static(input_ids_i, 0)
                torch._dynamo.mark_static(slot_mapping_i, 0)
                torch._dynamo.mark_static(position_ids_i, 0)
                torch._dynamo.mark_static(mask_i, 0)

                # seq dynamic
                torch._dynamo.mark_dynamic(input_ids_i, 1)
                torch._dynamo.mark_dynamic(slot_mapping_i, 1)
                torch._dynamo.mark_dynamic(position_ids_i, 1)
                torch._dynamo.mark_dynamic(mask_i, 2)
                torch._dynamo.mark_dynamic(mask_i, 3)

                # for k_i, v_i in current_kv_cache:
                #     torch._dynamo.mark_dynamic(k_i, 0)
                #     torch._dynamo.mark_dynamic(v_i, 0)

                only_last_token = kwargs.get("only_last_token", False)

                output, current_kv_cache = model(
                    input_ids_i,
                    slot_mapping=slot_mapping_i,
                    position_ids=position_ids_i,
                    mask=mask_i,
                    past_key_value_states=current_kv_cache,
                    use_cache=kwargs["use_cache"],
                    only_last_token=only_last_token,
                    # attn_name=kwargs["attn_name"],
                )

                outputs_list.append(output[0].squeeze(0))

            output = (torch.stack(outputs_list), current_kv_cache)

        # decode
        else:
            # mask is no longer used here

            # batch
            torch._dynamo.mark_dynamic(input_ids, 0)
            torch._dynamo.mark_dynamic(kwargs["block_table"], 0)
            torch._dynamo.mark_dynamic(kwargs["slot_mapping"], 0)
            torch._dynamo.mark_dynamic(kwargs["position_ids"], 0)
            torch._dynamo.mark_dynamic(kwargs["current_tkv_mask"], 0)
            torch._dynamo.mark_dynamic(kwargs["left_padded_prompt_mask"], 0)

            # seq
            torch._dynamo.mark_static(input_ids, 1)  # always 1
            torch._dynamo.mark_dynamic(kwargs["block_table"], 1)
            torch._dynamo.mark_static(kwargs["slot_mapping"], 1)  # always 1
            torch._dynamo.mark_static(kwargs["position_ids"], 1)  # always 1

            # for k_i, v_i in kwargs["past_key_value_states"]:
            #     torch._dynamo.mark_dynamic(k_i, 0)
            #     torch._dynamo.mark_dynamic(v_i, 0)

            output = model(input_ids, **kwargs)
        if use_cache:
            logits, past_key_value_states = output
            # TODO: this should go away when reduce-overhead issues are fixed, or
            # maybe could be moved into model code to be more portable.
            kwargs["past_key_value_states"] = past_key_value_states
        else:
            logits = output

        if "only_last_token" not in kwargs:
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