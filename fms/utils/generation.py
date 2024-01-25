from typing import Any, Callable, List, MutableMapping, Union

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


def generate(
    prefill_model: Union[Callable, torch.nn.Module],
    decode_model: Union[Callable, torch.nn.Module],
    input_ids: torch.Tensor,
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
    # Preallocate cache
    kwargs["past_key_value_states"] = [
        [
            torch.zeros((input_ids.size(0), decode_model.config.kvheads // 8, max_new_tokens+input_ids.size(1), decode_model.config.emb_dim // decode_model.config.nheads), device=input_ids.device, dtype=torch.half),
            torch.zeros((input_ids.size(0), decode_model.config.kvheads // 8, max_new_tokens+input_ids.size(1), decode_model.config.emb_dim // decode_model.config.nheads), device=input_ids.device, dtype=torch.half),
        ] for _ in range(len(decode_model.layers))
    ]
    kwargs["use_cache"] = use_cache
    kwargs["position_ids"] = torch.arange(0, input_ids.shape[1], 1, dtype=torch.long, device=input_ids.device).repeat(input_ids.size(0), 1)
    prefill_model.preallocate_mask(input_ids.shape[1] + max_new_tokens)
    
    for token_idx in range(max_new_tokens):
        # print(kwargs["position_ids"])
        input_ids = next_input[:, -max_seq_len:]
        if token_idx == 0:
            output = prefill_model(input_ids, **kwargs)
        else:
            output = decode_model(input_ids, **kwargs)
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

        if kwargs["position_ids"].shape[1] > 1:
            kwargs["position_ids"] = torch.tensor([result.shape[1]-1], dtype=torch.long, device=result.device).repeat(input_ids.size(0), 1)
        else:
            kwargs["position_ids"] += 1

    if not batched:
        result = result[0]
    return result.clone().detach()


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
