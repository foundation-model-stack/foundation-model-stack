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


def _repetition_penalty(input_ids, logits, penalty=1.0, compound=False):
    """
    Apply a repetition penalty to logits based on if the predicted outputs
    have already appeared in input_ids.

    Args:
    input_ids: the prior sequence to avoid repeating.
    logits: the predicted scores for the next token
    penalty: values greater than 1 penalize repetition. Values less than 1
                encourage repetition.
    compound: In the original paper (https://arxiv.org/pdf/1909.05858.pdf) the
                repetition penalty is only applied once if the token has been
                seen previously. When compound=True, apply the penalty multiple
                times, once per occurrance of the token.
    """
    if penalty == 1.0:
        return logits

    if compound:
        # the penalty will be same size as logits (e.g. batch x vocab size)
        result = torch.zeros_like(logits)
        # we add 1 at the index of input_ids for each occurance of input_ids
        result.scatter_add_(
            dim=-1,
            index=input_ids,
            src=input_ids.new_ones((), dtype=logits.dtype).expand_as(input_ids),
        )
        # i.e. if the penalty is 1.2, an input_id that appears 2x would be penalized 1.2*1.2=1.44
        penalty = penalty**result
        result = torch.where(logits < 0, logits * penalty, logits / penalty)
    else:
        score = torch.gather(logits, -1, input_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        result = logits.scatter(-1, input_ids, score)
    return result


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
    contiguous_cache: bool = False,
    repetition_penalty: float = 1.0,
    compound_repetition_penalty: bool = False,
    debug=False,
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
        repetition_penalty: A penalty to apply to repeatedly occurring tokens.
                    Values greater than one discourage repetition.
        compound_repetition_penalty: Whether to penalize more for multiple prior
                    occurrances of a token.
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
    kwargs["past_key_value_states"] = None
    kwargs["use_cache"] = use_cache

    for _ in range(max_new_tokens):
        input_ids = next_input[:, -max_seq_len:]
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

        logits = _repetition_penalty(
            result, logits, repetition_penalty, compound_repetition_penalty
        )

        if do_sample:
            # get logits from last value in sequence and scale
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
