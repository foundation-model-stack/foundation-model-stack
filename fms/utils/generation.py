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


def speculative_generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: torch.LongTensor,
    speculator: Union[Callable, torch.nn.Module],
    max_seq_len: int = 2048,
    new_tokens: int = 256,
    top_k: int = 5,
    threshes = [5,3,2],
    verbose_dict = None
):
    """
    A reference implementation of speculative decoding generation.
    Returns at least the specified number of tokens - the speculator may return a 
    few extra in the final step. 
    Currently does not support batched input, and reproduces behavior of greedy decoding only.

    Args:
        model: A function or nn.Module that takes a batch of input_ids and
            returns logits
        input_ids: A 1xn tensor of token IDs
        speculator: A function or nn.Module that takes a state vector and sampled token
            and returns a set of candidate suffixes
        max_seq_len: the sequence length of the base model
        new_tokens: number of tokens to generate
        top_k: only score the top k candidates from the speculator
        threshes: use top k predictions from each head to generate speculator candidate pool 
        verbose_dict: Optional HF tokenizer vocab dict. If provided, runs verbosely and prints 
            speculator behavior and scoring for each step
    """

    verbose = False
    if verbose_dict is not None:
        verbose = True
        vinv = {v:k for k,v in verbose_dict.items()}
        
    def decode_obo(x, vinv):
        return [vinv[z] for z in x.squeeze().tolist()]
    
    batched = False
    if type(input_ids) == torch.Tensor:
        if input_ids.dim() != 1:
            batched = True
    else:
        raise RuntimeError("generate() requires a tensor of token ids as the prefix")

    if not batched:
        input_ids = input_ids.unsqueeze(0)

    result = input_ids
    next_input = input_ids
    kwargs = dict()
    kwargs["past_key_value_states"] = None
    kwargs["use_cache"] = True

    # Build kv cache and get initial state vector
    output = model(input_ids[:,:-1], include_embeds=True, **kwargs)
    _, past_key_value_states, embeds = output
    embeds = embeds[:,-1:]
    kwargs["past_key_value_states"] = past_key_value_states
    next_input = next_input[:,-1:]
    
    n_gen = 0
    n_steps = 0
    n_kv_s = past_key_value_states
    while n_gen < new_tokens:
        n_steps += 1
        input_ids = next_input[:, -max_seq_len:]
        
        # Get candidate set of speculations
        adds = speculator.generate_tree(embeds, input_ids, threshes, top_k)
        
        n_adds = speculator.nheads
        adds = adds[0] # For now, non-batching and take only first sequence
        input_ids = torch.cat([input_ids.expand(top_k,1), adds], dim=-1) 

        # Build custom attention mask
        mask = torch.ones(input_ids.size(1),input_ids.size(1)+n_kv_s[0][0].size(2), device=input_ids.device)
        mask = mask.tril(diagonal=mask.size(1)-mask.size(0))
        mask = mask.unsqueeze(0).unsqueeze(0).log()
        
        # Base model forward pass
        output = model.forward(input_ids, include_embeds=True, mask=mask, **kwargs)
        logits, past_key_value_states, embeds = output
        logits = logits[:, -n_adds-1:, :]
        next_vals = torch.argmax(logits, dim=-1)
        
        # Check correctness of speculator predictions
        test = input_ids.roll(-1, 1).eq(next_vals).cumprod(1)
        n_correct = test.sum(1).clamp(0,n_adds) # clamp in case pred[0]==targ[-1]
        best_guess = n_correct.argmax()
        
        # Set global values to those of best guess
        next_vals = next_vals[best_guess].unsqueeze(0)
        n_correct = n_correct[best_guess]
        embeds = embeds[best_guess].unsqueeze(0)
        
        if verbose:
            print("Speculation:", decode_obo(input_ids[best_guess], vinv), "n_correct:", n_correct.item())
        
        # Toss any wrong speculator tokens
        next_vals = next_vals[:,:n_correct+1]
        n_gen += n_correct+1
        embeds = embeds[:,n_correct].unsqueeze(1)
            
        n_wrong = n_adds - n_correct
        # kv updates are required for torch.compile with
        # mode='reduce-overhead'
        n_kv_s = []
        for layer_idx in range(len(past_key_value_states)):
            n_kv_s.append([])
            for tensor_idx in range(2):
                base = past_key_value_states[layer_idx][tensor_idx]
                new = past_key_value_states[layer_idx][tensor_idx+2][best_guess].unsqueeze(0)
                if n_wrong > 0:
                    new = new[:,:,:-n_wrong]
                base = torch.cat([base, new], dim=2)
                n_kv_s[layer_idx].append(
                    base.clone(memory_format=torch.contiguous_format).detach()
                )
                # torch._dynamo.mark_dynamic(n_kv_s[layer_idx][tensor_idx], 2)
        kwargs["past_key_value_states"] = n_kv_s

        result = torch.cat((result, next_vals), dim=-1)
        next_input = next_vals[:,-1].unsqueeze(-1)

        if verbose:
            print("Updated output:", decode_obo(result, vinv))
            print()
        
    if not batched:
        result = result[0]
    return result, n_steps