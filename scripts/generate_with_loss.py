import torch
from typing import Union, Callable, Tuple, Optional, List
import os
from torch.nn.modules.loss import _Loss

from fms.utils.generation import _make_cache_contiguous

from fms.models import get_model
from fms.utils import tokenizers, generation


def ids_for_prompt(prompt):
    tokens = tokenizer.tokenize(prompt)
    tokens = ["<s>"] + tokens
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    return ids

def ids_for_labels(labels):
    tokens = tokenizer.tokenize(labels)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    return ids

def pad_inputs(input_ids: torch.Tensor, max_length: int, pad_id: int = 0, left_padding: bool = True) -> torch.Tensor:
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
    if left_padding:
        return torch.cat((pads_tensor, input_ids))
    else:
        return torch.cat((input_ids, pads_tensor))

def generate(
    model: Union[Callable, torch.nn.Module],
    input_ids_batch: List[torch.Tensor],
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
        input_ids_batch: torch.Tensor
            the list of sequence input ids to the model
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

    padded_batch = []
    for input_ids in input_ids_batch:
        padded = pad_inputs(input_ids, pad_to_max_length)
        padded_batch.append(padded)

    input_ids = torch.stack(padded_batch, dim=0)
    result = input_ids
    next_input = input_ids
    past_key_values = None
    losses = []
    correct_outputs = [-1 for _ in range(len(padded_batch))]

    for i in range(max_new_tokens):
        next_ground_truth = torch.tensor([label[i] for label in labels], device=input_ids.device)
        input_ids = next_input[:, -pad_to_max_length:] # slice off one of the pads (assume the prompt + max_new_tokens is always less than pad_to_max_length

        # create mask
        if i == 0 or not use_cache:
            is_pad = input_ids == 0
            mask = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)
            mask = mask.tril(diagonal=0)
        else:
            is_not_pad = result != 0
            mask = is_not_pad.unsqueeze(-2)

        # model forward
        output = model(input_ids, mask=mask, use_cache=use_cache, past_key_value_states=past_key_values)

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
        loss_list = []
        for j, next_ground_truth_j in enumerate(next_ground_truth):
            loss_list.append(loss_fn(logits[j, :], next_ground_truth_j).item())
        losses.append(loss_list)
        next_val = torch.argmax(logits, dim=-1).unsqueeze(0).t()
        result = torch.cat((result, next_val), dim=-1)

        # just do with simple loop as we are not testing performance here
        for j, next_val_j in enumerate(next_val.tolist()):
            if correct_outputs[j] == -1 and next_ground_truth[j].item() != next_val_j[0]:
                correct_outputs[j] = i

        if use_cache:
            next_input = next_val
        else:
            next_input = result

    return result, losses, correct_outputs

def print_result(result):
    if local_rank != 0:
        return
    result = generation.truncate_after_eos(
        result, tokenizer.convert_tokens_to_ids("</s>")
    )
    # print(result)
    # print(tokenizer.convert_ids_to_tokens(result))
    print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result)))
    print()

def infer(model, input_ids, labels, use_cache, max_input_length, max_new_tokens):
    # With greedy generation (do_sample=False) we _should_ always get the same results.
    # There is currently a bug in start_pos for batched rotary embeddings that can lead
    # varying results for the same prompt.
    if local_rank == 0:
        print("use_cache", use_cache)
        print("==================")

    result, losses, correct_tokens = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        labels=labels,
        pad_to_max_length=max_input_length
    )
    for i in range(result.shape[0]):
        print_result(result[i])
        print(f"correct tokens: {correct_tokens[i]}")
        print(f"loss: {[l[i] for l in losses]}")


local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
device = torch.device("cuda", local_rank)

torch.set_default_device(device)
torch.set_default_dtype(torch.half)

model = get_model(
    "llama",
    "7b",
    model_path="/lustre/llama_weights/7B-F",
    device_type="cuda",
    source="meta",
    norm_eps=1e-6
)
tokenizer = tokenizers.get_tokenizer("/lustre/llama_weights/tokenizer.model")
model.eval()
torch.set_grad_enabled(False)

template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

prompt = template.format("Explain some popular greetings in Spanish.")
prompt2 = template.format("Provide a list of instructions for preparing chicken soup.")

input_ids = [ids_for_prompt(prompt), ids_for_prompt(prompt2)]
label1 = """\nSome popular greetings in Spanish include "Hola" (hello), "Buenos días" (good morning), "Buenas tardes" (good afternoon), and "Buenas noches" (good evening). Additionally, it is common to use the phrase "¿Cómo estás?" (how are you?) when greeting someone, to which the response might be "Estoy bien" (I'm fine) or "Estoy mal" ("""
label2 = """\nSure! Here are the steps to prepare chicken soup:

1. Start by chopping 1 onion and 3 cloves of garlic.
2. In a large pot, heat 2 tablespoons of olive oil over medium heat.
3. Add the chopped onion and sauté until it's translucent.
4. Add the chopped garlic and sauté for an additional 2 minutes.
5"""
max_input_length = 2048
num_tokens_to_test = 100
label1_ids = pad_inputs(ids_for_labels(label1)[1:], num_tokens_to_test, -1, False)
label2_ids = pad_inputs(ids_for_labels(label2)[1:], num_tokens_to_test, -1, False)
label_ids = [label1_ids, label2_ids]

infer(model, input_ids, label_ids, True, max_input_length, num_tokens_to_test)