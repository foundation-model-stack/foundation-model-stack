import time
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.export import ExportedProgram, export, load, save


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
    padding_kwargs["mask"] = mask

    position_ids = torch.stack(position_ids_list)
    padding_kwargs["position_ids"] = position_ids

    return input_ids, padding_kwargs


def __update_padding_kwargs(
    use_cache: bool, model_specific_kwargs: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Generic function to prepare any model specific keyword arguments"""
    # extend the attention mask
    mask = model_specific_kwargs.get("mask", None)
    if mask is not None:
        # get the last row of the 3d mask
        mask = mask[:, -1:, :]
        # extend the mask one slot
        mask = torch.cat(
            (
                mask,
                torch.ones(mask.size(0), 1, 1, dtype=torch.bool, device=mask.device),
            ),
            dim=2,
        )
        model_specific_kwargs["mask"] = mask

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


class ExportedModule(torch.nn.Module):

    def __init__(
        self, module: torch.nn.Module, dynamic_shapes: bool = False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.exported_file_map: Dict[str, str] = {}
        self.exported_program_map: Dict[str, ExportedProgram] = {}
        self._base_module = module
        self.config = (
            self._base_module.config if hasattr(self._base_module, "config") else None
        )
        # todo: we will need this in order to only save a single version of the state dict, but for now it's not
        #  possible as the state_dict is a property of the ExportedProgram
        # self._key_with_state_dict = None

        # todo: dynamic shapes
        #  This will involve fixing certain dimensions of the key that are marked as dynamic
        #  also inferring the dynamic shapes input for export based on some attributes of the _base_module

    def __getstate__(self):
        return {
            "exported_file_map": self.exported_file_map,
            "key_with_state_dict": self._key_with_state_dict,
        }

    def __setstate__(self, d):
        self._base_module = None
        self.exported_file_map = d["exported_file_map"]
        # todo: we will need this in order to only save a single version of the state dict, but for now it's not
        #  possible as the state_dict is a property of the ExportedProgram
        # self._key_with_state_dict = d["key_with_state_dict"]

        for k, v in self.exported_file_map.items():
            self.exported_program_map[k] = load(v)

    def __append_tensor_shape_or_constant(self, key, param_key, param_value):
        if isinstance(param_value, torch.Tensor):
            key[param_key] = tuple(param_value.shape)
        elif isinstance(param_value, (bool, int, float)):
            key[param_key] = param_value
        elif isinstance(param_value, (list, tuple)):
            for v in param_value:
                self.__append_tensor_shape_or_constant(key, param_key, v)
        elif isinstance(param_value, dict):
            for k, v in param_value.items():
                self.__append_tensor_shape_or_constant(key, f"{param_key}_{k}", v)

    def _get_key(self, *args, **kwargs):
        key = {}
        for i, arg in enumerate(args):
            self.__append_tensor_shape_or_constant(key, f"arg{i}", arg)

        for kwarg_key, kwarg_value in kwargs.items():
            self.__append_tensor_shape_or_constant(key, kwarg_key, kwarg_value)
        return str(key)

    def _update(self, key, exported_program):
        # todo: we will need this in order to only save a single version of the state dict, but for now it's not
        #  possible as the state_dict is a property of the ExportedProgram
        # if len(self.exported_file_map) != 0:
        #     exported_program.state_dict = {}
        # else:
        #     self._key_with_state_dict = key

        # todo: this cannot be saved yet as per https://github.com/pytorch/pytorch/issues/130152
        # save(exported_program, f"{key}.pt2")
        self.exported_file_map[key] = f"{key}.pt2"
        self.exported_program_map[key] = exported_program

    def forward(self, *args, **kwargs):
        key = self._get_key(*args, **kwargs)
        if key not in self.exported_program_map:
            if self._base_module is not None:
                exported_program = export(self._base_module, args=args, kwargs=kwargs)
                self._update(key, exported_program)
                return self._base_module(*args, **kwargs)
            else:
                raise AttributeError(
                    "this ExportedModuleMap has not seen the given inputs, and is not backed by a module"
                )

        exported_program = self.exported_program_map[key]
        # todo: we will need this in order to only save a single version of the state dict, but for now it's not
        #  possible as the state_dict is a property of the ExportedProgram
        # exported_program.state_dict = self.exported_program_map[self._key_with_state_dict].state_dict
        return exported_program.module()(*args, **kwargs)


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
        eos_token_id: the optional token id representing the end of sequence
        timing: whether to measure timings: "per-token" for each token generation time,
            "e2e" for full generation loop. Both options make `generate` return a tuple
            with the following information:
            - "per-token": Array with `max_new_tokens` time measurements (in s)
            - "e2e": Array with a single e2e generation loop time measurement (in s)
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
    if timing != "":
        times: List[float] = []
        start_time = time.time()
    for i in range(max_new_tokens):
        input_ids = next_input[:, -max_seq_len:]

        # prepare any padding keyword arguments
        # iteration 0 is the prefill step (cache has not been filled yet), so no need to extend the mask/position_ids
        if i > 0:
            kwargs = __update_padding_kwargs(use_cache, kwargs)

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
