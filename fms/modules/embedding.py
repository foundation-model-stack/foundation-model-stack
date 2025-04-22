from typing import Dict, Optional, Set

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed.distributed_c10d import ProcessGroup

from fms import distributed
from fms.distributed.tensorparallel import (
    all_gather_from_tensor_model_parallel_region,
    copy_to_tensor_model_parallel_region,
)
from fms.modules.tp import TPModule


class WordEmbedding(nn.Module):
    """
    Input/output embedding layer for sequence models.
    Includes vocabulary and optional absolute positional encodings.
    Can optionally include output embeddings, to provide "reversed" output prediction logits.
    ...
    Args
    ----
    vocab_size : int
        Length of vocabulary
    emb_dim : int
        Dimensionality of latent space
    padding_idx : int|None
        Padding token index in the vocabulary. Sets embedding for this token to zero since it is functionally inert.
    max_pos : int
        Maximum sequence length the model can handle. Sequences of shorter length are allowed and handled gracefully.
    abs_pos : bool
        Include absolute positional encodings?
    reversible : bool
        Include support for output logit prediction?
    tie_weights : bool
        If reversible: share input and output embeddings, or learn them separately?
    """

    def __init__(
        self,
        vocab_size,
        emb_dim,
        padding_idx=None,
        max_pos=512,
        abs_pos=False,
        reversible=True,
        tie_weights=True,
        bias=False,
        debug=False,
    ):
        super(WordEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        if padding_idx is not None:
            padding_idx = (
                padding_idx if padding_idx >= 0 and padding_idx < vocab_size else None
            )
        self.padding_idx = padding_idx
        self.abs_pos = abs_pos
        self.reversible = reversible
        self.debug = debug
        self.tie_weights = tie_weights
        self.bias = bias
        self.max_pos = max_pos
        assert reversible or not tie_weights, (
            "Error: weights cannot be tied when there is no output head!"
        )
        if padding_idx is None:
            self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        else:
            self.emb = nn.Embedding(
                self.vocab_size, self.emb_dim, padding_idx=self.padding_idx
            )
        if abs_pos:
            self.pos_emb = nn.Embedding(max_pos, self.emb_dim)
            self.register_buffer("pos_id", torch.arange(max_pos).unsqueeze(0))
        if reversible:
            self.head = nn.Linear(self.emb_dim, self.vocab_size, bias=bias)
            if tie_weights:
                self.head.weight = self.emb.weight

    def reset_parameters(self):
        # Defaults to norm-preserving in reverse op, unit vector in forward op
        layers = ["emb"]
        if self.abs_pos:
            layers.append("pos_emb")
        if self.reversible and not self.tie_weights:
            layers.append("head")
        for layer in layers:
            nn.init.trunc_normal_(getattr(self, layer).weight, mean=0.0, std=0.02)
        if self.reversible and self.bias:
            self.head.bias.data.zero_()
        # Preserve pad index dummy-hood
        if self.padding_idx is not None:
            self.emb.weight.data[self.padding_idx].zero_()

    def to_tp(self, group: ProcessGroup) -> "TPWordEmbedding":
        return TPWordEmbedding.import_module(self, group)

    def forward(self, inp, reverse=False):
        # If reverse is False, compute input embeddings. If reverse is True, compute output logits.
        # vocab_idx: b n d if reverse, else b n
        if not reverse:
            if self.debug:
                assert inp.min().item() >= 0, (
                    f"Error: you have requested a negative vocab index: {inp.min().item()}"
                )
                assert inp.max().item() < self.vocab_size, (
                    f"Error: you have requested an out of vocab index: {inp.max().item()}"
                )
            out = self.emb(inp)
            if self.abs_pos:
                pos = self.pos_id[:, : inp.size(1)]
                is_pad = inp == self.padding_idx
                pos = pos.sub(is_pad.cumsum(1))
                pos = pos.clamp(
                    min=0
                )  # In case of left-padding, prevent negative indices (get zeroed anyways)
                out = out.addcmul(self.pos_emb(pos), ~is_pad.unsqueeze(-1))
            return out
        else:
            if self.debug:
                assert self.reversible, (
                    "Error: cannot make prediction when there is no output head!"
                )
            return self.head(inp)


class TPWordEmbedding(WordEmbedding, TPModule):
    """
    Input/output embedding layer for sequence models.
    Includes vocabulary and optional absolute positional encodings.
    Can optionally include output embeddings, to provide "reversed" output prediction logits.
    ...
    Args
    ----
    Check WordEmbedding for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
    """

    def __init__(
        self,
        vocab_size,
        emb_dim,
        padding_idx=None,
        max_pos=512,
        abs_pos=False,
        reversible=True,
        tie_weights=True,
        bias=False,
        debug=False,
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()
        rank, world_size = distributed.rank_and_world(group)
        assert emb_dim % world_size == 0, (
            "The embedding dimensions must be divisible by world size"
        )
        assert vocab_size % world_size == 0, (
            "The number of tokens must be divisible by world size"
        )
        WordEmbedding.__init__(
            self,
            vocab_size,
            emb_dim,
            padding_idx,
            max_pos,
            abs_pos,
            reversible,
            tie_weights,
            bias,
            debug,
        )

        if padding_idx is None:
            self.emb = nn.Embedding(self.vocab_size, self.emb_dim // world_size)
        else:
            self.emb = nn.Embedding(
                self.vocab_size,
                self.emb_dim // world_size,
                padding_idx=self.padding_idx,
            )
        if abs_pos:
            self.pos_emb = nn.Embedding(max_pos, self.emb_dim // world_size)
        if reversible:
            assert self.vocab_size % world_size == 0, (
                "The vocab size should be a multiple of the world size!"
            )
            self.head = nn.Linear(
                self.emb_dim, self.vocab_size // world_size, bias=bias
            )
            if tie_weights:
                self.head.weight = self.emb.weight
        self.setup_tp(rank, group)

    @staticmethod
    def import_module(we: WordEmbedding, group: ProcessGroup) -> "TPWordEmbedding":
        tp_we = TPWordEmbedding(
            vocab_size=we.vocab_size,
            emb_dim=we.emb_dim,
            padding_idx=we.padding_idx,
            max_pos=we.max_pos,
            abs_pos=we.abs_pos,
            reversible=we.reversible,
            tie_weights=we.tie_weights,
            bias=we.bias,
            debug=we.debug,
            group=group,
        )
        return tp_we

    def load_weights(
        self,
        tensor_values: Dict[str, torch.Tensor],
    ):
        # 1. Grab the weights from tensor_values
        used_keys: Set[str] = set()
        weight_count = 1
        emb_weight = self._get_sd_weight(tensor_values, used_keys, ["emb"])
        if self.abs_pos:
            pos_emb_weight = self._get_sd_weight(tensor_values, used_keys, ["pos_emb"])
            weight_count += 1
        if self.reversible and not self.tie_weights:
            head_weight = self._get_sd_weight(
                tensor_values, used_keys, ["head", "weight"]
            )
            weight_count += 1
            if self.bias:
                head_bias = self._get_sd_weight(
                    tensor_values, used_keys, ["head", "bias"]
                )
                weight_count += 1

        # 2. Raise exceptions
        if len(tensor_values) > weight_count:
            unused_keys = set(tensor_values.keys()).difference(used_keys)
            raise AttributeError(f"Unused weight(s): {', '.join(unused_keys)}")

        # 3. Load and shard the weights
        self.sharded_copy(self.emb.weight, emb_weight, 1, [self.world_size])
        if self.abs_pos:
            self.sharded_copy(self.pos_emb.weight, pos_emb_weight, 1, [self.world_size])
        if self.reversible and not self.tie_weights:
            self.sharded_copy(self.head.weight, head_weight, 0, [self.world_size])
            if self.bias:
                self.sharded_copy(self.head.bias, head_bias, 0, [self.world_size])

    def forward(self, inp, reverse=False):
        # If reverse is False, compute input embeddings. If reverse is True, compute output logits.
        # vocab_idx: b n d if reverse, else b n
        inp_par = copy_to_tensor_model_parallel_region(inp, self.group)
        out_par = WordEmbedding.forward(self, inp_par, reverse=reverse)
        # with ints this wasn't `torch.compile`ing
        return all_gather_from_tensor_model_parallel_region(
            out_par, self.rank, self.group
        )


class TPEmbedding(nn.Embedding, TPModule):
    """
    Input embedding layer for sequence models. Not to be confused with TPWordEmbedding.
    (TP)WordEmbedding supports fusing together the LM Head and the input Embedding, while
    this is a class for when you want them separate, like in headless models.

    Args
    ----
    Check nn.Embedding for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        group: Optional[ProcessGroup] = None,
        **kwargs,
    ):
        assert torch.distributed.is_initialized()
        rank, world_size = distributed.rank_and_world(group)
        assert embedding_dim % world_size == 0, (
            "The embedding dimensions must be divisible by world size"
        )
        nn.Embedding.__init__(
            self, num_embeddings, embedding_dim // world_size, **kwargs
        )
        self.setup_tp(rank, group)

    @staticmethod
    def import_module(e: nn.Embedding, group: ProcessGroup) -> "TPEmbedding":
        tp_e = TPEmbedding(
            num_embeddings=e.num_embeddings,
            embedding_dim=e.embedding_dim,
            padding_idx=e.padding_idx,
            max_norm=e.max_norm,
            norm_type=e.norm_type,
            scale_grad_by_freq=e.scale_grad_by_freq,
            sparse=e.sparse,
            _freeze=not e.weight.requires_grad,
            device=e.weight.device,
            dtype=e.weight.dtype,
            group=group,
        )
        return tp_e

    def load_weights(
        self,
        tensor_values: Dict[str, torch.Tensor],
    ):
        # 1. Grab the weights from tensor_values
        used_keys: Set[str] = set()
        emb_weight = self._get_sd_weight(tensor_values, used_keys, ["weight"])

        # 2. Raise exceptions
        if len(tensor_values) > 1:
            unused_keys = set(tensor_values.keys()).difference(used_keys)
            raise AttributeError(f"Unused weight(s): {', '.join(unused_keys)}")

        # 3. Load and shard the weights
        self.sharded_copy(self.weight, emb_weight, 1, [self.world_size])

    def forward(self, inp: torch.Tensor):
        # vocab_idx: b n d if reverse, else b n
        inp_par = copy_to_tensor_model_parallel_region(inp, self.group)
        out_par = nn.Embedding.forward(self, inp_par)
        # with ints this wasn't `torch.compile`ing
        return all_gather_from_tensor_model_parallel_region(
            out_par, self.rank, self.group
        )
