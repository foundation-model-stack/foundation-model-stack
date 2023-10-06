from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.distributed_c10d import ProcessGroup

from fms.distributed.tensorparallel import (
    all_gather_from_tensor_model_parallel_region,
    apply_colwise_tp,
    apply_embedding_tp,
    copy_to_tensor_model_parallel_region,
)


class AbsolutePositionEmbedding(nn.Module):
    """Special form of embedding that includes a position encoding"""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        padding_idx: Optional[int] = None,
        max_pos: int = 512,
    ):
        """
        Initialize an AbsolutePositionEmbedding

        Parameters
        ----------
        vocab_size: int
            the length of the vocabulary
        emb_dim: int
            dimensionality of latent space
        padding_idx: int, optional
            Padding token index in the vocabulary. Typically, the token for
            which positions will be zeroed out. If negative or None, no positions
            will be zeroed. (default is None)
        max_pos: int
            the maximum possible sequence length (default is 512)
        """
        super().__init__()
        self.max_pos = max_pos
        self.emb_dim = emb_dim
        self.padding_idx = padding_idx

        if self.padding_idx is None:
            self.emb = nn.Embedding(vocab_size, self.emb_dim)
        else:
            self.emb = nn.Embedding(
                vocab_size, self.emb_dim, padding_idx=self.padding_idx
            )

        self.pos_emb = nn.Embedding(max_pos, self.emb_dim)
        self.register_buffer("pos_id", torch.arange(max_pos).unsqueeze(0))

    def reset_params(self):
        # Defaults to norm-preserving in reverse op, unit vector in forward op
        layers = ["emb"]
        layers.append("pos_emb")
        for layer in layers:
            nn.init.trunc_normal_(
                getattr(self, layer).weight, mean=0.0, std=self.emb_dim**-0.5
            )
        # Preserve pad index dummy-hood
        if self.padding_idx is not None:
            self.emb.weight.data[self.padding_idx].zero_()

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        correct_pads: bool = False,
    ):
        """
        perform a forward pass of absolute positional embedding

        Parameters
        ----------
        x: torch.Tensor
            the input tensor
        position_ids: torch.LongTensor, optional
            a tensor which signifies the positions in the position embedding. If
            None, will simply be a range from beginning to end of sequence for
            each sequence in the batch (default is None)
        correct_pads: bool
            if True, will assume position_ids has not been corrected for padding
            and will perform the necessary shifting, otherwise assume
            position_ids has already been corrected (default is False)

        Returns
        -------
        torch.Tensor
            the output of absolute positional embeddings
        """
        x_emb = self.emb(x)

        if position_ids is None:
            # get the position ids from the shape
            _position_ids = self.pos_id[:, : x.size(1)]
        else:
            # use the position ids provided by the user directly
            _position_ids = position_ids

        # if padding_idx exists we want to zero out the associated positions
        if self.padding_idx is not None:
            is_pad = x == self.padding_idx
            # if correct_pads is true, rewind count for every pad token
            if correct_pads:
                _position_ids = _position_ids.sub(is_pad.cumsum(1))
                # In case of left-padding, prevent negative indices (get zeroed anyway)
                _position_ids = _position_ids.clamp(min=0)
            # zero out the associated position embeddings
            position_out = self.pos_emb(_position_ids).mul(~is_pad.unsqueeze(-1))
        else:
            # otherwise just look up the position embeddings
            position_out = self.pos_emb(_position_ids)

        return x_emb + position_out


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
        self.reversible = reversible
        self.debug = debug
        self.tie_weights = tie_weights
        self.bias = bias
        assert (
            reversible or not tie_weights
        ), "Error: weights cannot be tied when there is no output head!"
        if padding_idx is None:
            self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        else:
            self.emb = nn.Embedding(
                self.vocab_size, self.emb_dim, padding_idx=self.padding_idx
            )
        if reversible:
            self.head = nn.Linear(self.emb_dim, self.vocab_size, bias=bias)
            if tie_weights:
                self.head.weight = self.emb.weight
        self.reset_params()

    def reset_params(self):
        # Defaults to norm-preserving in reverse op, unit vector in forward op
        layers = ["emb"]
        if self.reversible and not self.tie_weights:
            layers.append("head")
        for layer in layers:
            nn.init.trunc_normal_(
                getattr(self, layer).weight, mean=0.0, std=self.emb_dim**-0.5
            )
        if self.reversible and self.bias:
            self.head.bias.data.zero_()
        # Preserve pad index dummy-hood
        if self.padding_idx is not None:
            self.emb.weight.data[self.padding_idx].zero_()

    def forward(self, inp, reverse=False):
        # If reverse is False, compute input embeddings. If reverse is True, compute output logits.
        # vocab_idx: b n d if reverse, else b n
        if not reverse:
            if self.debug:
                assert (
                    inp.min().item() >= 0
                ), f"Error: you have requested a negative vocab index: {inp.min().item()}"
                assert (
                    inp.max().item() < self.vocab_size
                ), f"Error: you have requested an out of vocab index: {inp.max().item()}"
            return self.emb(inp)
        else:
            if self.debug:
                assert (
                    self.reversible
                ), "Error: cannot make prediction when there is no output head!"
            return self.head(inp)


class TPWordEmbedding(WordEmbedding):
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
        reversible=True,
        tie_weights=True,
        bias=False,
        debug=False,
        group: ProcessGroup = None,
    ):
        assert torch.distributed.is_initialized()
        if group is None:
            group = torch.distributed.GroupMember.WORLD
        world_size = group.size()
        rank = group.rank()
        assert (
            emb_dim % world_size == 0
        ), "The embedding dimensions must be divisible by world size"
        assert (
            vocab_size % world_size == 0
        ), "The number of tokens must be divisible by world size"
        super(WordEmbedding, self).__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        if padding_idx is not None:
            padding_idx = (
                padding_idx if (padding_idx >= 0 and padding_idx < vocab_size) else None
            )
        self.padding_idx = padding_idx
        self.reversible = reversible
        self.debug = debug
        self.tie_weights = tie_weights
        self.bias = bias

        assert (
            reversible or not tie_weights
        ), "Error: weights cannot be tied when there is no output head!"
        if padding_idx is None:
            self.emb = nn.Embedding(self.vocab_size, self.emb_dim // world_size)
        else:
            self.emb = nn.Embedding(
                self.vocab_size,
                self.emb_dim // world_size,
                padding_idx=self.padding_idx,
            )
        if reversible:
            assert (
                self.vocab_size % world_size == 0
            ), "The vocab size should be a multiple of the world size!"
            self.head = nn.Linear(
                self.emb_dim, self.vocab_size // world_size, bias=bias
            )
            if tie_weights:
                self.head.weight = self.emb.weight

        self.rank = rank
        self.world_size = world_size

    @staticmethod
    def import_module(we: WordEmbedding, group: ProcessGroup) -> "TPWordEmbedding":
        tp_we = TPWordEmbedding(
            vocab_size=we.vocab_size,
            emb_dim=we.emb_dim,
            padding_idx=we.padding_idx,
            reversible=we.reversible,
            tie_weights=we.tie_weights,
            bias=we.bias,
            debug=we.debug,
            group=group,
        )
        return tp_we

    def import_weights(self, we: WordEmbedding):
        apply_embedding_tp(self.emb, we.emb, self.world_size, self.rank)
        if self.abs_pos:
            apply_embedding_tp(self.pos_emb, we.pos_emb, self.world_size, self.rank)
        if self.reversible and not self.tie_weights:
            apply_colwise_tp(self.head, we.head, self.world_size, self.rank)

    def forward(self, inp, reverse=False):
        # If reverse is False, compute input embeddings. If reverse is True, compute output logits.
        # vocab_idx: b n d if reverse, else b n
        inp_par = copy_to_tensor_model_parallel_region(inp)
        out_par = WordEmbedding.forward(self, inp_par, reverse=reverse)
        # with ints this wasn't `torch.compile`ing
        rank = torch.tensor(self.rank)
        world_size = torch.tensor(self.world_size)
        return all_gather_from_tensor_model_parallel_region(out_par, rank, world_size)
