from typing import List, Optional

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


class ClassificationHead(nn.Module):
    """
    A general purpose Classification Head. When applied on the output of a
    Headless model, will project from the embedding space to a space equal to
    the number of classes provided.
    """

    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        activation_fn: nn.Module,
        layer_norm: Optional[nn.Module] = None,
        dense_bias: bool = True,
        head_bias: bool = True,
        dropout: float = 0.0,
        apply_pooling_fn: bool = False,
    ):
        """
        Initialize a ClassificationHead

        Parameters
        ----------
        emb_dim: int
            the embedding dimension
        num_classes: int
            the output number of classes
        activation_fn: nn.Module
            the activation function to use prior to apply the dense layer
        layer_norm: nn.Module, optional
            the layer norm to apply prior to running the model head, (default is no layer_norm)
        dense_bias: bool
            the bias param in the dense layer (default is True)
        head_bias: bool
            the bias param in the head layer (default is True)
        dropout: float
            the dropout to use directly after activation (default is 0.0)
        apply_pooling_fn: bool
            if True, will take the first token for each sequence in the batch as input to the dense layer. Otherwise,
            use the entire sequence as input to the dense layer
        """
        super().__init__()
        self.dense = nn.Linear(emb_dim, emb_dim, bias=dense_bias)
        self.act = activation_fn
        self.dropout = nn.Dropout(dropout)
        self.ln = layer_norm
        self.head = nn.Linear(emb_dim, num_classes, bias=head_bias)
        self.apply_pooling_fn = apply_pooling_fn

    def forward(self, x: torch.Tensor):
        """Run the forward method of a classification head

        Parameters
        ----------
        x: torch.Tensor
            typically the output from a headless model

        Returns
        -------
        torch.Tensor
            a tensor projected to a space given by num_classes
        """
        if self.apply_pooling_fn:
            x = x[:, 0]
        x = self.dense(x)
        x = self.act(x)
        x = self.dropout(x)
        if self.ln is not None:
            x = self.ln(x)
        x = self.head(x)
        return x


class LMHead(nn.Linear):
    # To differentiate for TP
    def forward(self, x):
        return super().forward(x)


class TPLMHead(LMHead, TPModule):
    """
    Output embedding layer for sequence models.

    Args
    ----
    Check nn.Linear for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
    """

    def __init__(
        self,
        vocab_size,
        emb_dim,
        bias=False,
        device=None,
        dtype=None,
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()
        rank, world_size = distributed.rank_and_world(group)
        assert (
            vocab_size % world_size == 0
        ), "The number of tokens must be divisible by world size"
        LMHead.__init__(
            self,
            emb_dim,
            vocab_size // world_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.setup_tp(rank, world_size)

    @staticmethod
    def import_module(head: LMHead, group: ProcessGroup) -> "TPLMHead":
        tp_lmh = TPLMHead(
            vocab_size=head.out_features,
            emb_dim=head.in_features,
            bias=head.bias,
            device=head.weight.device,
            dtype=head.weight.dtype,
            group=group,
        )
        return tp_lmh

    def colwise_param_names(self) -> List[str]:
        return ["self"]

    def forward(self, inp):
        # vocab_idx: b n d if reverse, else b n
        inp_par = copy_to_tensor_model_parallel_region(inp)
        out_par = LMHead.forward(self, inp_par)
        # with ints this wasn't `torch.compile`ing
        rank = torch.tensor(self.rank)
        world_size = torch.tensor(self.world_size)
        return all_gather_from_tensor_model_parallel_region(out_par, rank, world_size)
