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
