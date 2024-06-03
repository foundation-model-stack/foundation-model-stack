from typing import Dict, Optional, Set

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.distributed_c10d import ProcessGroup

import fms.triton.pytorch_ops as triton_ops  # registers the PT custom ops
from fms import distributed
from fms.distributed.tensorparallel import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from fms.modules.tp import TPModule


class FeedForwardBlock(nn.Module):
    """
    A two-layer, symmetric, fully-connected MLP structure.
    ...
    Args
    ----
    emb_dim : int
        Dimensionality of input and output vectors.
    hidden_grow_factor : float
        Sets dimensionality of inner latent space (emb_dim * hidden_grow_factor)
    multiple_of : Optional[int]
        Ensure inner latent space is a multiple of this parameter if defined (useful for
        TensorParallel as well as GPU kernel speed)
    activation_fn : nn.Module
        An activation function over torch.FloatTensors applied to inner latent space.
    p_dropout : float|None
        Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
    use_bias : bool
        Include bias terms in fully-connected sublayers?
    """

    def __init__(
        self,
        emb_dim,
        hidden_grow_factor=4.0,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=True,
    ):
        super(FeedForwardBlock, self).__init__()
        self.hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            self.hidden_dim = multiple_of * (
                (self.hidden_dim + multiple_of - 1) // multiple_of
            )
        self.w1 = nn.Linear(emb_dim, self.hidden_dim, bias=use_bias)
        self.a = activation_fn
        self.p_dropout = p_dropout
        if p_dropout:
            self.d = nn.Dropout(p_dropout)
        self.w2 = nn.Linear(self.hidden_dim, emb_dim, bias=use_bias)
        self.use_bias = use_bias

    def reset_parameters(self):
        for layer in ["w1", "w2"]:
            nn.init.trunc_normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=0.02,
            )
            if self.use_bias:
                getattr(self, layer).bias.data.zero_()

    def to_tp(self, group: ProcessGroup) -> "TPFeedForwardBlock":
        return TPFeedForwardBlock.import_module(self, group)

    def forward(self, x):
        out = self.a(self.w1(x))
        if self.p_dropout:
            out = self.d(out)
        out = self.w2(out)
        return out


class TPFeedForwardBlock(FeedForwardBlock, TPModule):
    """
    A two-layer, symmetric, fully-connected MLP structure with Tensor Parallel support.

    Args
    ----
    Check FeedForwardBlock for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
    """

    def __init__(
        self,
        emb_dim,
        hidden_grow_factor: float = 4,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=True,
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()
        hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        rank, world_size = distributed.rank_and_world(group)
        assert (
            hidden_dim % world_size == 0
        ), "Hidden dim must be divisible by world size"
        FeedForwardBlock.__init__(
            self,
            emb_dim,
            hidden_grow_factor / world_size,
            multiple_of,
            activation_fn,
            p_dropout,
            use_bias,
        )
        self.setup_tp(rank, world_size)

    def load_weights(
        self,
        tensor_values: Dict[str, torch.Tensor],
    ):
        # 1. Grab the weights from tensor_values
        used_keys: Set[str] = set()
        w1_weight = self._get_sd_weight(tensor_values, used_keys, ["w1", "weight"])
        w2_weight = self._get_sd_weight(tensor_values, used_keys, ["w2", "weight"])
        if self.use_bias:
            w1_bias = self._get_sd_weight(tensor_values, used_keys, ["w1", "bias"])
            w2_bias = self._get_sd_weight(tensor_values, used_keys, ["w2", "bias"])

        # 2. Raise exceptions for extra weights in tensor_values
        if len(tensor_values) > (4 if self.use_bias else 2):
            unused_keys = set(tensor_values.keys()).difference(used_keys)
            raise AttributeError(f"Unused weight(s): {', '.join(unused_keys)}")

        # 3. Load and shard the weights
        self.sharded_copy(self.w1.weight, w1_weight, 0, [self.world_size])
        self.sharded_copy(self.w2.weight, w2_weight, 1, [self.world_size])
        if self.use_bias:
            self.sharded_copy(self.w1.bias, w1_bias, 0, [self.world_size])
            self.sharded_copy(self.w2.bias, w2_bias, 1, [self.world_size], False)

    @staticmethod
    def import_module(
        ffb: FeedForwardBlock, group: ProcessGroup
    ) -> "TPFeedForwardBlock":
        tp_ffb = TPFeedForwardBlock(
            emb_dim=ffb.w1.in_features,
            hidden_grow_factor=ffb.hidden_dim / ffb.w1.in_features,
            multiple_of=None,
            activation_fn=ffb.a,
            p_dropout=ffb.p_dropout,
            use_bias=ffb.use_bias,
            group=group,
        )
        return tp_ffb

    def forward(self, x):
        x_par = copy_to_tensor_model_parallel_region(x)
        out_par = FeedForwardBlock.forward(self, x_par)
        return reduce_from_tensor_model_parallel_region(out_par)


class GatedLinearUnit(nn.Module):
    """
    A two-point-five-layer, fully-connected gated linear MLP structure (GLU).
    Contains 50% extra params compared to FeedForwardBlock, adjust accordingly.
    ...
    Args
    ----
    emb_dim : int
        Dimensionality of input and output vectors.
    hidden_grow_factor : float
        Sets dimensionality of inner latent space (emb_dim * hidden_grow_factor)
    multiple_of : Optional[int]
        Ensure inner latent space is a multiple of this parameter if defined (useful for
        TensorParallel as well as GPU kernel speed)
    activation_fn : nn.Module
        An activation function over torch.FloatTensors applied to inner latent gates.
    p_dropout : float|None
        Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
    use_bias : bool
        Include bias terms in fully-connected sublayers?
    """

    def __init__(
        self,
        emb_dim,
        hidden_grow_factor: float = 4,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=True,
    ):
        super(GatedLinearUnit, self).__init__()
        self.hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            self.hidden_dim = multiple_of * (
                (self.hidden_dim + multiple_of - 1) // multiple_of
            )
        self.wg1_fused = nn.Linear(emb_dim, 2 * self.hidden_dim, bias=use_bias)
        self.a = activation_fn
        self.p_dropout = p_dropout
        if p_dropout:
            self.d = nn.Dropout(p_dropout)
        self.w2 = nn.Linear(self.hidden_dim, emb_dim, bias=use_bias)
        self.use_bias = use_bias
        self.width = emb_dim
        self.grow_factor = hidden_grow_factor

    def reset_parameters(self):
        for layer in ["wg1_fused", "w2"]:
            nn.init.trunc_normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=0.02,
            )
            if self.use_bias:
                getattr(self, layer).bias.data.zero_()

    def to_tp(self, group: ProcessGroup) -> "TPGatedLinearUnit":
        return TPGatedLinearUnit.import_module(self, group)

    def forward(self, x):
        out_fused = self.wg1_fused(x)
        wg, w1 = torch.split(out_fused, [self.hidden_dim, self.hidden_dim], dim=2)
        out = self.a(wg) * w1
        if self.p_dropout:
            out = self.d(out)
        return self.w2(out)


class TPGatedLinearUnit(GatedLinearUnit, TPModule):
    """
    A two-point-five-layer, fully-connected gated linear MLP structure (GLU).
    Contains 50% extra params compared to FeedForwardBlock, adjust accordingly.
    This subclass adds Tensor Parallel support.

    Args
    ----
    Check GatedLinearUnit for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
    """

    def __init__(
        self,
        emb_dim,
        hidden_grow_factor: float = 4,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=True,
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()
        rank, world_size = distributed.rank_and_world(group)

        hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert (
            hidden_dim % world_size == 0
        ), "Hidden dim must be divisible by world size"
        GatedLinearUnit.__init__(
            self,
            emb_dim,
            hidden_grow_factor / world_size,
            multiple_of,
            activation_fn,
            p_dropout,
            use_bias,
        )
        self.setup_tp(rank, world_size)

    def load_weights(
        self,
        tensor_values: Dict[str, torch.Tensor],
    ):
        # 1. Grab the weights from tensor_values
        used_keys: Set[str] = set()
        wg_weight = self._get_sd_weight(
            tensor_values, used_keys, ["wg1_fused", "weight"]
        )
        w2_weight = self._get_sd_weight(tensor_values, used_keys, ["w2", "weight"])
        if self.use_bias:
            wg_bias = self._get_sd_weight(
                tensor_values, used_keys, ["wg1_fused", "bias"]
            )
            w2_bias = self._get_sd_weight(tensor_values, used_keys, ["w2", "bias"])

        # 2. Raise exceptions
        if len(tensor_values) > (4 if self.use_bias else 2):
            unused_keys = set(tensor_values.keys()).difference(used_keys)
            raise AttributeError(f"Unused weight(s): {', '.join(unused_keys)}")

        # 3. Load and shard the weights
        self.sharded_copy(
            self.wg1_fused.weight, wg_weight, 0, [self.world_size, self.world_size]
        )
        self.sharded_copy(self.w2.weight, w2_weight, 1, [self.world_size])
        if self.use_bias:
            self.sharded_copy(
                self.wg1_fused.bias, wg_bias, 0, [self.world_size, self.world_size]
            )
            self.sharded_copy(self.w2.bias, w2_bias, 1, [self.world_size], False)

    @staticmethod
    def import_module(glu: GatedLinearUnit, group: ProcessGroup) -> "TPGatedLinearUnit":
        tp_glu = TPGatedLinearUnit(
            emb_dim=glu.width,
            hidden_grow_factor=glu.hidden_dim / glu.width,
            multiple_of=None,
            activation_fn=glu.a,
            p_dropout=glu.p_dropout,
            use_bias=glu.use_bias,
            group=group,
        )

        return tp_glu

    def forward(self, x):
        x_par = copy_to_tensor_model_parallel_region(x)
        out_par = GatedLinearUnit.forward(self, x_par)
        return reduce_from_tensor_model_parallel_region(out_par)


class ConditionalFeedForward(nn.Module):
    """
    This class represents the expert feed forward networks of an MoE FF layer.

    For more information, see the review paper in https://arxiv.org/pdf/2209.01667.pdf

    Args
    ----
    num_experts : int
        The number of expert feed forward networks.
    dim : int
        The embedding dimension for the transformer model.
    intermediate_size : int
        The intermediate size for the expert networks.
    """

    def __init__(self, num_experts: int, dim: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.intermediate_size = intermediate_size
        self.w13 = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, intermediate_size))

    def reset_parameters(self):
        for param in ["w13", "w2"]:
            nn.init.trunc_normal_(
                getattr(self, param),
                mean=0.0,
                std=0.02,
            )

    def to_tp(self, group: ProcessGroup) -> "TPConditionalFeedForward":
        return TPConditionalFeedForward.import_module(self, group)

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        # Check constraints.
        assert x.shape[1] == self.w13.shape[2], "Hidden size mismatch"
        assert x.is_contiguous(), "Hidden_states must be contiguous"
        assert self.w13.is_contiguous(), "Expert weights 1 must be contiguous"
        assert self.w2.is_contiguous(), "Expert weights 2 must be contiguous"

        M, D = x.shape
        E, N, _ = self.w13.shape
        _, A = expert_indices.shape

        #  if x.device.type == "cuda":
        ## Triton path

        if expert_indices.numel() <= E:
            padding_size = 16
        else:
            padding_size = 64

        (
            padded_token_ids_per_block,
            expert_block_mapping,
            total_padded_tokens,
        ) = triton_ops.moe_align_block_size(expert_indices, padding_size, E)

        x1, x3 = (
            torch.ops.moe.moe_mm(
                x,
                self.w13,
                expert_indices,
                padded_token_ids_per_block,
                expert_block_mapping,
                total_padded_tokens,
                expert_indices.shape[1],
                padding_size,
            )
            .view(-1, N)
            .chunk(2, dim=1)
        )
        return torch.ops.moe.moe_mm(
            F.silu(x1) * x3,
            self.w2,
            expert_indices,
            padded_token_ids_per_block,
            expert_block_mapping,
            total_padded_tokens,
            1,
            padding_size,
        )


class TPConditionalFeedForward(ConditionalFeedForward, TPModule):
    """
    This class represents the expert feed forward networks of an MoE FF layer.
    This subclass adds TP support.

    Args
    ----
    num_experts : int
        The number of expert feed forward networks.
    dim : int
        The embedding dimension for the transformer model.
    intermediate_size : int
        The intermediate size for the expert networks.
    """

    def __init__(
        self,
        num_experts: int,
        dim: int,
        intermediate_size: int,
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()
        rank, world_size = distributed.rank_and_world(group)

        assert (
            intermediate_size % world_size == 0
        ), "Intermediate size must be divisible by world size"
        ConditionalFeedForward.__init__(
            self,
            num_experts,
            dim,
            intermediate_size // world_size,
        )
        self.setup_tp(rank, world_size)

    def load_weights(
        self,
        tensor_values: Dict[str, torch.Tensor],
    ):
        # 1. Grab the weights from tensor_values
        used_keys: Set[str] = set()
        w13_weight = self._get_sd_weight(tensor_values, used_keys, ["w13"])
        w2_weight = self._get_sd_weight(tensor_values, used_keys, ["w2"])

        # 2. Raise exceptions
        if len(tensor_values) > 2:
            unused_keys = set(tensor_values.keys()).difference(used_keys)
            raise AttributeError(f"Unused weight(s): {', '.join(unused_keys)}")

        # 3. Load and shard the weights
        self.sharded_copy(self.w13, w13_weight, 1, [self.world_size, self.world_size])
        self.sharded_copy(self.w2, w2_weight, 2, [self.world_size])

    @staticmethod
    def import_module(
        cff: ConditionalFeedForward, group: ProcessGroup
    ) -> "TPConditionalFeedForward":
        tp_cff = TPConditionalFeedForward(
            num_experts=cff.num_experts,
            dim=cff.dim,
            intermediate_size=cff.intermediate_size,
            group=group,
        )

        return tp_cff

    def forward(self, x, expert_indices):
        x_par = copy_to_tensor_model_parallel_region(x)
        out_par = ConditionalFeedForward.forward(self, x_par, expert_indices)
        return reduce_from_tensor_model_parallel_region(out_par)


class MOEFeedForward(nn.Module):
    """
    A Sparse Mixture Of Experts (MoE) Feed Forward layer. The output of this layer for a
    given input is determined by the weighted sum of the outputs of a subset of size
    `num_activated_experts` of the `num_experts` expert networks. The weights are given
    by the gating network, then passed through a topK and a softmax filter to make it _sparse_.

    For more information, see the review paper in https://arxiv.org/pdf/2209.01667.pdf

    Args
    ----
    num_experts : int
        The number of expert feed forward networks.
    num_activated_experts : int
        How many experts can be activated at any single time.
    dim : int
        The embedding dimension for the transformer model.
    intermediate_size : int
        The intermediate size for the expert networks.
    """

    def __init__(
        self,
        num_experts: int,
        num_activated_experts: int,
        dim: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(num_experts, dim, intermediate_size)
        self.dim = dim
        self.num_activated_experts = num_activated_experts

    def reset_parameters(self):
        nn.init.trunc_normal_(
            self.gate.weight,
            mean=0.0,
            std=0.02,
        )

        self.cond_ffn.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape[:2]
        x = x.view(-1, self.dim)
        # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
        # x: [T, D]
        scores = self.gate(x)  # [T, E]
        expert_weights = F.softmax(scores, dim=-1)
        expert_weights, expert_indices = torch.topk(
            expert_weights, self.num_activated_experts, dim=-1
        )  # [T, A], [T, A]
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)  # [T, A]
        expert_outs = self.cond_ffn(x, expert_indices)
        int_v1 = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)
        int_v2 = int_v1.view(B, S, self.dim)
        return int_v2
