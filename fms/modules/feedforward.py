from typing import List, Optional

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.distributed_c10d import ProcessGroup

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
        gain=1,
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
        self.reset_params(gain=gain)

    def reset_params(self, gain=1):
        # Fulfills following constraints in expectation:
        #  - Norm of w1 and w2 are equal (for step-normalizing optimizers like AdamW / Sophia)
        #  - Norm of output equals norm of input times gamma
        # when activation is relu-like
        for layer in ["w1", "w2"]:
            nn.init.trunc_normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=(2**0.5 * gain / self.w1.weight.numel() ** 0.5) ** 0.5,
            )
            if self.use_bias:
                getattr(self, layer).bias.data.zero_()

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
        gain=1,
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
            gain,
        )
        self.setup_tp(rank, world_size)

    def colwise_param_names(self) -> List[str]:
        return ["w1"]

    def rowwise_param_names(self) -> List[str]:
        return ["w2"]

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
        gain=1,
    ):
        super(GatedLinearUnit, self).__init__()
        self.hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            self.hidden_dim = multiple_of * (
                (self.hidden_dim + multiple_of - 1) // multiple_of
            )
        self.w1 = nn.Linear(emb_dim, self.hidden_dim, bias=use_bias)
        self.wg = nn.Linear(emb_dim, self.hidden_dim, bias=use_bias)
        self.a = activation_fn
        self.p_dropout = p_dropout
        if p_dropout:
            self.d = nn.Dropout(p_dropout)
        self.w2 = nn.Linear(self.hidden_dim, emb_dim, bias=use_bias)
        self.use_bias = use_bias
        self.width = emb_dim
        self.grow_factor = hidden_grow_factor
        self.reset_params(gain=gain)

    def reset_params(self, gain=1):
        # Fulfills following constraints in expectation:
        #  - Norm of w1, wg and w2 are equal (for step-normalizing optimizers like AdamW / Sophia)
        #  - Norm of output equals norm of input times gamma
        # when activation is relu-like and input is standard normal
        for layer in ["w1", "w2", "wg"]:
            nn.init.trunc_normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=(2 * gain**2 / self.grow_factor) ** (1 / 6) / self.width**0.5,
            )
            if self.use_bias:
                getattr(self, layer).bias.data.zero_()

    def forward(self, x):
        out = self.a(self.wg(x)) * self.w1(x)
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
        gain=1,
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
            gain,
        )
        self.setup_tp(rank, world_size)

    def colwise_param_names(self) -> List[str]:
        return ["w1", "wg"]

    def rowwise_param_names(self) -> List[str]:
        return ["w2"]

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
    def __init__(self, num_experts, dim, intermediate_size):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.intermediate_size = intermediate_size
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, intermediate_size, dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, intermediate_size, dim))

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        w1_weights = self.w1[expert_indices].transpose(-1, -2)  # [T, A, D, D]
        w3_weights = self.w3[expert_indices].transpose(-1, -2)  # [T, A, D, D]
        w2_weights = self.w2[expert_indices]  # [T, A, D, D]
        x1 = F.silu(torch.einsum("ti,taio -> tao", x, w1_weights))
        x3 = torch.einsum("ti, taio -> tao", x, w3_weights)
        expert_outs = torch.einsum("tao, taoi -> tai", (x1 * x3), w2_weights)
        return expert_outs


class TPConditionalFeedForward(ConditionalFeedForward, TPModule):
    """
    Args
    ----
    Check ConditionalFeedForward for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
    """

    def __init__(
        self,
        num_experts,
        dim,
        intermediate_size,
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()
        rank, world_size = distributed.rank_and_world(group)

        assert (
            intermediate_size % world_size == 0
        ), "Hidden dim must be divisible by world size"
        ConditionalFeedForward.__init__(
            self,
            num_experts,
            dim,
            intermediate_size // world_size,
        )
        self.setup_tp(rank, world_size)

    def moe_param_names(self) -> List[str]:
        return ["w1", "w2", "w3"]

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
    def __init__(
        self, num_experts, num_activated_experts, dim, intermediate_size
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(num_experts, dim, intermediate_size)
        self.dim = dim
        self.num_activated_experts = num_activated_experts

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
        return torch.einsum("tai,ta -> ti", expert_outs, expert_weights).view(
            B, S, self.dim
        )
