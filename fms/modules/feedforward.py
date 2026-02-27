from typing import Any, Mapping, Optional, Set

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
from fms.modules.linear import (
    LinearModuleShardingInfo,
    get_all_linear_type_to_sharding_maps,
    get_linear,
    get_linear_type,
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
        linear_config: Optional[Mapping[str, Any]] = None,
    ):
        super(FeedForwardBlock, self).__init__()
        self.hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            self.hidden_dim = multiple_of * (
                (self.hidden_dim + multiple_of - 1) // multiple_of
            )
        self.w1 = get_linear(
            emb_dim,
            self.hidden_dim,
            bias=use_bias,
            linear_config=linear_config,
        )
        self.a = activation_fn
        self.p_dropout = p_dropout
        if p_dropout:
            self.d = nn.Dropout(p_dropout)
        self.w2 = get_linear(
            self.hidden_dim,
            emb_dim,
            bias=use_bias,
            linear_config=linear_config,
        )
        self.use_bias = use_bias
        self.linear_config = linear_config

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
        linear_config: Optional[Mapping[str, Any]] = None,
    ):
        assert torch.distributed.is_initialized()
        hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        rank, world_size = distributed.rank_and_world(group)
        assert hidden_dim % world_size == 0, (
            "Hidden dim must be divisible by world size"
        )
        FeedForwardBlock.__init__(
            self,
            emb_dim,
            hidden_grow_factor / world_size,
            multiple_of,
            activation_fn,
            p_dropout,
            use_bias,
            linear_config,
        )
        self.setup_tp(rank, group)

        # linear_type must handle module_name = None to support TP of FNN
        self.linear_type = get_linear_type(self.linear_config)

    def load_weights(
        self,
        tensor_values: dict[str, torch.Tensor],
    ) -> None:
        """Define name of FFN modules to TP-shard, their name-to-module mapping,
        per-module base sharding dimension, and per-module max partition size.
        """

        # sharding modules struct: {'module_name': (module_obj, sharding_dim, max_partition)}
        module_sharding_info = {
            "w1": LinearModuleShardingInfo(self.w1, 0, [self.world_size]),
            "w2": LinearModuleShardingInfo(self.w2, 1, [self.world_size]),
        }

        type_sharding_map = get_all_linear_type_to_sharding_maps()
        unused_keys = type_sharding_map[self.linear_type](
            tensor_values,
            self,
            module_sharding_info,
        )
        return unused_keys

    @staticmethod
    def import_module(
        ffb: FeedForwardBlock, group: ProcessGroup
    ) -> "TPFeedForwardBlock":
        tp_ffb = TPFeedForwardBlock(
            emb_dim=getattr(ffb.w1, "in_features"),
            hidden_grow_factor=ffb.hidden_dim / getattr(ffb.w1, "in_features"),
            multiple_of=None,
            activation_fn=ffb.a,
            p_dropout=ffb.p_dropout,
            use_bias=ffb.use_bias,
            group=group,
            linear_config=ffb.linear_config,
        )
        return tp_ffb

    def forward(self, x):
        x_par = copy_to_tensor_model_parallel_region(x, self.group)
        out_par = FeedForwardBlock.forward(self, x_par)
        return reduce_from_tensor_model_parallel_region(out_par, self.group)


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
        fused: bool = True,
        linear_config: Optional[Mapping[str, Any]] = None,
    ):
        super(GatedLinearUnit, self).__init__()
        self.hidden_dim = int(hidden_grow_factor * emb_dim)
        self.fused = fused
        self.multiple_of = multiple_of
        if multiple_of:
            self.hidden_dim = multiple_of * (
                (self.hidden_dim + multiple_of - 1) // multiple_of
            )
        if self.fused:
            self.wg1_fused = get_linear(
                emb_dim,
                2 * self.hidden_dim,
                bias=use_bias,
                linear_config=linear_config,
            )
        else:
            self.w1 = get_linear(
                emb_dim,
                self.hidden_dim,
                bias=use_bias,
                linear_config=linear_config,
            )
            self.wg = get_linear(
                emb_dim,
                self.hidden_dim,
                bias=use_bias,
                linear_config=linear_config,
            )
        self.a = activation_fn
        self.p_dropout = p_dropout
        if p_dropout:
            self.d = nn.Dropout(p_dropout)
        self.w2 = get_linear(
            self.hidden_dim,
            emb_dim,
            bias=use_bias,
            linear_config=linear_config,
        )
        self.use_bias = use_bias
        self.width = emb_dim
        self.grow_factor = hidden_grow_factor
        self.linear_config = linear_config

    def reset_parameters(self):
        layers = ["w2"]
        if self.fused:
            layers.append("wg1_fused")
        else:
            layers.extend(["w1", "wg"])

        for layer in layers:
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
        if self.fused:
            out_fused = self.wg1_fused(x)
            wg, w1 = torch.split(out_fused, [self.hidden_dim, self.hidden_dim], dim=2)
            out = self.a(wg) * w1
        else:
            out = self.a(self.wg(x)) * self.w1(x)
        if self.p_dropout:
            out = self.d(out)
        return self.w2(out)

    def _initialize_empty_module(self):
        with torch.device("meta"):
            return GatedLinearUnit(
                self.width,
                self.grow_factor,
                self.multiple_of,
                self.a,
                self.p_dropout,
                self.use_bias,
                fused=False,
            )

    def unfuse_weights(self):
        result = self._initialize_empty_module()
        wg, w1 = torch.split(
            self.wg1_fused.weight, [self.hidden_dim, self.hidden_dim], dim=0
        )
        result.wg.weight = torch.nn.Parameter(wg)
        result.w1.weight = torch.nn.Parameter(w1)
        result.w2.weight = torch.nn.Parameter(self.w2.weight)
        if self.use_bias:
            wg_bias, w1_bias = torch.split(
                self.wg1_fused.bias, [self.hidden_dim, self.hidden_dim], dim=0
            )
            result.wg.bias = torch.nn.Parameter(wg_bias)
            result.w1.bias = torch.nn.Parameter(w1_bias)
            result.w2.bias = torch.nn.Parameter(self.w2.bias)
        return result


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
        fused: bool = True,
        linear_config: Optional[Mapping[str, Any]] = None,
    ):
        assert torch.distributed.is_initialized()
        rank, world_size = distributed.rank_and_world(group)

        hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % world_size == 0, (
            "Hidden dim must be divisible by world size"
        )
        GatedLinearUnit.__init__(
            self,
            emb_dim,
            hidden_grow_factor / world_size,
            multiple_of,
            activation_fn,
            p_dropout,
            use_bias,
            fused,
            linear_config,
        )
        self.setup_tp(rank, group)

    def load_weights(
        self,
        tensor_values: dict[str, torch.Tensor],
    ) -> Optional[set]:
        """Define sharding info of GLU module as:
        {'module_name': (module_obj, sharding_dim, max_partition)}
        Then, call the pre-registered sharding function associated with
        self.linear_type.

        `sharding_dim` is sharding dimension of the `weights` parameter
        of nn.Linear. It may differ for other types of linear or other
        parameters.

        The numbers in `max_partition` signify the largest world size
        till we need to duplicate. For instance if we have nheads=16 and
        world_size=32, then first 2 ranks will get first 1/16th of query
        """

        # sharding modules struct: {'module_name': (module_obj, sharding_dim, max_partition)}
        if self.fused:
            module_sharding_info = {
                "wg1_fused": LinearModuleShardingInfo(
                    self.wg1_fused, 0, [self.world_size, self.world_size]
                ),
            }
        else:
            module_sharding_info = {
                "w1": LinearModuleShardingInfo(self.w1, 0, [self.world_size]),
                "wg": LinearModuleShardingInfo(self.wg, 0, [self.world_size]),
            }
        module_sharding_info["w2"] = LinearModuleShardingInfo(
            self.w2, 1, [self.world_size]
        )

        # TODO: Remove assumption that all layers in module share quantization
        module_name = getattr(self.w2, "module_name", None)
        linear_type = get_linear_type(self.linear_config, module_name)
        type_sharding_map = get_all_linear_type_to_sharding_maps()
        unused_keys = type_sharding_map[linear_type](
            tensor_values,
            self,
            module_sharding_info,
        )
        return unused_keys

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
            fused=glu.fused,
            linear_config=glu.linear_config,
        )

        return tp_glu

    def forward(self, x):
        x_par = copy_to_tensor_model_parallel_region(x, self.group)
        out_par = GatedLinearUnit.forward(self, x_par)
        return reduce_from_tensor_model_parallel_region(out_par, self.group)

    def _initialize_empty_module(self):
        return TPGatedLinearUnit(
            self.width,
            self.grow_factor * self.world_size,
            self.multiple_of,
            self.a,
            self.p_dropout,
            self.use_bias,
            fused=False,
        ).to(self.w2.weight.device)


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

        assert intermediate_size % world_size == 0, (
            "Intermediate size must be divisible by world size"
        )
        ConditionalFeedForward.__init__(
            self,
            num_experts,
            dim,
            intermediate_size // world_size,
        )
        self.setup_tp(rank, group)

    def load_weights(
        self,
        tensor_values: dict[str, torch.Tensor],
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
        x_par = copy_to_tensor_model_parallel_region(x, self.group)
        out_par = ConditionalFeedForward.forward(self, x_par, expert_indices)
        return reduce_from_tensor_model_parallel_region(out_par, self.group)


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
