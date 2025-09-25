import dataclasses
import logging
import math
import re
from typing import Any, List, Mapping, Optional, Tuple
from typing_extensions import Unpack

import torch
import torch.nn as nn

from fms import models
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
    get_attention_type,
)
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.modules.ssm import SSM, SSMCacheUnit
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig
from fms.utils.headless import gather_outputs


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BambaConfig(ModelConfig):
    src_vocab_size: int = 32768
    emb_dim: int = 4096
    nheads: int = 128
    kvheads: int = 8
    head_dim: int = 64
    norm_eps: float = 1e-5
    nlayers: int = 64
    activation_fn: str = "swish"
    attn_layer_indices: List[int] = dataclasses.field(default_factory=lambda: [])
    max_expected_seq_len: int = 262144
    ntk_scaling: bool = False
    tie_heads: bool = False
    rope_theta: float = 10_000.0
    p_dropout: float = 0.0
    conv_kernel: int = 4
    state_size: int = 256
    hidden_grow_factor: float = 2.0
    mamba_expand: float = 2.0
    mamba_n_heads: int = 128
    multiple_of: int = 256
    use_bias: bool = False
    use_conv_bias: bool = True
    n_groups: int = 8
    chunk_size: int = 256
    linear_config: Optional[Mapping[str, Any]] = None
    fused_weights: bool = True


class BambaBlock(nn.Module):
    def __init__(self, config: BambaConfig, rotary_emb, layer_index: int):
        super(BambaBlock, self).__init__()
        self.config = config
        self.layer_index = layer_index

        if layer_index in config.attn_layer_indices:
            if self.config.kvheads == 0:
                kvheads = self.config.nheads
            else:
                kvheads = self.config.kvheads
                assert self.config.nheads % self.config.kvheads == 0

            self.attn = MultiHeadAttention(
                self.config.emb_dim,
                self.config.emb_dim // self.config.nheads,
                self.config.emb_dim // self.config.nheads,
                self.config.nheads,
                kvheads,
                p_dropout=self.config.p_dropout,
                position_encoder=rotary_emb,
                fused=self.config.fused_weights,
                linear_config=self.config.linear_config,
            )
        else:
            self.ssm = SSM(
                self.config.mamba_n_heads,
                self.config.emb_dim,
                self.config.state_size,
                self.config.conv_kernel,
                self.config.mamba_expand,
                self.config.use_bias,
                self.config.use_conv_bias,
                self.config.activation_fn,
                self.config.norm_eps,
                self.config.n_groups,
                self.config.head_dim,
                self.config.chunk_size,
            )

        self.is_mamba_layer = hasattr(self, "ssm")

        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=False,
            fused=self.config.fused_weights,
            linear_config=None,
        )

        self.ln = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.ff_ln = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )

        if self.config.p_dropout != 0:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x,
        *,
        position_ids=None,
        past_key_value_state=None,
        use_cache=False,
        cache_position=None,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        seqlen_offset = x.shape[1]
        residual = x
        x = self.ln(x)

        if self.is_mamba_layer:
            x = self.ssm(
                x,
                past_key_value_state=past_key_value_state,
                use_cache=use_cache,
                cache_position=cache_position,
                mask=attn_kwargs.get("mask", None),
            )
        else:
            x = self.attn(
                x,
                position_ids=position_ids,
                past_key_value_state=past_key_value_state,
                use_cache=use_cache,
                **attn_kwargs,
            )

        cache = None
        if use_cache or isinstance(x, tuple):
            x, cache = x

        if self.config.p_dropout != 0:
            x = self.dropout(x)

        # residual connection
        x = x + residual
        residual = x

        x = self.ff_ln(x)
        x = self.ff_sub_layer(x)
        if self.config.p_dropout != 0:
            x = self.dropout(x)

        # another residual
        x = x + residual
        if use_cache:
            if self.is_mamba_layer:
                cache.seqlen_offset += seqlen_offset  # type: ignore
                cache.has_previous_state = True  # type: ignore

            return x, cache
        else:
            return x


class BambaHeadless(nn.Module):
    def __init__(self, config: BambaConfig, distributed_strategy: DistributedStrategy):
        super(BambaHeadless, self).__init__()
        self.config = config
        self.distributed_strategy = distributed_strategy

        self.embedding = nn.Embedding(self.config.src_vocab_size, self.config.emb_dim)

        rope_scaling = {"rope_type": "ntk" if self.config.ntk_scaling else "regular"}

        self.rot_emb = RotaryEmbedding(
            dim=self.config.emb_dim // self.config.nheads,
            scaling=rope_scaling,
            max_seq_len=self.config.max_expected_seq_len,
            ratio=self.config.rope_theta,
            partial_rope=0.5,
        )

        # RoPE init
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

        layers = []
        for i in range(self.config.nlayers):
            block: nn.Module = BambaBlock(self.config, self.rot_emb, i)
            block = self.distributed_strategy.distribute_layer(block, i)
            layers.append(block)
        self.layers = nn.ModuleList(layers)

        dec_norm = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.dec_norm = self.distributed_strategy.distribute_module(
            dec_norm, final_layers=True
        )

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)

        self.attn_layer_indices = set(self.config.attn_layer_indices)
        self.attn_layer_ind = (
            -1
            if len(self.config.attn_layer_indices) == 0
            else next(iter(self.config.attn_layer_indices))
        )

    def reset_parameters(self):
        nn.init.trunc_normal_(
            self.embedding.weight, mean=0.0, std=self.config.emb_dim**-0.5
        )

        # RoPE init
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

        # Call reset_parameters for relevant sub-layers
        for m in self.modules():
            if (
                isinstance(m, MultiHeadAttention)
                or isinstance(m, GatedLinearUnit)
                or isinstance(m, LayerNormParameterized)
            ):
                m.reset_parameters()

    def _clean_up_rot_emb_cache(
        self,
        cached_freqs: dict[Optional[torch.device], dict[int, torch.Tensor]],
        max_seq_len_cached: dict[Optional[torch.device], int],
    ):
        # remove meta tensors from cached_freqs
        for dev in list(cached_freqs.keys()):
            for alp in list(cached_freqs[dev].keys()):
                if cached_freqs[dev][alp].device == torch.device("meta"):
                    del cached_freqs[dev][alp]
                    if len(cached_freqs[dev]) == 0:
                        del cached_freqs[dev]
                        del max_seq_len_cached[dev]

    def post_init(self):
        # This function is called in `get_model` after the model is
        # fully initalized on the correct device

        self._clean_up_rot_emb_cache(
            self.rot_emb.cached_freqs,
            self.rot_emb.max_seq_len_cached,
        )

        # init RoPE on the right device(s)
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

    def forward(
        self,
        x_in,
        position_ids=None,
        past_key_value_states=None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        # Embed the given vocabulary indices using the given attention mask, with pre-/post-norm and dropout as specified
        # x_in: batch_size x seq_len
        # mask: batch_size x seq_len x seq_len
        # bias: nheads x seq_len x seq_len
        if past_key_value_states is None or len(past_key_value_states) == 0:
            if use_cache:
                past_key_value_states = []
                for i in range(len(self.layers)):
                    if i in self.attn_layer_indices:
                        past_key_value_states.append(None)
                    else:
                        past_key_value_states.append(
                            SSMCacheUnit(
                                self.config.emb_dim,
                                self.config.mamba_n_heads,
                                self.config.head_dim,
                                self.config.conv_kernel,
                                self.config.mamba_expand,
                                self.config.n_groups,
                                self.config.state_size,
                                x_in.size(0),
                                self.embedding.weight.dtype,
                                str(self.embedding.weight.device),
                            )
                        )
            else:
                past_key_value_states = [None for _ in range(len(self.layers))]

        klen = x_in.size(1)

        # if we are using the cache, the key length needs to be extended with the past keys length
        if self.attn_layer_ind != -1:
            if use_cache and past_key_value_states[self.attn_layer_ind] is not None:
                klen += past_key_value_states[self.attn_layer_ind][0].size(-2)

        x_in = self.embedding(x_in)

        # this is the output cache for all the decoder layers
        present_key_value_states = []

        if position_ids is None:
            cache_position = (
                torch.arange(x_in.shape[1], device=x_in.device) + klen
            )  # TODO: Explore issue with this path
        else:
            cache_position = position_ids.max(dim=0).values

        for i, layer in enumerate(self.layers):
            output = layer(
                x=x_in,
                position_ids=position_ids,
                past_key_value_state=past_key_value_states[i],
                use_cache=use_cache,
                cache_position=cache_position,
                **attn_kwargs,
            )

            if use_cache:
                x_in, present_key_value_state = output
                present_key_value_states.append(present_key_value_state)
            else:
                x_in = output

        dec_out = x_in
        dec_out = self.dec_norm(dec_out)
        if self.config.p_dropout:
            dec_out = self.dropout(dec_out)

        return dec_out, present_key_value_states


class Bamba(nn.Module):
    def __init__(
        self,
        config: Optional[BambaConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(Bamba, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = BambaConfig()
        self.config = self.config.updated(**kwargs)

        self.distributed_strategy = distributed_strategy

        self.base_model = BambaHeadless(self.config, self.distributed_strategy)
        self.head = nn.Linear(
            self.config.emb_dim, self.config.src_vocab_size, bias=False
        )

    @classmethod
    def from_config(cls, config: BambaConfig) -> "Bamba":
        return cls(config)

    def get_config(self) -> BambaConfig:
        return self.config

    def reset_parameters(self):
        self.head.weight.data.normal_(
            0,
            1 / math.sqrt(math.sqrt(self.config.emb_dim * self.config.src_vocab_size)),
        )
        self.base_model.reset_parameters()

    def post_init(self):
        # if this model ties weights, they are tied here
        if self.config.tie_heads:
            # handle assignment of non-meta weights to meta parameters
            if self.head.weight.device == torch.device("meta"):
                self.head.weight = self.base_model.embedding.weight
            else:
                self.base_model.embedding.weight = self.head.weight

        self.base_model.post_init()

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[
            List[SSMCacheUnit | Tuple[torch.FloatTensor,]]
        ] = None,
        use_cache: bool = False,
        last_n_tokens: int = 0,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        get_attention_type(**attn_kwargs)["validate_attn_kwargs"](
            input_ids=x,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            **attn_kwargs,
        )
        output, cache = self.base_model(
            x,
            position_ids,
            past_key_value_states,
            use_cache,
            **attn_kwargs,
        )

        output = gather_outputs(output, last_n_tokens, **attn_kwargs)
        preds = self.head(output)

        if use_cache:
            return preds, cache
        else:
            return preds


_architecture_name = "bamba"


def _bamba_factory_factory(config):
    def factory(**kwargs):
        return Bamba(config, **kwargs)

    return factory


_bamba_9_8b_config = BambaConfig(
    src_vocab_size=128256,
    emb_dim=4096,
    tie_heads=False,
    norm_eps=1e-5,
    kvheads=8,
    nlayers=32,
    nheads=32,
    use_bias=False,
    head_dim=64,
    n_groups=1,
    hidden_grow_factor=3.5,
    mamba_expand=2.0,
    state_size=128,
    conv_kernel=4,
    use_conv_bias=True,
    chunk_size=256,
    attn_layer_indices=[9, 18, 27],
    mamba_n_heads=128,
    max_expected_seq_len=262144,
    p_dropout=0.0,
)

models.register_model(
    _architecture_name, "9_8b", _bamba_factory_factory(_bamba_9_8b_config)
)


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        (r"^lm_head.weight", "head.weight"),
        (r"^model.embed_tokens.weight", "base_model.embedding.weight"),
        (r"^model.final_layernorm", "base_model.dec_norm"),
        (r"^model.layers", "base_model.layers"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mamba\.conv1d", "ssm.conv1d"),
        (r"mamba\.in_proj", "ssm.in_proj"),
        (r"mamba\.norm", "ssm.norm"),
        (r"mamba\.out_proj", "ssm.out_proj"),
        (r"mamba\.dt_bias", "ssm.dt_bias"),
        (r"mamba\.D", "ssm.D"),
        (r"mamba\.A_log", "ssm.A_log"),
        (r"feed_forward\.gate_proj", "ff_sub_layer.wg"),
        (r"feed_forward\.up_proj", "ff_sub_layer.w1"),
        (r"feed_forward\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"pre_ff_layernorm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param
    return new_sd


serialization.register_adapter_step("bamba", "hf_to_fms_names", _hf_to_fms_names)


def _get_rope_params(linear_type: str) -> list[str]:
    if "gptq" in linear_type:
        return ["qweight", "scales", "qzeros", "bias"]
    else:  # torch.nn.Linear
        return ["weight", "bias"]


def _hf_to_fms_rope(
    input_sd: Mapping[str, Any], model_config: Optional[BambaConfig] = None, **kwargs
) -> Mapping[str, Any]:
    new_sd = {}

    if model_config:
        head_size = model_config.emb_dim // model_config.nheads
        linear_type = "torch_linear"
        if model_config.linear_config:
            linear_type = model_config.linear_config["linear_type"]
    else:
        logger.warning("Missing model_config, assuming defaults for head_size")
        head_size = 128  # Good default for most models
        linear_type = "torch_linear"

    rope_params = _get_rope_params(linear_type)
    trans_required_pattern = re.compile(
        f"base_model.layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(rope_params)})"
    )
    for name, param in input_sd.items():
        # hf -> fms requires a transpose operation for the query and key
        # weight and bias parameters for Bamba models
        # This transpose is due to the different implementation of RoPE in
        # HF and FMS. While FMS follows the original RoPE paper
        # (https://arxiv.org/abs/2104.09864), HF has its own implementation
        # that doesn't respect the order of outputs. This is OK as long as you
        # rearrange the weights of the query and key projections, as the
        # combination projection + RoPE ends up producing the same outputs.
        # Therefore, to make FMS produce the correct order of outputs when
        # loading from an HF checkpoint, we need to undo the transformation
        # that HF does from the original Meta weights:
        if bool(trans_required_pattern.match(name)):
            temp = param
            if "gptq" in linear_type and temp.dim() == 2:
                # GPTQ qweights are [in_feat, out_feat] (unlike usual [out_feat, in_feat])
                # and are fully transposed before & after process
                temp = temp.transpose(0, 1)
            # num_heads is used in the transformation required for hf->fms
            # can't be precomputed because q and k might have different num_heads
            num_heads = temp.size(0) // head_size
            rope_size = int(head_size * 0.5)

            if temp.dim() == 2:  # weight
                temp_heads = temp.view(num_heads, -1, temp.size(1))
                temp_rope = temp_heads[:, :rope_size]
                temp_rope_view = temp_rope.view(num_heads, 2, -1, temp.size(1))
            else:  # bias
                temp_heads = temp.view(num_heads, -1)
                temp_rope = temp_heads[:, :rope_size]
                temp_rope_view = temp_rope.view(num_heads, 2, -1)
            temp_rope = temp_rope_view.transpose(1, 2).reshape(*temp_rope.size())
            temp = torch.cat([temp_rope, temp_heads[:, rope_size:]], dim=-2).reshape(
                *temp.size()
            )

            if "gptq" in linear_type and temp.dim() == 2:
                temp = temp.transpose(0, 1)

            new_sd[name] = temp
        else:
            new_sd[name] = param

    return new_sd


serialization.register_adapter_step("bamba", "hf_to_fms_rope", _hf_to_fms_rope)


def _weight_fusion(
    input_sd: Mapping[str, Any], model_config: Optional[BambaConfig] = None, **kwargs
) -> Mapping[str, Any]:
    has_fused_weights = True
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False

    new_sd = input_sd
    if has_fused_weights:
        new_sd = serialization._mlp_glu_unfused_to_fused_adapter_step(
            serialization._attn_unfused_to_fused_step(new_sd)
        )
    return new_sd


serialization.register_adapter_step("bamba", "weight_fusion", _weight_fusion)


def _hf_gptq_bamba_check(
    input_sd: Mapping[str, Any], model_config: Optional[BambaConfig] = None, **kwargs
) -> Mapping[str, Any]:
    has_fused_weights = True
    linear_type = "torch_linear"
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False
        if model_config.linear_config:
            linear_type = model_config.linear_config["linear_type"]

    if "gptq" in linear_type and has_fused_weights:
        raise ValueError(
            "GPTQ HF Bamba checkpoints cannot be loaded into a model with fused weights"
        )

    return input_sd


serialization.register_adapter_step(
    "bamba", "hf_gptq_fusion_check", _hf_gptq_bamba_check
)

serialization.register_adapter(
    "bamba",
    "hf",
    ["hf_to_fms_names", "hf_to_fms_rope", "hf_gptq_fusion_check", "weight_fusion"],
)
