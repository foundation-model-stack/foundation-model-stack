# -*- coding: utf-8 -*-

"""
Part of code to support Qwen3 models
"""

# pylint: disable=unknown-option-value,protected-access
# pylint: disable=unused-argument
# pylint: disable=too-many-instance-attributes
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-many-arguments,too-many-positional-arguments

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Unpack

import torch
from torch import nn, Tensor

from fms import models
from fms.modules.positions import PositionEncoder
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
)
from fms.modules.attention import (MultiHeadAttention,
                                   AttentionKwargs,
                                   QKV, FusedQKV, UnfusedQKV,
                                   get_attention_type)

from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.linear import get_linear
from fms.modules.positions import RotaryEmbedding
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig

logger = logging.getLogger(__name__)


class Qwen3MuliHeadAttention(MultiHeadAttention):
    """Customize for Qwen3

    Args
    ----
        emb_dim : int
            Latent dimensionality of input and output tensors.
        emb_kq : int
            Latent dimensionality of each head in key and query projections (attention dimension).
        emb_v : int
            Latent dimensionality of each head in value projection (mixing dimension).
        nheads : int
            Number of attention heads.
        p_dropout : float|None
            Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
        use_bias : bool
            Include bias terms in fully-connected sublayers?
        fused : bool
            If True, qkv weights will be fused, otherwise qkv weights will be unfused.
        linear_config : Mapping[str, Any] | None
            Configuration for selection of linear modules (QKV, dense).
            Pass as {"linear_type": [str | callable], <other kwargs>}.
            "linear_type" should provide the string identifier of a registered type
            (e.g., "torch_linear", "gptq", ...) or a callable for module selection depending
            on module name. Additional config options should be provided as kwargs in
            linear_config.
    """
    
    def __init__(
        self,
        emb_dim,
        emb_kq,
        emb_v,
        nheads,
        kvheads,
        p_dropout=None,
        use_bias=False,
        position_encoder: Optional[PositionEncoder] = None,
        fused: bool = True,
        linear_config: Optional[Mapping[str, Any]] = None,
        scale_factor: Optional[float] = None,
    ):

        super(MultiHeadAttention, self).__init__()
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_dim = emb_dim
        self.emb_kq_per_head = emb_kq
        self.emb_v_per_head = emb_v
        self.p_dropout = p_dropout if p_dropout is not None else 0.0
        self.use_bias = use_bias
        self.fused = fused
        self.linear_config = linear_config
        self.scale_factor = scale_factor
        self.norm_eps = 1e-6
        
        self.in_proj: QKV = (FusedQKV if self.fused else UnfusedQKV)(
            self.emb_dim,
            self.nheads,
            self.kvheads,
            self.emb_kq_per_head,
            self.emb_v_per_head,
            self.use_bias,
            linear_config=linear_config,
        )

        self.dense = get_linear(
            self.nheads * self.emb_v_per_head,
            self.emb_dim,
            bias=use_bias,
            linear_config=linear_config,
        )
        self.q_norm = LayerNormParameterized(
                    self.emb_kq_per_head,
                    elementwise_scale=True,
                    elementwise_shift=False,
                    use_mean=False,
                    eps=self.norm_eps,
                    use_high_precision_pow=True,)

        self.k_norm = LayerNormParameterized(
                self.emb_kq_per_head,
                elementwise_scale=True,
                elementwise_shift=False,
                use_mean=False,
                eps=self.norm_eps,
                use_high_precision_pow=True,)
        if self.p_dropout:
            self.attn_dropout = nn.Dropout(self.p_dropout)
        self.position_encoder = position_encoder

    def forward(
    self,
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    position_ids=None,
    past_key_value_state: Optional[Tuple[Tensor | None, Tensor | None]] = None,
    use_cache=False,
    **attn_kwargs: Unpack[AttentionKwargs],
):
        """
        past_key_value_state: tuple
            the cache to be used in attention of the form (<self/cross>_key, <self/cross>_value)
        position_ids: Optional[torch.LongTensor]
            The position of each of the tokens encoded in q and k. Used for RoPE embeddings
        use_cache: bool
            if True, the kv states for self/cross attention will be saved,
            otherwise they will not be saved

        Returns
        -------
        tensor or tuple
            If use_cache=False, only the hidden state will be returned as a tensor.
            If use_cache=True, a tuple will be
            returned in the form (hidden_state, cache) where hidden_state is a
            tensor and cache is of the form specified in past_key_value_state
        """
        # q, k, v: batch_size x seq_len x emb_dim
        # mask: batch_size x seq_len x seq_len
        batch_size, q_len, _ = q.size()

        # if this is self attention, we always recompute
        # cross attention only gets computed when a cache does not exist
        # if we dont have the cache yet, we need to compute
        # d x (h x ds)
        # b x kvlen x d
        # b x kvlen x h x ds
        # b x h x kvlen x ds
        # todo: Cross attention (This always is true for now)
        q_out, k_out, v_out = self.in_proj(q, k, v)

        # note: transposes will be moved in a later PR to fix dis-contiguous tensor issues
        # queries = q_out.view(batch_size, q_len, self.nheads, self.emb_kq_per_head)
        queries = self.q_norm(q_out.view(batch_size, q_len, self.nheads, self.emb_kq_per_head))
        # keys = k_out.view(batch_size, q_len, self.kvheads, self.emb_kq_per_head)
        keys = self.k_norm(k_out.view(batch_size, q_len, self.kvheads, self.emb_kq_per_head))
        values = v_out.view(batch_size, q_len, self.kvheads, self.emb_v_per_head)

        # You want to apply rotary embeddings pre-cache
        if self.position_encoder is not None:
            queries, keys = self.position_encoder.adjusted_qk(
                queries, keys, position_ids, past_key_value_state, use_cache
            )

        attn_compute_dict = get_attention_type(**attn_kwargs)

        if use_cache:
            if past_key_value_state is None:
                past_key_value_state = (None, None)

            keys_compute, values_compute, keys_return, values_return = (
                attn_compute_dict["store"](
                    keys,
                    values,
                    past_key_value_state[0],
                    past_key_value_state[1],
                    **attn_kwargs,
                )
            )
        else:
            keys_compute, values_compute = keys, values

        if attn_compute_dict["is_prefill"](**attn_kwargs):
            attn = attn_compute_dict["compute_prefill"](
                queries,
                keys_compute,
                values_compute,
                self.nheads,
                self.kvheads,
                self.p_dropout if self.training else 0.0,
                self.scale_factor,
                **attn_kwargs,
            )
        else:
            attn = attn_compute_dict["compute_decode"](
                queries,
                keys_compute,
                values_compute,
                self.nheads,
                self.kvheads,
                self.p_dropout if self.training else 0.0,
                self.scale_factor,
                **attn_kwargs,
            )

        attn = attn.view(batch_size, q_len, self.nheads * self.emb_v_per_head)
        out = self.dense(attn)

        # if use_cache=True, we return the hidden_state as well as the kv cache
        if use_cache:
            return out, (keys_return, values_return) # type: ignore
        return out


@dataclass
class Qwen3Config(ModelConfig):

    # -----------------------------------------------------------------------
    # From transformers/src/transformers/models/qwen3/configuration_qwen3.py
    #  AND some FMS config name changes
    # -----------------------------------------------------------------------
    activation_fn:str = "silu"  # hf_config.hidden_act:str = "silu"
    attention_bias:bool = False
    emb_dim: int = 4096  # hf_config.hidden_size:int = 4096
    fused_weights: bool = True  # FMS Specific -- For CPU/GPU = T, AIU = F
    head_dim:int = 128
    hidden_grow_factor: float = 6144 / 2048  # hf_config.intermediate_size / hf_config.hidden_size
    initializer_range:float = 0.02
    intermediate_size:int = 22016
    kvheads: int = 8  # hf_config.num_key_value_heads:int = 8
    # layer_types:List[str] = []
    linear_config: Optional[Mapping[str, Any]] = None  # To support quantization
    max_position_embeddings:int = 40960
    max_expected_seq_len:int = 40960
    max_window_layers:int = 28
    multiple_of: int = 256  # borrowed from llama
    nheads: int = 16  # hf_config.num_attention_heads:int = 16
    nlayers: int = 28  # hf_config.num_hidden_layers:int = 28
    norm_eps: float = 1e-06  # hf_config.rms_norm_eps:float = 1e-6
    p_dropout: float = 0.0  # hf_config. attention_dropout:float = 0.0
    pad_id: int = -1  # borrowed from granite, we do need it
    # rope_scaling: Dict[str, Any] = {}
    rope_base:int = 1000000  # hf_config.rope_theta:int = 1000000
    sliding_window = None
    tie_heads: bool = True  # hf_config.tie_word_embeddings: bool = True
    use_cache:bool = True
    use_sliding_window:bool = False
    src_vocab_size:int = 15193  # hf_config.vocab_size:int = 15193


# Qwen3-1.7B
_1_7b_config = Qwen3Config()


class Qwen3Block(nn.Module):


    def __init__(self, config: Qwen3Config, rotary_emb: RotaryEmbedding):

        super().__init__()
        self.config = config
        emb_kq = self.config.emb_dim // self.config.nheads
        emb_v = self.config.emb_dim // self.config.nheads

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


        if self.config.kvheads == 0:
            kvheads = self.config.nheads
        else:
            kvheads = self.config.kvheads
            assert self.config.nheads % self.config.kvheads == 0
        # construct MultiHeadAttention object and pass in emb_dim: 2048, emb_kq: 2048/16, emb_v dim: 2048/16, 
        self.attn = Qwen3MuliHeadAttention(
            self.config.emb_dim,
            emb_kq,
            emb_v,
            self.config.nheads,
            kvheads,
            p_dropout=self.config.p_dropout,
            use_bias=False,
            position_encoder=rotary_emb,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
        )
        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn), # type: ignore
            p_dropout=self.config.p_dropout,
            use_bias=False,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
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
        **attn_kwargs: Unpack[AttentionKwargs],
    ):

        self_attn_past_key_value = past_key_value_state

        # first we do MHA and Add&Norm
        residual = x
        x = self.ln(x)
        x = self.attn(
            q=x,
            position_ids=position_ids,
            past_key_value_state=self_attn_past_key_value,
            use_cache=use_cache,
            **attn_kwargs,
        )
        cache = None
        if use_cache:
            x, cache = x
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        # residual connection
        x = x + residual

        # then we do FF and Add&Norm
        residual = x
        x = self.ff_ln(x)
        x = self.ff_sub_layer(x)
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        # another residual
        x = x + residual

        if use_cache:
            return (x, cache)
        else:
            return x


class Qwen3Headless(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
    ):
        super().__init__()
        self.config = config
        self.distributed_strategy = distributed_strategy

        self.embedding = nn.Embedding(
            self.config.src_vocab_size,
            self.config.emb_dim,
            padding_idx=self.config.pad_id,
        )

        self.rot_emb = RotaryEmbedding(
            dim=self.config.emb_dim // self.config.nheads,
            # ntk_scaling=False,
            max_seq_len=self.config.max_expected_seq_len,
            ratio=self.config.rope_base,
        )
        # RoPE init
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

        layers = []
        for i in range(self.config.nlayers):
            block: nn.Module = Qwen3Block(self.config, self.rot_emb)
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
            self.rot_emb.cached_freqs, # type: ignore
            self.rot_emb.max_seq_len_cached, # type: ignore
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
        if past_key_value_states is None or len(past_key_value_states) == 0:
            past_key_value_states = [None for _ in range(len(self.layers))]

        x_in = self.embedding(x_in)

        # this is the output cache for all the decoder layers
        present_key_value_states = []

        for i, layer in enumerate(self.layers):
            output = layer(
                x=x_in,
                position_ids=position_ids,
                past_key_value_state=past_key_value_states[i],
                use_cache=use_cache,
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


class Qwen3(nn.Module):

    def __init__(
        self,
        config: Optional[Qwen3Config] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = Qwen3Config()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.base_model = Qwen3Headless(self.config, self.distributed_strategy)
        self.head = nn.Linear(
            self.config.emb_dim, self.config.src_vocab_size, bias=False
        )

    @classmethod
    def from_config(cls, config: Qwen3Config) -> "Qwen3":
        return cls(config)

    def get_config(self) -> Qwen3Config:
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
        x: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[Tuple[torch.FloatTensor,]] = None,
        use_cache: bool = False,
        only_last_token: bool = False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        get_attention_type(**attn_kwargs)["validate_attn_kwargs"](
            input_ids=x,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            **attn_kwargs,
        )
        output, cache = self.base_model(
            x, position_ids, past_key_value_states, use_cache, **attn_kwargs
        )

        if only_last_token:
            output = output[:, -1, :]
        preds = self.head(output)

        if use_cache:
            return preds, cache
        else:
            return preds


_ARCHITECTURE_NAME = "qwen3"


def _qwen3_factory_factory(config):
    def factory(**kwargs):
        return Qwen3(config, **kwargs)

    return factory


models.register_model(_ARCHITECTURE_NAME, "1.7b", _qwen3_factory_factory(_1_7b_config))


# =============== Serialization ==================


serialization.register_adapter_step(
    _ARCHITECTURE_NAME,
    "swiglu_unfused_to_fused",
    serialization._mlp_glu_unfused_to_fused_adapter_step,
)


def _weight_fusion(
    input_sd: Mapping[str, Any], model_config: Optional[Qwen3Config] = None, **kwargs
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


serialization.register_adapter_step(_ARCHITECTURE_NAME, "weight_fusion", _weight_fusion)


def _hf_gptq_qwen3_check(
    input_sd: Mapping[str, Any], model_config: Optional[Qwen3Config] = None, **kwargs
) -> Mapping[str, Any]:
    """_summary_

    Args:
        input_sd (Mapping[str, Any]): _description_
        model_config (Optional[QwenConfig], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        Mapping[str, Any]: _description_
    """
    has_fused_weights = True
    linear_type = "torch_linear"
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False
        if model_config.linear_config:
            linear_type = model_config.linear_config["linear_type"]

    if "gptq" in linear_type and has_fused_weights:
        raise ValueError(
            "GPTQ HF Qwen checkpoints cannot be loaded into a model with fused weights"
        )

    return input_sd


serialization.register_adapter_step(
    _ARCHITECTURE_NAME, "hf_gptq_fusion_check", _hf_gptq_qwen3_check
)

# pylint: disable=wrong-import-position,wrong-import-order
import atexit  # noqa: E402
import os  # noqa: E402
KWR_DEBUG = len(os.getenv("KWR_DEBUG", "")) > 0
mapping_dict: Dict[str, str] = {}
no_mapping_dict: Dict[str, int] = {}
if KWR_DEBUG:
    def mapping_dict_cleanup() -> None:
        """
        This function will be called automatically when the script exits.
        """
        size = len(mapping_dict)  # noqa: F821
        print(f"qwen3.py:_hf_to_fms_names():mapping_dict()/{size}", flush=True)
        for key in sorted(mapping_dict.keys()):  # noqa: F821
            print(f"  {key:<60} : {mapping_dict[key]}", flush=True)  # noqa: F821
        size = len(no_mapping_dict)  # noqa: F821
        print(f"qwen3.py:_hf_to_fms_names():no_mapping_dict()/{size}", flush=True)
        for key in sorted(no_mapping_dict.keys()):  # noqa: F821
            print(f"  {key:<60} : {no_mapping_dict[key]}", flush=True)  # noqa: F821
    atexit.register(mapping_dict_cleanup)


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    """_summary_

    Args:
        input_sd (Mapping[str, Any]): _description_

    Returns:
        Mapping[str, Any]: _description_
    """
    # base_model.layers.3.attn.in_proj.q_norm.weight
    replacements = [
        (r"^lm_head.weight", "head.weight"),
        (r"^model.embed_tokens.weight", "base_model.embedding.weight"),
        (r"^model.norm", "base_model.dec_norm"),
        (r"^model.layers", "base_model.layers"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        # (r"self_attn\.k_norm", "attn.in_proj.k_norm"),
        # (r"self_attn\.q_norm", "attn.in_proj.q_norm"),
        (r"self_attn\.k_norm", "attn.k_norm"),
        (r"self_attn\.q_norm", "attn.q_norm"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
        (r"self_attn\.k_norm", "attn.in_proj.k_norm"),
        (r"self_attn\.q_norm", "attn.in_proj.q_norm"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param
        if KWR_DEBUG:
            if new_name == name:
                global no_mapping_dict   # pylint: disable=global-variable-not-assigned
                if name in no_mapping_dict:
                    no_mapping_dict[name] += 1
                else:
                    no_mapping_dict[name] = 1
            global mapping_dict # pylint: disable=global-variable-not-assigned
            if name in mapping_dict:
                print(f"[WARNING]: key '{name}' already in mapping_dict")
            else:
                mapping_dict[name] = new_name
    return new_sd


serialization.register_adapter_step(
    _ARCHITECTURE_NAME, "hf_to_fms_names", _hf_to_fms_names
)


def _get_rope_params(linear_type: str) -> list[str]:
    """_summary_

    Args:
        linear_type (str): _description_

    Returns:
        list[str]: _description_
    """
    if "gptq" in linear_type:
        return ["qweight", "scales", "qzeros", "bias"]
    # torch.nn.Linear
    return ["weight", "bias"]


def _hf_to_fms_rope(
    input_sd: Mapping[str, Any], model_config: Optional[Qwen3Config] = None, **kwargs
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
        f"layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(rope_params)})"
    )
    for name, param in input_sd.items():
        # hf -> fms requires a transpose operation for the query and key
        # weight and bias parameters for Llama models
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

            if temp.dim() == 2:  # weight
                temp_view = temp.view(num_heads, 2, -1, temp.size(1))
            else:  # bias
                temp_view = temp.view(num_heads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*temp.size())

            if "gptq" in linear_type and temp.dim() == 2:
                temp = temp.transpose(0, 1)

            new_sd[name] = temp
        else:
            new_sd[name] = param

    return new_sd


serialization.register_adapter_step(
    _ARCHITECTURE_NAME, "hf_to_fms_rope", _hf_to_fms_rope
)

serialization.register_adapter(
    _ARCHITECTURE_NAME,
    "hf",
    ["hf_to_fms_names", "hf_to_fms_rope", "hf_gptq_fusion_check", "weight_fusion"],
)


if __name__ == "__main__":
    print("Nothing to do.")
