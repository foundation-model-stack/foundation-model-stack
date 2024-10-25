import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from fms import models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
    TensorParallelStrategy,
    UniformModelParallelStrategy,
)
from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig
from fms.utils.tokenizers import _has_hf, get_tokenizer


logger = logging.getLogger(__name__)


@dataclass
class GraniteConfig(ModelConfig):
    src_vocab_size: int = 32_000  # can be set by tokenizer
    emb_dim: int = 4096
    norm_eps: float = 1e-5
    nheads: int = 32
    kvheads: int = 0
    nlayers: int = 32
    pad_id: int = -1
    hidden_grow_factor: float = 8 / 3
    multiple_of: int = 256
    activation_fn: str = "swish"
    p_dropout: float = 0.0
    max_expected_seq_len: int = 4096
    ntk_scaling: bool = False
    attn_bias: bool = False
    mlp_bias: bool = False
    tie_heads: bool = False
    rope_theta: float = 10_000.0
    embedding_multiplier: float = 1.0
    logits_scaling: float = 1.0
    residual_multiplier: float = 1.0
    attention_multiplier: float = 1.0
    linear_config: Optional[Mapping[str, Any]] = None
    unfuse_strategy: Optional[str] = None  # TODO: could be an Enum


class GraniteBlock(nn.Module):
    def __init__(self, config: GraniteConfig, rotary_emb: RotaryEmbedding):
        super(GraniteBlock, self).__init__()
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

        self.attn = MultiHeadAttention(
            self.config.emb_dim,
            emb_kq,
            emb_v,
            self.config.nheads,
            kvheads,
            p_dropout=self.config.p_dropout,
            use_bias=self.config.attn_bias,
            position_encoder=rotary_emb,
            fused=(self.config.unfuse_strategy != "pre"),
            linear_config=self.config.linear_config,
            scale_factor=self.config.attention_multiplier,
        )
        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=self.config.mlp_bias,
            fused=(self.config.unfuse_strategy != "pre"),
            linear_config=self.config.linear_config,
        )

        if self.config.p_dropout != 0:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x,
        *,
        mask=None,
        position_ids=None,
        past_key_value_state=None,
        use_cache=False,
        is_causal_mask=False,
        attn_algorithm=None,
    ):
        # if the cache is not empty, we need to get the kv cache for self and cross attention
        self_attn_past_key_value = past_key_value_state
        # if past_key_value_state is not None:
        #     self_attn_past_key_value = past_key_value_state[:2]
        # else:
        #     self_attn_past_key_value = None

        # first we do MHA and Add&Norm
        residual = x
        x = self.ln(x)
        x = self.attn(
            q=x,
            mask=mask,
            position_ids=position_ids,
            attn_algorithm=attn_algorithm,
            past_key_value_state=self_attn_past_key_value,
            use_cache=use_cache,
            is_self=True,
            is_causal_mask=is_causal_mask,
        )
        cache = None
        if use_cache:
            x, cache = x
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        # residual connection
        x = x * self.config.residual_multiplier + residual

        # then we do FF and Add&Norm
        residual = x
        x = self.ff_ln(x)
        x = self.ff_sub_layer(x)
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        # another residual
        x = x * self.config.residual_multiplier + residual

        if use_cache:
            return (x, cache)
        else:
            return x


class GraniteHeadless(nn.Module):
    def __init__(
        self,
        config: GraniteConfig,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
    ):
        super(GraniteHeadless, self).__init__()
        self.config = config
        self.distributed_strategy = distributed_strategy

        self.width = self.config.emb_dim
        self.pad_id = self.config.pad_id
        self.max_expected_seq_len = self.config.max_expected_seq_len

        self.embedding = nn.Embedding(
            self.config.src_vocab_size,
            self.config.emb_dim,
            padding_idx=self.config.pad_id,
        )

        self.rot_emb = RotaryEmbedding(
            dim=self.config.emb_dim // self.config.nheads,
            ntk_scaling=self.config.ntk_scaling,
            max_seq_len=self.config.max_expected_seq_len,
            ratio=self.config.rope_theta,
        )
        # RoPE init
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

        layers = []
        for i in range(self.config.nlayers):
            block: nn.Module = GraniteBlock(self.config, self.rot_emb)
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
        mask=None,
        position_ids=None,
        past_key_value_states=None,
        use_cache=False,
        attn_algorithm=None,
    ):
        # Embed the given vocabulary indices using the given attention mask, with pre-/post-norm and dropout as specified
        # x_in: batch_size x seq_len
        # mask: batch_size x seq_len x seq_len
        # bias: nheads x seq_len x seq_len
        if past_key_value_states is None or len(past_key_value_states) == 0:
            past_key_value_states = [None for _ in range(len(self.layers))]

        qlen = x_in.size(1)
        klen = x_in.size(1)

        # if we are using the cache, the key length needs to be extended with the past keys length
        if use_cache and past_key_value_states[0] is not None:
            klen += past_key_value_states[0][0].size(-2)

        # if mask is none, we need to specify causal mask
        if mask is None:
            # we are caching and can assume all 1s in the mask
            if use_cache and klen != 1 and qlen == 1:
                # b x h x qlen x kvlen
                is_causal_mask = False
            else:
                is_causal_mask = True
        else:
            is_causal_mask = False

        x_in = self.embedding(x_in)
        x_in = x_in * self.config.embedding_multiplier

        # this is the output cache for all the decoder layers
        present_key_value_states = []

        for i, layer in enumerate(self.layers):
            output = layer(
                x=x_in,
                mask=mask,
                position_ids=position_ids,
                past_key_value_state=past_key_value_states[i],
                use_cache=use_cache,
                is_causal_mask=is_causal_mask,
                attn_algorithm=attn_algorithm,
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


class Granite(nn.Module):
    def __init__(
        self,
        config: Optional[GraniteConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(Granite, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = GraniteConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.base_model = GraniteHeadless(self.config, self.distributed_strategy)
        self.head = nn.Linear(
            self.config.emb_dim, self.config.src_vocab_size, bias=False
        )

    @classmethod
    def from_config(cls, config: GraniteConfig) -> "Granite":
        return cls(config)

    def get_config(self) -> GraniteConfig:
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
        x,
        mask=None,
        position_ids=None,
        past_key_value_states=None,
        use_cache=False,
        only_last_token=False,
        attn_algorithm=None,
    ):
        output, cache = self.base_model(
            x, mask, position_ids, past_key_value_states, use_cache, attn_algorithm
        )

        if only_last_token:
            output = output[:, -1, :]
        preds = self.head(output)
        preds = preds / self.config.logits_scaling

        if use_cache:
            return preds, cache
        else:
            return preds


_8b_config = GraniteConfig(
    src_vocab_size=49155,
    emb_dim=4096,
    norm_eps=1e-5,
    nheads=32,
    kvheads=8,
    nlayers=40,
    hidden_grow_factor=12800 / 4096,
    max_expected_seq_len=8192,
    rope_theta=10_000.0,
    pad_id=0,
    p_dropout=0.0,  # overwriting config.json
    tie_heads=True,
    embedding_multiplier=12.0,
    logits_scaling=16.0,
    residual_multiplier=0.22,
    attention_multiplier=0.0078125,
)

_architecture_name = "granite"


def _granite_factory_factory(config):
    def factory(**kwargs):
        return Granite(config, **kwargs)

    return factory


models.register_model(_architecture_name, "8b", _granite_factory_factory(_8b_config))


_convert_to_fused = lambda sd: serialization._legacy_mlp_glu_unfused_to_fused_adapter(
    serialization._legacy_attn_unfused_to_fused_adapter(sd)
)


def _hf_sd_to_fms_sd(hf_sd: Mapping) -> Mapping:
    replacements = [
        (r"^lm_head.weight", "head.weight"),
        (r"^model.embed_tokens.weight", "base_model.embedding.weight"),
        (r"^model.norm", "base_model.dec_norm"),
        (r"^model.layers", "base_model.layers"),
        (r"self_attn\.k_proj", "attn.key"),
        (r"self_attn\.v_proj", "attn.value"),
        (r"self_attn\.q_proj", "attn.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]
    new_sd = {}

    trans_required_pattern = re.compile(
        "base_model.layers.[0-9]+.attn.(query|key).(weight|bias)"
    )
    for name, param in hf_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

        # hf -> fms requires a transpose operation for the query and key
        # weight and bias parameters
        # This transpose is due to the different implementation of RoPE in
        # HF and FMS. While FMS follows the original RoPE paper
        # (https://arxiv.org/abs/2104.09864), HF has its own implementation
        # that doesn't respect the order of outputs. This is OK as long as you
        # rearrange the weights of the query and key projections, as the
        # combination projection + RoPE ends up producing the same outputs.
        # Therefore, to make FMS produce the correct order of outputs when
        # loading from an HF checkpoint, we need to undo the transformation
        # that HF does from the original Meta weights:
        if bool(trans_required_pattern.match(new_name)):
            temp = new_sd[new_name]
            # nheads is used in the transformation required for hf->fms
            if temp.size(0) == 2560:
                head_size = 80  # granite 3b code
            else:
                head_size = 128  # every other Granite model in existence
            nheads = int(temp.size(0) / head_size)

            if temp.dim() == 2:  # weight
                temp_view = temp.view(nheads, 2, -1, temp.size(1))
            else:  # bias
                temp_view = temp.view(nheads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*temp.size())

            new_sd[new_name] = temp

    fused_sd = _convert_to_fused(new_sd)

    return fused_sd


serialization.register_adapter("granite", "hf", _hf_sd_to_fms_sd)


def convert_hf_granite(hf_model: "GraniteForCausalLM") -> Granite:  # type: ignore
    """
    Convert a Granite huggingface model to an fms model

    Parameters
    ----------
    hf_model: GraniteForCausalLM
        a Granite Huggingface model

    Returns
    -------
    Granite
        an FMS Granite model
    """

    if not _has_hf:
        raise ImportError(
            "in order to convert huggingface weights, you need to have transformers installed"
        )

    import re

    config = GraniteConfig(
        src_vocab_size=hf_model.config.vocab_size,
        emb_dim=hf_model.config.hidden_size,
        norm_eps=hf_model.config.rms_norm_eps,
        nheads=hf_model.config.num_attention_heads,
        nlayers=hf_model.config.num_hidden_layers,
        hidden_grow_factor=hf_model.config.intermediate_size
        / hf_model.config.hidden_size,
        multiple_of=1,  # this is set to 1 as it is encoded in the hidden dimension
        activation_fn=hf_model.config.hidden_act,
        max_expected_seq_len=hf_model.config.max_position_embeddings,
        rope_theta=hf_model.config.rope_theta,
    )
    model = Granite(config)
    count_parameters = lambda m: sum(p.numel() for p in m.parameters())
    assert count_parameters(model) == count_parameters(hf_model)

    hf_sd = hf_model.model.state_dict()

    replacements = [
        (r"^embed_tokens.weight", "base_model.embedding.weight"),
        (r"^norm", "base_model.dec_norm"),
        (r"^layers", "base_model.layers"),
        (r"self_attn\.k_proj", "attn.key"),
        (r"self_attn\.v_proj", "attn.value"),
        (r"self_attn\.q_proj", "attn.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in hf_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

    model.load_state_dict(new_sd, strict=False)
    model.shared.head.weight = hf_model.lm_head.weight

    # model.rot_emb.freqs = hf_model.model.layers[0].self_attn.rotary_emb.inv_freq
    for layer in model.layers:
        q = layer.attn.query.weight.data
        q = (
            q.view(model.config.nheads, 2, -1, q.size(1))
            .transpose(1, 2)
            .reshape(*q.size())
        )
        layer.attn.query.weight.data = q

        k = layer.attn.key.weight.data
        k = (
            k.view(model.config.nheads, 2, -1, k.size(1))
            .transpose(1, 2)
            .reshape(*k.size())
        )
        layer.attn.key.weight.data = k

    return model
