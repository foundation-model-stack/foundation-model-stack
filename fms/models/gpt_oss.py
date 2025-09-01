from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple
import re
import math

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
from fms.utils import serialization
from fms.utils.config import ModelConfig

from fms.modules.feedforward import MOEFeedForward
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.positions import RotaryEmbedding

from fms.modules.ssm import RMSNormGated

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


@dataclass
class GptOssConfig(ModelConfig):
    num_experts: int = 128
    src_vocab_size: int = 201088
    emb_dim: int = 2880
    hidden_dim: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    sliding_window: int = 128
    rope_base: float = 150000.0
    tie_heads = False
    activation_fn: str = "silu"
    initializer_range: float = 0.02
    max_expected_seq_len = 131072
    top_k_experts = 4
    router_aux_loss_coef: float = 0.9
    output_router_logits = False
    use_cache = True
    layer_types = None
    pad_id: int = -1
    nheads: int = 64
    nlayers: int = 24
    norm_eps: float = 1e-05
    kvheads: int = 8
    p_dropout: float = 0.0
    fused_weights: bool = True
    linear_config: Optional[Mapping[str, Any]] = None
    hidden_grow_factor: float = hidden_dim / emb_dim
    multiple_of: int = 256
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    logits_scaling: float = 1.0
    attention_multiplier: float = 1.0


class GptOssBlock(nn.Module):
    def __init__(self, config: GptOssConfig, rotary_emb: RotaryEmbedding):
        super(GptOssBlock, self).__init__()
        self.config = config
        emb_kq = self.config.head_dim
        emb_v = self.config.head_dim

        self.ln = RMSNormGated(config.emb_dim, eps=config.norm_eps)
        self.ff_ln = RMSNormGated(config.emb_dim, eps=config.norm_eps)

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
            use_bias=True,
            position_encoder=rotary_emb,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
            has_sinks=True,
        )

        if self.config.p_dropout != 0:
            self.dropout = nn.Dropout(self.config.p_dropout)

        self.ff_sub_layer = MOEFeedForward(
            self.config.num_experts,
            self.config.top_k_experts,
            self.config.hidden_dim,
            self.config.hidden_dim,
            use_bias=True,
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
        # if the cache is not empty, we need to get the kv cache for self and cross attention
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


class GptOssHeadless(nn.Module):
    def __init__(
        self,
        config: GptOssConfig,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(GptOssHeadless, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = GptOssConfig
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.embedding = nn.Embedding(
            self.config.src_vocab_size,
            self.config.emb_dim,
            padding_idx=self.config.pad_id,
        )

        rope_scaling = {"rope_type": "regular"}

        self.rot_emb = RotaryEmbedding(
            dim=self.config.head_dim,
            scaling=rope_scaling,
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
            block: nn.Module = GptOssBlock(self.config, self.rot_emb)
            block = self.distributed_strategy.distribute_layer(block, i)
            layers.append(block)
        self.layers = nn.ModuleList(layers)

        dec_norm = RMSNormGated(config.emb_dim, eps=config.norm_eps)

        self.dec_norm = self.distributed_strategy.distribute_module(
            dec_norm, final_layers=True
        )

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def reset_parameters(self):
        """_summary_"""
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
                or isinstance(m, MOEFeedForward)
            ):
                m.reset_parameters()

    def _clean_up_rot_emb_cache(
        self,
        cached_freqs: dict[Optional[torch.device], dict[int, torch.Tensor]],
        max_seq_len_cached: dict[Optional[torch.device], int],
    ):
        for dev in list(cached_freqs.keys()):
            for alp in list(cached_freqs[dev].keys()):
                if cached_freqs[dev][alp].device == torch.device("meta"):
                    del cached_freqs[dev][alp]
                    if len(cached_freqs[dev]) == 0:
                        del cached_freqs[dev]
                        del max_seq_len_cached[dev]

    def post_init(self):
        """_summary_"""
        # This function is called in `get_model` after the model is
        # fully initalized on the correct device

        self._clean_up_rot_emb_cache(
            self.rot_emb.cached_freqs,  # type: ignore
            self.rot_emb.max_seq_len_cached,  # type: ignore
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
        # x_in: batch_size x seq_len x emb_dim if input is already embedded, otherwise batch_size x seq_len
        # mask: batch_size x seq_len x seq_len
        # bias: nheads x seq_len x seq_len
        if past_key_value_states is None or len(past_key_value_states) == 0:
            past_key_value_states = [None for _ in range(len(self.layers))]

        if x_in.dim() == 2:  # input is not already embedded
            x_in = self.embedding(x_in)
        x_in = x_in * self.config.embedding_multiplier

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


class GptOss(nn.Module):
    def __init__(
        self,
        config: Optional[GptOssConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(GptOss, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = GptOssConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.base_model = GptOssHeadless(self.config, self.distributed_strategy)
        self.head = nn.Linear(
            self.config.emb_dim, self.config.src_vocab_size, bias=False
        )

    @classmethod
    def from_config(cls, config: GptOssConfig) -> "GptOss":
        return cls(config)

    def get_config(self) -> GptOssConfig:
        return self.config

    def reset_parameters(self):
        self.head.weight.data.normal_(
            0,
            1 / math.sqrt(math.sqrt(self.config.emb_dim * self.config.src_vocab_size)),
        )
        self.base_model.reset_parameters()

    def post_init(self):
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
            x,
            position_ids,
            past_key_value_states,
            use_cache,
            **attn_kwargs,
        )

        if only_last_token:
            output = output[:, -1, :]

        if output.dtype != self.head.weight.dtype:
            output_dtype = output.to(self.head.weight.dtype)
            preds = self.head(output_dtype)
        else:
            preds = self.head(output)

        preds = preds / self.config.logits_scaling

        if use_cache:
            return preds, cache
        else:
            return preds


_architecture_name = "gpt_oss"
_20b_config = GptOssConfig()


def _gpt_oss_factory_factory(config):
    def factory(**kwargs):
        return GptOss(config, **kwargs)

    return factory


models.register_model(_architecture_name, "20b", _gpt_oss_factory_factory(_20b_config))


# =============== Serialization ==================


serialization.register_adapter_step(
    _architecture_name,
    "swiglu_unfused_to_fused",
    serialization._mlp_glu_unfused_to_fused_adapter_step,
)


def _weight_fusion(
    input_sd: Mapping[str, Any], model_config: Optional[GptOssConfig] = None, **kwargs
) -> Mapping[str, Any]:
    has_fused_weights = True
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False

    new_sd = input_sd
    if has_fused_weights:
        for key in list(new_sd.keys()):
            if key not in new_sd:
                continue
            if "w1" in key and "_bias" not in key:
                fused_name = key.replace("w1", "w13")
                new_sd[fused_name] = torch.cat([new_sd[key], new_sd[key]], dim=1)
                del new_sd[key]

        new_sd = dict(serialization._attn_unfused_to_fused_step(new_sd))

    return new_sd


serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)


def _hf_gptq_gpt_oss_check(
    input_sd: Mapping[str, Any], model_config: Optional[GptOssConfig] = None, **kwargs
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
            "GPTQ HF GptOss checkpoints cannot be loaded into a model with fused weights"
        )

    return input_sd


serialization.register_adapter_step(
    _architecture_name, "hf_gptq_fusion_check", _hf_gptq_gpt_oss_check
)


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        (r"^lm_head.weight", "head.weight"),
        (r"^model.embed_tokens.weight", "base_model.embedding.weight"),
        (r"^model.layers", "base_model.layers"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp.experts.gate_up_proj_blocks", "ff_sub_layer.cond_ffn.w1"),
        (r"mlp.experts.down_proj_blocks", "ff_sub_layer.cond_ffn.w2"),
        (r"mlp.router", "ff_sub_layer.gate"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
        (r"^model.norm", "base_model.dec_norm"),
    ]
    gpt_oss_experts_specific = [
        (r"mlp.experts.gate_up_proj_blocks", "ff_sub_layer.cond_ffn.w1"),
        (r"mlp.experts.down_proj_blocks", "ff_sub_layer.cond_ffn.w2"),
        (r"mlp.experts.gate_up_proj_bias", "ff_sub_layer.cond_ffn.w1_bias"),
        (r"mlp.experts.down_proj_bias", "ff_sub_layer.cond_ffn.w2_bias"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        unpacked_tensors = None
        if re.search("gate_up_proj|down_proj", name) and "bias" not in name:
            if "scales" in name:
                continue
            elif "blocks" in name:
                # deal with packed weights
                blocks = input_sd[name]
                scales = input_sd[name.replace("blocks", "scales")]
                new_name = name.replace(".blocks", "")
                unpacked_tensors = _convert_moe_packed_tensors(
                    blocks, scales, dtype=torch.bfloat16
                )
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)

        if re.search("gate_up_proj|down_proj", new_name) and re.search(
            "base_model.layers", new_name
        ):
            for pattern, repl in gpt_oss_experts_specific:
                new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = unpacked_tensors if unpacked_tensors is not None else param
    return new_sd


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)


def _get_rope_params(linear_type: str) -> list[str]:
    if "gptq" in linear_type:
        return ["qweight", "scales", "qzeros", "bias"]
    # torch.nn.Linear
    return ["weight", "bias"]


def _hf_to_fms_rope(
    input_sd: Mapping[str, Any], model_config: Optional[GptOssConfig] = None, **kwargs
) -> Mapping[str, Any]:
    new_sd = {}

    if model_config:
        head_size = model_config.emb_dim // model_config.nheads
        linear_type = "torch_linear"
        if model_config.linear_config:
            linear_type = model_config.linear_config["linear_type"]
    else:
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


def _convert_moe_packed_tensors(
    blocks,
    scales,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    import math

    scales = scales.to(torch.int32) - 127

    assert blocks.shape[:-1] == scales.shape, (
        f"{blocks.shape=} does not match {scales.shape=}"
    )

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp

    out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
    out = out.to(torch.float8_e5m2).permute(0, 2, 1).contiguous()
    return out


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_rope", _hf_to_fms_rope
)

serialization.register_adapter(
    _architecture_name,
    "hf",
    ["hf_to_fms_names", "hf_to_fms_rope", "hf_gptq_fusion_check", "weight_fusion"],
)
