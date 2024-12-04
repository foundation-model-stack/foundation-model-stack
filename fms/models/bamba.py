import dataclasses
import math
from typing import List, Optional, Tuple, Mapping, Any

import torch
import torch.nn as nn

from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.modules.ssm import SSM, SSMCacheUnit
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig

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
    attn_layer_indices: set[int] =dataclasses.field(default_factory=lambda: set([]))
    ntk_scaling: bool = False
    max_expected_seq_len: int = 4096
    tie_heads: bool = False
    rope_theta: float = 10_000.0
    p_dropout: float = 0.0
    conv_kernel: int = 4
    state_size: int = 256
    hidden_grow_factor: float = 2.0
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
        self.layer_index = layer_index
        self.norm = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )

        if layer_index in config.attn_layer_indices:
            if self.config.kvheads == 0:
                kvheads = self.config.nheads
            else:
                kvheads = self.config.kvheads
                assert self.config.nheads % self.config.kvheads == 0

            self.attn_ssm = MultiHeadAttention(
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
            self.attn_ssm = SSM(
                self.config.nheads,
                self.config.emb_dim,
                self.config.state_size,
                self.config.conv_kernel,
                self.config.hidden_grow_factor,
                self.config.use_bias,
                self.config.use_conv_bias,
                self.config.activation_fn,
                self.config.norm_eps,
                self.config.n_groups,
                self.config.head_dim,
                self.config.chunk_size,
            )

        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=False,
            fused=True,
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
        mask=None,
        position_ids=None,
        past_key_value_state=None,
        use_cache=False,
        is_causal_mask=False,
        attn_algorithm=None,
    ):
        residual = x
        x = self.ln(x)

        x = self.attn_ssm(
            x,
            mask=mask,
            position_ids=position_ids,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            # everything after here is ignored in ssm
            attn_algorithm=attn_algorithm,
            is_self=True,
            is_causal_mask=is_causal_mask,
        )

        cache = None
        if use_cache:
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
            return x, cache
        else:
            return x

class BambaHeadless(nn.Module):

    def __init__(self, config: BambaConfig, distributed_strategy: DistributedStrategy):
        super(BambaHeadless, self).__init__()
        self.config = config
        self.distributed_strategy = distributed_strategy

        self.embedding = nn.Embedding(
            self.config.src_vocab_size,
            self.config.emb_dim
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

        self.attn_layer_ind = -1 if len(self.attn_layer_ind) == 0 else next(iter(self.config.attn_layer_indices))

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
        if self.attn_layer_ind != -1:
            if use_cache and past_key_value_states[self.attn_layer_ind] is not None:
                klen += past_key_value_states[self.attn_layer_ind][0].size(-2)

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

class Bamba(nn.Module):
    def __init__(self, config: Optional[BambaConfig] = None, distributed_strategy: DistributedStrategy = NoOpStrategy, **kwargs):
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
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[List[SSMCacheUnit | Tuple[torch.FloatTensor,]]] = None,
        use_cache: bool = False,
        only_last_token: bool = False,
        attn_algorithm: Optional[str] = None,
    ):
        output, cache = self.base_model(
            x, mask, position_ids, past_key_value_states, use_cache, attn_algorithm
        )

        if only_last_token:
            output = output[:, -1, :]
        preds = self.head(output)

        if use_cache:
            return preds, cache
        else:
            return preds
