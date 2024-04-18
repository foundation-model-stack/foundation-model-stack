import warnings
from typing import List, Optional, Tuple, Union
from functools import partial

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaModel,
    LlamaForCausalLM,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaFlashAttention2,
)
from transformers.modeling_outputs import (
    MoeModelOutputWithPast,
    MoeCausalLMOutputWithPast,
)
from transformers.cache_utils import Cache, StaticCache, DynamicCache
from transformers.utils import logging

from .configuration_mixllama import MixLlamaConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MixLlamaConfig"

def load_balancing_loss_func(
        gate_logits: torch.Tensor,
        num_experts: torch.Tensor = None,
        top_k=2,
        layer_mode="sum") -> float:
   
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[-1].device
        layer_gate_logits = torch.stack(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
            )

    # at this point, layer_gate_logits has shape of (layer, bs*seq, num_exp)
    routing_weights = torch.nn.functional.softmax(layer_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_masks = torch.nn.functional.one_hot(selected_experts, num_experts).sum(-2).float()

    # https://arxiv.org/pdf/2403.07816.pdf Section 3.3
    routing_weights = routing_weights.mean(1)
    routing_masks = routing_masks.mean(1)
    loss_lb = num_experts * torch.sum(routing_weights * routing_masks, dim=-1)
    
    if layer_mode == "sum":
        loss_lb = loss_lb.sum()
    elif layer_mode == "average":
        loss_lb = loss_lb.mean()
    else:
        raise NotImplementedError
    
    return loss_lb

def make_moe_modules(moe_class_partial):

    class MoeModules(nn.Module):

        def __init__(self, config):

            super().__init__()
            self.num_experts = config.num_local_experts
            self.top_k = config.num_experts_per_tok
            self.always_on_idx = config.always_on_idx

            self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
            self.experts = nn.ModuleList([moe_class_partial() for _ in range(self.num_experts)])
            
            # storing purpose, not to disrupt original forward
            self.router_logits = None

        def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:

            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.router(hidden_states)
            self.router_logits = router_logits

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            if self.always_on_idx > -1:
                always_on_constant, _ = torch.topk(routing_weights, self.top_k, dim=-1)
                routing_weights[:,self.always_on_idx] += always_on_constant.detach().sum(-1)
                routing_weights /= routing_weights.sum(-1, keepdim=True)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )

            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

            for expert_idx in range(self.num_experts):
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])

                if top_x.shape[0] == 0:
                    continue
                top_x_list = top_x.tolist()
                idx_list = idx.tolist()

                current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(
                    current_state, *args, **kwargs
                ) * routing_weights[top_x_list, idx_list, None]

                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

            return final_hidden_states
        
    return MoeModules

# NOTE: DOES NOT SUPPORT EAGER MODE, SINCE EAGER MODE HF LLAMA ATTENTION ACCESS HEAD.weight !     
class MixLlamaAttention(LlamaFlashAttention2):

    def __init__(self, config: MixLlamaConfig, layer_idx:int):

        super().__init__(config, layer_idx)
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.always_on_idx = config.always_on_idx
        self.moe_query = config.moe_query
        self.moe_key = config.moe_key
        self.moe_value = config.moe_value

        # replacing
        if self.moe_query:
            delattr(self, "q_proj")
            LINEAR_CLS = make_moe_modules(partial(
                nn.Linear,
                in_features=self.hidden_size,
                out_features=self.num_heads * self.head_dim,
                bias=config.attention_bias))
            self.q_proj = LINEAR_CLS(config=config)
        if self.moe_key:
            delattr(self, "k_proj")
            LINEAR_CLS = make_moe_modules(partial(
                nn.Linear,
                in_features=self.hidden_size,
                out_features=self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias))
            self.k_proj = LINEAR_CLS(config=config)
        if self.moe_value:
            delattr(self, "v_proj")
            LINEAR_CLS = make_moe_modules(partial(
                nn.Linear,
                in_features=self.hidden_size,
                out_features=self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias))
            self.v_proj = LINEAR_CLS(config=config)
    
    @property
    def router_logits(self):
        router_logits = ()
        if self.moe_query:
            router_logits += (self.q_proj.router_logits,)
        if self.moe_key:
            router_logits += (self.k_proj.router_logits,)
        if self.moe_value:
            router_logits += (self.v_proj.router_logits,)
        return router_logits
    
class MixLlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(self, config: MixLlamaConfig, layer_idx: int):
        
        super().__init__(config=config, layer_idx=layer_idx)
        self.moe_mlp = config.moe_mlp
        self.moe_attention = \
            config.moe_query or config.moe_query or config.moe_value

        if self.moe_attention:
            delattr(self, "self_attn")
            self.self_attn = MixLlamaAttention(config, layer_idx=layer_idx)
       
        if config.moe_mlp:
            delattr(self, "mlp")
            MOE_MLP = make_moe_modules(partial(LlamaMLP, config=config))
            self.mlp = MOE_MLP(config)


    def forward(
        self,
        *args,
        output_router_logits: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
       
        outputs = super().forward(*args, **kwargs)
        
        if output_router_logits:
            if self.moe_attention:
                router_logits = self.self_attn.router_logits
            else:
                router_logits = ()

            if self.moe_mlp:
                router_logits += (self.mlp.router_logits,)

            outputs += (router_logits,)
            
            
        return outputs


class MixLlamaPreTrainedModel(LlamaPreTrainedModel):
    config_class = MixLlamaConfig
    _no_split_modules = ["MixLlamaDecoderLayer"]


class MixLlamaModel(MixLlamaPreTrainedModel, LlamaModel):

    def __init__(self, config: MixLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MixLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += layer_outputs[-1]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )
    
class MixLlamaForCausalLM(MixLlamaPreTrainedModel, LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = MixLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1], self.num_experts, self.num_experts_per_tok
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )