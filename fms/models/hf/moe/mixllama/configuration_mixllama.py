from transformers import LlamaConfig
from transformers.utils import logging

import torch

logger = logging.get_logger(__name__)

class MixLlamaConfig(LlamaConfig):

    model_type = "mixllama"

    def __init__(
        self,
        num_experts_per_tok=2,
        num_local_experts=4,
        output_router_logits=False,
        router_aux_loss_coef=0.01,
        moe_query_head=False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.moe_query_head = moe_query_head
        if moe_query_head:
            self._attn_implementation = "flash_attention_2"
            assert self.torch_dtype in [torch.float16, torch.bfloat16]