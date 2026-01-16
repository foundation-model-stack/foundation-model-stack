import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fms.models.llama import LLaMA, LLaMAConfig
from fms.models.siglip_vision import SiglipVision, SiglipVisionConfig
from fms.utils.config import ModelConfig

from .connector import Idefics3Connector
from .pack import pack_image_embeddings


def _attention_mask_2d_to_3d(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a standard HF-style attention mask (1=token, 0=pad) into the 3D boolean
    mask expected by FMS SDPA attention (padding block-diagonal; causality handled separately).
    """
    is_pad = attention_mask == 0
    return is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)


def _make_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    return position_ids


_smolvlm_vision_config = SiglipVisionConfig(
    hidden_size=768,
    intermediate_size=3072,
    nlayers=12,
    nheads=12,
    num_channels=3,
    image_size=512,
    patch_size=16,
    hidden_act="gelu-tanh",
    layer_norm_eps=1e-6,
    attention_dropout=0.0,
    fused_weights=True,
)

_smolvlm_text_config = LLaMAConfig(
    src_vocab_size=49280,
    emb_dim=576,
    norm_eps=1e-5,
    nheads=9,
    kvheads=3,
    nlayers=30,
    pad_id=0,
    hidden_grow_factor=1536 / 576,
    multiple_of=1,
    activation_fn="swish",
    p_dropout=0.0,
    max_expected_seq_len=2048,
    attn_bias=False,
    mlp_bias=False,
    tie_heads=False,
    fused_weights=True,
)


@dataclass
class Idefics3Config(ModelConfig):
    # Defaults match HuggingFaceTB/SmolVLM-256M-Instruct
    vision_config: SiglipVisionConfig = field(
        default_factory=lambda: SiglipVisionConfig(**_smolvlm_vision_config.as_dict())
    )
    text_config: LLaMAConfig = field(
        default_factory=lambda: LLaMAConfig(**_smolvlm_text_config.as_dict())
    )

    image_token_id: int = 49190
    image_span_len: int = 64
    connector_scale: int = 4

    # Generation defaults (greedy by default in parity tests)
    max_new_tokens: int = 64


class Idefics3(nn.Module):
    """
    FMS-native Idefics3/SmolVLM-style VLM:
    - Vision tower: FMS SiglipVision (patch embeddings)
    - Connector: pixel-unshuffle + projection
    - Text: FMS LLaMA (with optional `inputs_embeds` path via LLaMAHeadless)
    """

    def __init__(self, config: Optional[Idefics3Config] = None, **kwargs: Any):
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = Idefics3Config()
        self.config = self.config.updated(**kwargs)

        self.vision_tower = SiglipVision(self.config.vision_config)
        self.text_model = LLaMA(self.config.text_config)

        self.connector = Idefics3Connector(
            vision_hidden=self.config.vision_config.hidden_size,
            text_hidden=self.config.text_config.emb_dim,
            scale=self.config.connector_scale,
        )

    def get_config(self) -> Idefics3Config:
        return self.config

    @classmethod
    def from_config(cls, config: Idefics3Config) -> "Idefics3":
        return cls(config)

    def post_init(self):
        if getattr(self.vision_tower, "post_init", None) and callable(
            self.vision_tower.post_init
        ):
            self.vision_tower.post_init()
        if getattr(self.text_model, "post_init", None) and callable(
            self.text_model.post_init
        ):
            self.text_model.post_init()

    def reset_parameters(self):
        # SiglipVisionHeadless.reset_parameters() recurses via `self.modules()`;
        # initialize the vision tower explicitly to avoid calling that helper.
        if getattr(self.vision_tower, "head", None) is not None and hasattr(
            self.vision_tower.head, "reset_parameters"
        ):
            self.vision_tower.head.reset_parameters()
        if getattr(self.vision_tower, "base_model", None) is not None:
            base = self.vision_tower.base_model
            if hasattr(base, "embeddings") and hasattr(
                base.embeddings, "reset_parameters"
            ):
                base.embeddings.reset_parameters()
            if hasattr(base, "encoder") and hasattr(base.encoder, "reset_parameters"):
                base.encoder.reset_parameters()
            if hasattr(base, "post_layernorm") and hasattr(
                base.post_layernorm, "reset_parameters"
            ):
                base.post_layernorm.reset_parameters()
        if getattr(self.text_model, "reset_parameters", None) and callable(
            self.text_model.reset_parameters
        ):
            self.text_model.reset_parameters()
        nn.init.xavier_uniform_(self.connector.proj.weight)

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, 3, H, W) or (B, N, 3, H, W)
        is_5d = images.dim() == 5
        if is_5d:
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)

        vision_last_hidden, _ = self.vision_tower(images)
        # FMS SiglipVision has no CLS token; expect (B*N, 1024, hidden)
        if is_5d:
            vision_last_hidden = vision_last_hidden.view(
                B, N, *vision_last_hidden.shape[1:]
            )
        return vision_last_hidden

    def _connector_tokens(self, patch_feats: torch.Tensor) -> torch.Tensor:
        # patch_feats: (B, 1024, Dv) or (B, N, 1024, Dv)
        is_4d = patch_feats.dim() == 4
        if is_4d:
            B, N, L, Dv = patch_feats.shape
            patch_feats = patch_feats.view(B * N, L, Dv)

        grid = int(math.isqrt(patch_feats.shape[1]))
        img_tokens = self.connector(patch_feats, H=grid, W=grid)  # (B*N, 64, Dt)

        if is_4d:
            img_tokens = img_tokens.view(B, N, img_tokens.shape[1], img_tokens.shape[2])
        return img_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value_states: Optional[list] = None,
        **unused,
    ) -> dict[str, torch.Tensor]:
        if images is None:
            images = pixel_values
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Text embeddings
        text_embeds = self.text_model.base_model.embedding(input_ids)

        if images is not None:
            patch_feats = self._encode_images(images)
            img_tokens = self._connector_tokens(patch_feats)
            img_tokens = img_tokens.to(dtype=text_embeds.dtype)
            packed = pack_image_embeddings(
                input_ids=input_ids,
                inputs_embeds=text_embeds,
                image_features=img_tokens,
                image_token_id=self.config.image_token_id,
                expected_L=self.config.image_span_len,
            )
        else:
            packed = text_embeds

        mask_3d = _attention_mask_2d_to_3d(attention_mask)
        position_ids = _make_position_ids(attention_mask)

        hidden, present = self.text_model.base_model(
            None,
            inputs_embeds=packed,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            use_cache=use_cache,
            mask=mask_3d,
            is_causal_mask=True,
        )

        logits = self.text_model.head(hidden)

        loss = None
        if labels is not None:
            labels = labels.clone()
            labels = labels.masked_fill(input_ids == self.config.image_token_id, -100)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        out: dict[str, torch.Tensor] = {"logits": logits}
        if loss is not None:
            out["loss"] = loss
        if use_cache:
            # keep naming consistent with llama internals
            out["past_key_value_states"] = present  # type: ignore[assignment]
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 20,
        eos_token_id: Optional[int] = None,
        **unused,
    ) -> torch.Tensor:
        self.eval()
        if images is None:
            images = pixel_values
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Prefill (packed embeds)
        text_embeds = self.text_model.base_model.embedding(input_ids)
        if images is not None:
            patch_feats = self._encode_images(images)
            img_tokens = self._connector_tokens(patch_feats).to(dtype=text_embeds.dtype)
            packed = pack_image_embeddings(
                input_ids=input_ids,
                inputs_embeds=text_embeds,
                image_features=img_tokens,
                image_token_id=self.config.image_token_id,
                expected_L=self.config.image_span_len,
            )
        else:
            packed = text_embeds

        mask_3d = _attention_mask_2d_to_3d(attention_mask)
        position_ids = _make_position_ids(attention_mask)

        hidden, past = self.text_model.base_model(
            None,
            inputs_embeds=packed,
            position_ids=position_ids,
            past_key_value_states=None,
            use_cache=True,
            mask=mask_3d,
            is_causal_mask=True,
        )

        generated = input_ids
        last_logits = self.text_model.head(hidden[:, -1:, :]).squeeze(1)
        next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

        for _ in range(max_new_tokens):
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None:
                if torch.all(next_token.squeeze(1) == eos_token_id):
                    break

            # decode step: embed only the new token
            next_embeds = self.text_model.base_model.embedding(next_token)
            # no padding during generation; mask can be omitted
            pos_next = position_ids[:, -1:] + 1

            hidden, past = self.text_model.base_model(
                None,
                inputs_embeds=next_embeds,
                position_ids=pos_next,
                past_key_value_states=past,
                use_cache=True,
            )

            last_logits = self.text_model.head(hidden[:, -1:, :]).squeeze(1)
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            position_ids = torch.cat([position_ids, pos_next], dim=1)

        return generated
