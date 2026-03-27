import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from fms.models.llama import LLaMA, LLaMAConfig
from fms.models.siglip_vision import SiglipVision, SiglipVisionConfig
from fms.utils.config import ModelConfig
from fms.utils.generation import generate as fms_generate

from .connector import Idefics3Connector
from .pack import pack_image_embeddings


def _left_pad_to_max_len(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize a batch of (possibly right-padded) sequences to left-padded form.

    This avoids generation bugs where cached decoding starts from padded positions.
    Assumes `attention_mask` is 1 for tokens and 0 for pads, with no "holes".
    """
    B, T = input_ids.shape
    new_input_ids = input_ids.new_full((B, T), int(pad_id))
    new_attention_mask = attention_mask.new_zeros((B, T))
    for b in range(B):
        m = attention_mask[b].to(dtype=torch.bool)
        # "Holes" means the mask changes more than once (e.g., 1..0..1 or 0..1..0).
        # Left-padding (0..0..1..1) and right-padding (1..1..0..0) are both allowed.
        transitions = (m[1:] != m[:-1]).sum()
        if int(transitions.item()) > 1:
            raise ValueError("attention_mask must be contiguous (no 0/1 holes)")
        tokens = input_ids[b][m]
        L = int(tokens.numel())
        if L == 0:
            continue
        new_input_ids[b, T - L : T] = tokens
        new_attention_mask[b, T - L : T] = 1
    return new_input_ids, new_attention_mask


def _attention_mask_2d_to_3d(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a standard HF-style attention mask (1=token, 0=pad) into the 3D boolean
    mask expected by FMS SDPA attention.

    FMS passes boolean masks through to `torch.nn.functional.scaled_dot_product_attention`,
    where boolean `attn_mask` is interpreted as an "allowed" mask (True = can attend).

    We use a block-diagonal padding mask (token<->token allowed; pad<->pad allowed; cross blocked).
    Allowing pad<->pad avoids all-False rows for padded query positions, which can otherwise
    produce NaNs in SDPA. Causality is handled separately via `is_causal_mask=True`.
    """
    is_pad = attention_mask == 0
    return is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)


def _attention_mask_2d_to_sdpa_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert an HF-style attention mask (1=token, 0=pad) into a 3D additive SDPA mask.

    - Allowed positions are 0.0
    - Disallowed positions are -inf

    We intentionally allow pad<->pad attention (i.e., padded query positions can attend to padded
    key positions). This avoids all-disallowed rows for padded queries, which can produce NaNs
    in SDPA. Causality is handled by applying a lower-triangular mask.

    The mask is made causal (lower triangular), matching the behavior of
    `fms.utils.generation.pad_input_ids` and ensuring compatibility with SDPA's
    `update_attn_kwargs` decoding updates.
    """
    is_pad = attention_mask == 0
    mask = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)
    mask = mask.tril()
    return torch.where(mask, 0.0, -torch.inf)


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
    use_navit_position_buckets=True,
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
        def _reset_if_present(module):
            reset_fn = getattr(module, "reset_parameters", None)
            if callable(reset_fn):
                reset_fn()

        _reset_if_present(getattr(self.vision_tower, "head", None))

        base = getattr(self.vision_tower, "base_model", None)
        if base is not None:
            _reset_if_present(getattr(base, "embeddings", None))
            _reset_if_present(getattr(base, "encoder", None))
            _reset_if_present(getattr(base, "post_layernorm", None))

        _reset_if_present(self.text_model)
        nn.init.xavier_uniform_(self.connector.proj.weight)

    def _encode_images(
        self,
        images: torch.Tensor,
        pixel_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # images: (B, 3, H, W) or (B, N, 3, H, W)
        is_5d = images.dim() == 5
        if is_5d:
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
            if pixel_attention_mask is not None:
                if pixel_attention_mask.dim() == 4:
                    hm, wm = pixel_attention_mask.shape[2:]
                    if pixel_attention_mask.shape[1] == N:
                        # (B, N, Hm, Wm) -> (B*N, Hm, Wm)
                        pixel_attention_mask = pixel_attention_mask.contiguous().view(
                            B * N, hm, wm
                        )
                    elif pixel_attention_mask.shape[1] == 1:
                        # (B, 1, Hm, Wm) -> broadcast to (B, N, Hm, Wm) then flatten.
                        pixel_attention_mask = (
                            pixel_attention_mask.expand(B, N, hm, wm)
                            .contiguous()
                            .view(B * N, hm, wm)
                        )
                    else:
                        raise ValueError(
                            "For 5D images (B, N, C, H, W), 4D pixel_attention_mask must be "
                            f"(B, N, Hm, Wm) or (B, 1, Hm, Wm); got {tuple(pixel_attention_mask.shape)}"
                        )
                elif pixel_attention_mask.dim() == 3:
                    if pixel_attention_mask.shape[0] == B:
                        # (B, Hm, Wm) -> apply to all N images
                        pixel_attention_mask = (
                            pixel_attention_mask.unsqueeze(1)
                            .expand(B, N, *pixel_attention_mask.shape[1:])
                            .contiguous()
                            .view(B * N, *pixel_attention_mask.shape[1:])
                        )
                    elif pixel_attention_mask.shape[0] == B * N:
                        # Already flattened to match (B*N, C, H, W) images.
                        pixel_attention_mask = pixel_attention_mask.contiguous()
                    else:
                        raise ValueError(
                            "For 5D images (B, N, C, H, W), 3D pixel_attention_mask first dim must be "
                            f"B or B*N; got shape {tuple(pixel_attention_mask.shape)} with B={B}, N={N}"
                        )

        patch_size = self.config.vision_config.patch_size

        # Validate that image dimensions are divisible by patch_size
        if images.shape[2] % patch_size != 0 or images.shape[3] % patch_size != 0:
            raise ValueError(
                f"Image dimensions ({images.shape[2]}, {images.shape[3]}) must be divisible by patch_size ({patch_size})"
            )
        patch_attention_mask: Optional[torch.Tensor] = None
        if pixel_attention_mask is not None:
            pam = pixel_attention_mask
            if pam.dim() == 4 and pam.shape[1] == 1:
                pam = pam.squeeze(1)
            if pam.dim() != 3:
                raise ValueError(
                    "pixel_attention_mask must be (B, Hm, Wm) or (B, 1, Hm, Wm); "
                    f"got {tuple(pixel_attention_mask.shape)}"
                )

            pam = pam.to(device=images.device).to(dtype=torch.bool)
            hm, wm = pam.shape[-2:]
            ph, pw = images.shape[2] // patch_size, images.shape[3] // patch_size
            if (hm, wm) == (ph, pw):
                patch_attention_mask = pam
            elif (hm, wm) == (images.shape[2], images.shape[3]):
                # Pixel-level mask -> patch-level by pooling within each patch.
                patch_attention_mask = (
                    pam.view(pam.shape[0], ph, patch_size, pw, patch_size)
                    .any(dim=-1)
                    .any(dim=-2)
                )
            else:
                raise ValueError(
                    "pixel_attention_mask shape does not match image or patch grid: "
                    f"mask.shape={tuple(pam.shape)}, images.shape={tuple(images.shape)}, patch_size={patch_size}"
                )

        vision_last_hidden, _ = self.vision_tower(
            images, patch_attention_mask=patch_attention_mask
        )
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
        position_ids: Optional[torch.Tensor] = None,
        past_key_value_states: Optional[list] = None,
        use_cache: bool = False,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        **attn_kwargs,
    ):
        if images is None:
            images = pixel_values

        token_ids: Optional[torch.Tensor]
        if inputs is not None:
            token_ids = input_ids if input_ids.dim() == 2 else None
            text_embeds = inputs
        elif input_ids.dim() == 2:
            token_ids = input_ids
            text_embeds = self.text_model.base_model.embedding(input_ids)
        elif input_ids.dim() == 3:
            token_ids = None
            text_embeds = input_ids
        else:
            raise ValueError(
                f"Idefics3.forward expects input_ids to be token ids (B,T) or embeddings (B,T,D); got {tuple(input_ids.shape)}"
            )

        if token_ids is not None and images is not None:
            patch_feats = self._encode_images(
                images, pixel_attention_mask=pixel_attention_mask
            )
            img_tokens = self._connector_tokens(patch_feats)
            img_tokens = img_tokens.to(
                device=text_embeds.device, dtype=text_embeds.dtype
            )
            packed = pack_image_embeddings(
                input_ids=token_ids,
                inputs_embeds=text_embeds,
                image_features=img_tokens,
                image_token_id=self.config.image_token_id,
                expected_L=self.config.image_span_len,
            )
        else:
            packed = text_embeds

        if attention_mask is None:
            if token_ids is not None:
                attention_mask = torch.ones_like(token_ids)
            else:
                # If we're running from precomputed embeddings, assume no padding unless provided.
                attention_mask = torch.ones(
                    packed.shape[0],
                    packed.shape[1],
                    device=packed.device,
                    dtype=torch.long,
                )

        if position_ids is None:
            position_ids = _make_position_ids(attention_mask)

        if "mask" not in attn_kwargs:
            attn_kwargs["mask"] = _attention_mask_2d_to_sdpa_mask(attention_mask)

        hidden, present = self.text_model.base_model(
            None,
            inputs_embeds=packed,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            use_cache=use_cache,
            **attn_kwargs,
        )

        logits = self.text_model.head(hidden)
        if use_cache:
            return logits, present
        return logits

    def prepare_inputs_for_generation(self, iteration: int, input_ids, kwargs):
        """
        Hook for `fms.utils.generation.generate` to build multimodal embeddings.

        This follows the LlavaNext pattern:
        - Prefill (iteration == 0): encode images once and pack into the text embeddings.
        - Decode (iteration > 0 with use_cache=True): embed only new tokens; do not re-encode images.
        """
        if kwargs.get("use_cache") and iteration > 0:
            # Cached decoding stage: no image work; embed only the new token(s).
            embeds = self.text_model.base_model.embedding(input_ids)
            kwargs.pop("pixel_values", None)
            kwargs.pop("pixel_attention_mask", None)
            kwargs.pop("images", None)
            return embeds, kwargs

        images = kwargs.get("images")
        if images is None:
            images = kwargs.get("pixel_values")
        pixel_attention_mask = kwargs.get("pixel_attention_mask")

        # No image data to pack.
        if images is None or images.numel() == 0:
            inputs = kwargs.get("inputs")
            if inputs is not None:
                return inputs, kwargs
            return input_ids, kwargs

        inputs = kwargs.get("inputs")
        if inputs is None:
            inputs = self.text_model.base_model.embedding(input_ids)

        patch_feats = self._encode_images(
            images, pixel_attention_mask=pixel_attention_mask
        )
        img_tokens = self._connector_tokens(patch_feats)
        img_tokens = img_tokens.to(device=inputs.device, dtype=inputs.dtype)

        packed = pack_image_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs,
            image_features=img_tokens,
            image_token_id=self.config.image_token_id,
            expected_L=self.config.image_span_len,
        )

        # Drop image tensors so we don't carry them through cached decoding.
        kwargs.pop("pixel_values", None)
        kwargs.pop("pixel_attention_mask", None)
        kwargs.pop("images", None)
        return packed, kwargs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
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

        # Normalize right-padded batches to left-padding for correct cached decoding behavior.
        input_ids, attention_mask = _left_pad_to_max_len(
            input_ids, attention_mask, pad_id=self.config.text_config.pad_id
        )

        extra_kwargs: dict[str, Any] = {
            "mask": _attention_mask_2d_to_sdpa_mask(attention_mask),
            "position_ids": _make_position_ids(attention_mask),
        }
        if images is not None:
            extra_kwargs["pixel_values"] = images
            if pixel_attention_mask is not None:
                extra_kwargs["pixel_attention_mask"] = pixel_attention_mask

        return fms_generate(
            model=self,
            input_ids=input_ids,
            max_seq_len=self.config.text_config.max_expected_seq_len,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos_token_id,
            prepare_model_inputs_hook=self.prepare_inputs_for_generation,
            extra_kwargs=extra_kwargs,
        )
