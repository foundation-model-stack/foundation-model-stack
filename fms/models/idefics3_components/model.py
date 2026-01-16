"""
Idefics3-style multimodal model for Foundation Model Stack.

End-to-end model combining:
  - SigLIP vision encoder (patch embeddings)
  - Idefics3Connector (pixel-unshuffle + projection)
  - pack_image_embeddings (inject vision features into text sequence)
  - Text LLM (decoder-only for generation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import torch
import torch.nn as nn

from fms.utils.config import ModelConfig

from .connector import Idefics3Connector
from .pack import pack_image_embeddings


class HiddenAlign(nn.Module):
    """
    Optional projection layer to align connector output dimension
    to text model hidden dimension if they differ.
    """

    def __init__(self, src_dim: int, dst_dim: int):
        super().__init__()
        self.need_projection = src_dim != dst_dim
        self.proj = (
            nn.Linear(src_dim, dst_dim, bias=False)
            if self.need_projection
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T_img, src_dim) -> (B, T_img, dst_dim)"""
        return self.proj(x)


@dataclass
class Idefics3Config(ModelConfig):
    """Configuration for Idefics3FMSModel"""

    # Image token configuration
    image_token_id: int = 32000
    image_span_len: int = 64  # After connector: 32x32 patches -> 8x8 grid = 64 tokens

    # Vision encoder config (SigLIP defaults)
    vision_hidden: int = 768  # SigLIP patch embedding dimension
    image_size: int = 512
    patch_size: int = 16
    num_patches_per_side: int = 32  # image_size / patch_size = 512/16 = 32

    # Connector config
    connector_out_dim: int = 576  # SmolVLM-256M text hidden size
    connector_scale: int = 4

    # Text model config
    text_hidden_size: int = 576  # Must match or be projected to connector_out_dim

    # Model paths (for HF adapters)
    vision_name_or_path: str = "google/siglip-base-patch16-512"
    text_name_or_path: str = "HuggingFaceTB/SmolVLM-256M-Instruct"

    # Generation defaults
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True


class Idefics3FMSModel(nn.Module):
    """
    End-to-end Idefics3-style vision-language model.

    Workflow:
      1. Tokenize prompt with image placeholder tokens
      2. Run vision encoder on images -> patch features
      3. Apply connector to compress patches -> image tokens
      4. Pack image tokens into text embedding sequence
      5. Generate text with LLM

    Args:
        config: Idefics3Config with model dimensions and paths
        tokenizer: HuggingFace tokenizer with <image> token
        vision_encoder: Vision model returning (B, num_patches, vision_hidden)
        text_backbone: Decoder-only LLM with embed_tokens and generate methods
        device: Target device (cpu/cuda)
    """

    def __init__(
        self,
        config: Optional[Idefics3Config] = None,
        tokenizer=None,
        vision_encoder: Optional[nn.Module] = None,
        text_backbone: Optional[nn.Module] = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        if config is not None:
            self.cfg = config.updated(**kwargs)
        else:
            self.cfg = Idefics3Config(**kwargs)

        if vision_encoder is None:
            raise ValueError(
                "Idefics3FMSModel requires `vision_encoder` to be provided"
            )
        if text_backbone is None:
            raise ValueError("Idefics3FMSModel requires `text_backbone` to be provided")

        self.tokenizer = tokenizer
        self.vision_encoder = vision_encoder

        # Vision-to-text connector
        self.connector = Idefics3Connector(
            vision_hidden=self.cfg.vision_hidden,
            text_hidden=self.cfg.connector_out_dim,
            scale=self.cfg.connector_scale,
        )

        # Optional alignment if text hidden != connector output
        self.align = HiddenAlign(self.cfg.connector_out_dim, self.cfg.text_hidden_size)

        self.text_backbone = text_backbone
        self.device = torch.device(device)

        def _move_if_needed(module: nn.Module) -> nn.Module:
            try:
                first_param = next(module.parameters())
            except StopIteration:
                first_param = None
            if first_param is None or first_param.device != self.device:
                return module.to(self.device)
            return module

        # Move internal components to device; external backbones are moved only if needed.
        self.vision_encoder = _move_if_needed(self.vision_encoder)
        self.text_backbone = _move_if_needed(self.text_backbone)
        self.connector = self.connector.to(self.device)
        self.align = self.align.to(self.device)

        # Resolve image_token_id from tokenizer if possible
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            token_id = self.tokenizer.convert_tokens_to_ids("<image>")
            if isinstance(token_id, int):
                self.cfg.image_token_id = token_id

    def get_config(self) -> Idefics3Config:
        return self.cfg

    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Extract token embeddings from text backbone.
        Handles different model structures (embed_tokens, transformer.wte, etc.)
        """
        text_backbone = cast(Any, self.text_backbone)

        embed_tokens = getattr(text_backbone, "embed_tokens", None)
        if callable(embed_tokens):
            return cast(torch.Tensor, embed_tokens(input_ids))

        get_input_embeddings = getattr(text_backbone, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            emb_module = get_input_embeddings()
            return cast(torch.Tensor, emb_module(input_ids))

        transformer = getattr(text_backbone, "transformer", None)
        wte = getattr(transformer, "wte", None) if transformer is not None else None
        if callable(wte):
            # GPT-2 style
            return cast(torch.Tensor, wte(input_ids))

        raise AttributeError(
            "Text backbone must have embed_tokens, get_input_embeddings, or transformer.wte"
        )

    def _process_vision(self, images: torch.Tensor) -> torch.Tensor:
        """
        Process images through vision encoder and handle output formats/CLS token.

        Args:
            images: (B, 3, H, W) or (B, N, 3, H, W) input images

        Returns:
            patch_feats: (B, num_patches, hidden_dim) or (B, N, num_patches, hidden_dim)
        """
        # Handle 5D input (B, N, C, H, W)
        is_5d = images.dim() == 5
        if is_5d:
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)

        vision_output = self.vision_encoder(images.to(self.device))

        # Handle different vision encoder output formats
        if hasattr(vision_output, "last_hidden_state"):
            patch_feats = vision_output.last_hidden_state
        elif isinstance(vision_output, tuple):
            patch_feats = vision_output[0]
        else:
            patch_feats = vision_output

        # Handle CLS token if present (common in ViT)
        expected_patches = self.cfg.num_patches_per_side**2
        if patch_feats.shape[1] == expected_patches + 1:
            patch_feats = patch_feats[:, 1:, :]  # Drop CLS
        elif patch_feats.shape[1] != expected_patches:
            raise ValueError(
                f"Unexpected num tokens: {patch_feats.shape[1]} "
                f"(expected {expected_patches} or {expected_patches + 1} with CLS)"
            )

        # Restore 5D structure if needed
        if is_5d:
            # (B*N, L, D) -> (B, N, L, D)
            patch_feats = patch_feats.view(
                B, N, patch_feats.shape[1], patch_feats.shape[2]
            )

        return patch_feats

    @torch.no_grad()
    def prepare_inputs(
        self,
        prompts: Optional[List[str]] = None,
        images: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for generation by tokenizing text and packing vision features.

        Args:
            prompts: List of text prompts with <image> placeholders
            images: Image tensor (B, 3, H, W), preprocessed for vision encoder
            image_features: Precomputed patch features (B, 1024, 768)
            input_ids: Optional pre-tokenized input_ids (B, T). If provided, prompts are ignored.
            attention_mask: Optional attention mask.

        Returns:
            Dict with input_ids, attention_mask, and inputs_embeds
        """
        if input_ids is None and prompts is None:
            raise ValueError("Must provide either prompts or input_ids")

        if input_ids is not None:
            B = int(input_ids.shape[0])
        else:
            assert prompts is not None
            B = len(prompts)

        # Validate input: exactly one of images or image_features must be provided
        if images is None and image_features is None:
            raise ValueError("Must provide either images or image_features")
        if images is not None and image_features is not None:
            raise ValueError(
                "Cannot provide both images and image_features, choose one"
            )

        if input_ids is None:
            assert prompts is not None
            # Expand <image> tags in prompts to match image_span_len
            expanded_prompts = []
            for p in prompts:
                if "<image>" in p:
                    replacement = "<image>" * self.cfg.image_span_len
                    p = p.replace("<image>", replacement)
                expanded_prompts.append(p)
            prompts = expanded_prompts

            # Tokenize text
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"].to(self.device)  # (B, T)
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(
                self.device
            )
        else:
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids).to(self.device)

        # Vision path: images -> patches -> compressed tokens
        if images is not None:
            images = images.to(self.device)
            if images.shape[0] != B:
                raise ValueError(
                    f"Batch size mismatch: {B} prompts vs {images.shape[0]} images"
                )

            # Validate image shape (B, 3, H, W) or (B, N, 3, H, W)
            if images.dim() == 4:
                if images.shape[1] != 3:
                    raise ValueError(
                        f"Image channel mismatch: expected 3 (RGB), got {images.shape[1]}"
                    )
                if (
                    images.shape[2] != self.cfg.image_size
                    or images.shape[3] != self.cfg.image_size
                ):
                    raise ValueError(
                        f"Image shape mismatch: expected (B, 3, {self.cfg.image_size}, {self.cfg.image_size}), "
                        f"got {images.shape}"
                    )
            elif images.dim() == 5:
                if images.shape[2] != 3:
                    raise ValueError(
                        f"Image channel mismatch: expected 3 (RGB), got {images.shape[2]}"
                    )
                if (
                    images.shape[3] != self.cfg.image_size
                    or images.shape[4] != self.cfg.image_size
                ):
                    raise ValueError(
                        f"Image shape mismatch: expected (B, N, 3, {self.cfg.image_size}, {self.cfg.image_size}), "
                        f"got {images.shape}"
                    )
            else:
                raise ValueError(f"Images must be 4D or 5D, got {images.dim()}D")

            patch_feats = self._process_vision(images)

        else:
            # Use precomputed features
            assert image_features is not None
            patch_feats = image_features.to(self.device).contiguous()
            # Handle 4D features (B, N, L, D)
            if patch_feats.dim() not in [3, 4]:
                raise ValueError(
                    f"image_features must be 3D or 4D, got {patch_feats.dim()}D"
                )
            if patch_feats.shape[0] != B:
                raise ValueError(
                    f"Batch size mismatch: {B} prompts vs {patch_feats.shape[0]} features"
                )

            expected_patches = self.cfg.num_patches_per_side**2
            # Check last dim and second to last dim
            if patch_feats.shape[-2] != expected_patches:
                raise ValueError(
                    f"Expected {expected_patches} patches, got {patch_feats.shape[-2]}"
                )
            if patch_feats.shape[-1] != self.cfg.vision_hidden:
                raise ValueError(
                    f"Expected {self.cfg.vision_hidden} dims, got {patch_feats.shape[-1]}"
                )

        # Apply connector with correct H, W
        # Flatten patches for connector if needed
        is_multi_patch = patch_feats.dim() == 4
        if is_multi_patch:
            B_p, N_p, L_p, D_p = patch_feats.shape
            patch_feats = patch_feats.view(B_p * N_p, L_p, D_p)

        H = W = self.cfg.num_patches_per_side
        img_tokens = self.connector(patch_feats, H=H, W=W)  # (B, 64, 576)

        # Restore multi-patch structure
        if is_multi_patch:
            img_tokens = img_tokens.view(
                B_p, N_p, img_tokens.shape[1], img_tokens.shape[2]
            )
            # Do NOT flatten for packing if we want to support multiple spans
            # pack_image_embeddings supports (B, N, L, D)

        img_tokens = self.align(
            img_tokens
        )  # (B, 64, text_hidden) or (B, N, 64, text_hidden)

        # Get text embeddings and pack vision features
        txt_embs = self.get_text_embeddings(input_ids)  # (B, T, text_hidden)

        # Match dtypes (text backbone might be bf16/fp16)
        img_tokens = img_tokens.to(dtype=txt_embs.dtype)

        # Pack vision tokens into text sequence
        # If img_tokens is (B, N, L, D), expected_L should be L
        expected_L = int(img_tokens.shape[-2])

        packed_embs = pack_image_embeddings(
            input_ids=input_ids,
            inputs_embeds=txt_embs,
            image_features=img_tokens,
            image_token_id=self.cfg.image_token_id,
            expected_L=expected_L,
        )  # (B, T, text_hidden)

        assert attention_mask is not None
        return {
            "inputs_embeds": packed_embs,
            "attention_mask": attention_mask,
            "input_ids": input_ids,  # Return input_ids for reference
        }

    @torch.no_grad()
    def generate(
        self,
        prompts: Optional[List[str]] = None,
        images: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate text from prompts and images and return decoded strings.
        """
        model_inputs = self.prepare_inputs(
            prompts=prompts,
            images=images,
            image_features=image_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Merge config defaults with user overrides
        gen_config = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "do_sample": self.cfg.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs,
        }

        text_backbone = cast(Any, self.text_backbone)

        # Use generate_from_embeds if available (CausalLMWrapper)
        # This ensures generation starts from packed embeddings
        generate_from_embeds = getattr(text_backbone, "generate_from_embeds", None)
        if callable(generate_from_embeds):
            out_ids = generate_from_embeds(
                inputs_embeds=model_inputs["inputs_embeds"],
                attention_mask=model_inputs["attention_mask"],
                **gen_config,
            )
        # Try inputs_embeds-based generation if supported
        elif callable(getattr(text_backbone, "generate", None)):
            try:
                # Modern HF models support inputs_embeds in generate
                out_ids = text_backbone.generate(
                    inputs_embeds=model_inputs["inputs_embeds"],
                    attention_mask=model_inputs["attention_mask"],
                    **gen_config,
                )
            except TypeError as e:
                raise TypeError(
                    f"Text backbone does not support inputs_embeds generation: {e}. "
                    "This is required for multimodal generation."
                )
        else:
            raise AttributeError("Text backbone does not support generation")

        # Decode outputs
        texts = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        return texts

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training (not generation).

        Args:
            input_ids: Token IDs (B, T)
            images: Image tensors (B, 3, H, W)
            attention_mask: Attention mask (B, T)
            labels: Target labels for loss computation (B, T)

        Returns:
            Dict with loss and logits
        """
        B = input_ids.shape[0]
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        if images is not None:
            if images.shape[0] != B:
                raise ValueError(
                    f"Batch size mismatch: {B} prompts vs {images.shape[0]} images"
                )

            # Validate image shape (B, 3, H, W) or (B, N, 3, H, W)
            if images.dim() == 4:
                if images.shape[1] != 3:
                    raise ValueError(
                        f"Image channel mismatch: expected 3 (RGB), got {images.shape[1]}"
                    )
                if (
                    images.shape[2] != self.cfg.image_size
                    or images.shape[3] != self.cfg.image_size
                ):
                    raise ValueError(
                        f"Image shape mismatch: expected (B, 3, {self.cfg.image_size}, {self.cfg.image_size}), "
                        f"got {images.shape}"
                    )
            elif images.dim() == 5:
                if images.shape[2] != 3:
                    raise ValueError(
                        f"Image channel mismatch: expected 3 (RGB), got {images.shape[2]}"
                    )
                if (
                    images.shape[3] != self.cfg.image_size
                    or images.shape[4] != self.cfg.image_size
                ):
                    raise ValueError(
                        f"Image shape mismatch: expected (B, N, 3, {self.cfg.image_size}, {self.cfg.image_size}), "
                        f"got {images.shape}"
                    )
            else:
                raise ValueError(f"Images must be 4D or 5D, got {images.dim()}D")

            # Process vision
            patch_feats = self._process_vision(images)

            # Flatten patches for connector if needed
            is_multi_patch = patch_feats.dim() == 4
            if is_multi_patch:
                B_p, N_p, L_p, D_p = patch_feats.shape
                patch_feats = patch_feats.view(B_p * N_p, L_p, D_p)

            H = W = self.cfg.num_patches_per_side
            img_tokens = self.connector(patch_feats, H=H, W=W)

            # Restore multi-patch structure
            if is_multi_patch:
                img_tokens = img_tokens.view(
                    B_p, N_p, img_tokens.shape[1], img_tokens.shape[2]
                )

            img_tokens = self.align(img_tokens)

            txt_embs = self.get_text_embeddings(input_ids)

            # Match dtypes
            img_tokens = img_tokens.to(dtype=txt_embs.dtype)

            inputs_embeds = pack_image_embeddings(
                input_ids=input_ids,
                inputs_embeds=txt_embs,
                image_features=img_tokens,
                image_token_id=self.cfg.image_token_id,
                expected_L=self.cfg.image_span_len,
            )

            # Mask labels for image tokens if provided
            if labels is not None:
                labels = labels.clone()
                # Create a mask for image tokens
                image_mask = input_ids == self.cfg.image_token_id
                # Set labels to -100 (ignore index) where image tokens are
                labels = labels.masked_fill(image_mask, -100)
        else:
            # Text-only forward
            # Validate that we don't have stray image tokens
            if (input_ids == self.cfg.image_token_id).any():
                raise ValueError(
                    f"Input contains image token ID {self.cfg.image_token_id} but no images were provided. "
                    "If you intended to use images, please pass the 'images' argument."
                )
            inputs_embeds = self.get_text_embeddings(input_ids)

        # Forward through text backbone
        outputs = self.text_backbone(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )

        return outputs
