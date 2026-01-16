import torch
import torch.nn as nn
import math


class Idefics3Connector(nn.Module):
    """
    Idefics-3 / SmolVLM connector.

    Performs 4×4 (default) pixel-unshuffle followed by a bias-free linear projection.
    Reduces spatial resolution by `scale`× while increasing channel dim by `scale²`.

    Input:  (B, H*W, C_vit)   → e.g. (B, 1024, 768) for SigLIP-base-512
    Output: (B, (H//scale)*(W//scale), D_text) → (B, 64, 576) by default
    """

    def __init__(self, vision_hidden: int, text_hidden: int, scale: int = 4):
        super().__init__()
        self.scale = int(scale)
        if self.scale <= 0 or (self.scale & (self.scale - 1)) != 0:
            raise ValueError("scale must be positive power of 2")

        self.scale_sq = self.scale**2
        self.in_features = vision_hidden * self.scale_sq
        self.out_features = text_hidden
        self.proj = nn.Linear(self.in_features, self.out_features, bias=False)

    @staticmethod
    def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
        """
        Convert (B, H*W, C) → (B, (H//scale)*(W//scale), scale²*C)

        Args:
            x: (B, H*W, C)
            scale: downsample factor (must divide H and W)

        Returns:
            (B, L', scale²*C) where L' = (H//scale)*(W//scale)
        """
        B, HW, C = x.shape
        H = W = math.isqrt(HW)  # assumes square input (true for all ViTs we use)
        if H * W != HW:
            raise ValueError(f"Non-square grid detected: HW={HW}, sqrt≈{HW**0.5}")
        if H % scale != 0 or W % scale != 0:
            raise ValueError(f"H={H}, W={W} not divisible by scale={scale}")

        x = x.contiguous().view(B, H, W, C)
        x = x.view(B, H // scale, scale, W // scale, scale, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H', W', s, s, C)
        return x.view(B, -1, scale * scale * C)  # (B, H'*W', s²*C)

    def forward(self, patch_embeds: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            patch_embeds: (B, H*W, C_vit)
            H, W: grid dimensions (e.g. 32 for 512/16)
        """
        if patch_embeds.shape[1] != H * W:
            raise ValueError(f"Expected {H * W} patches, got {patch_embeds.shape[1]}")

        tokens = self.pixel_unshuffle(
            patch_embeds, scale=self.scale
        )  # (B, L', scale²*C)

        # Validate output length
        expected_L = (H // self.scale) * (W // self.scale)
        if tokens.shape[1] != expected_L:
            raise RuntimeError(
                f"Output length mismatch: got {tokens.shape[1]}, expected {expected_L}"
            )

        if tokens.shape[-1] != self.in_features:
            raise RuntimeError(
                f"Connector input dim mismatch: {tokens.shape[-1]} vs {self.in_features}"
            )

        return self.proj(tokens)  # (B, L', D_text)
