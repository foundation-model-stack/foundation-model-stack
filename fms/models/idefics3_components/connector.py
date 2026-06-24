import torch
import torch.nn as nn


class Idefics3Connector(nn.Module):
    """
    Idefics-3 / SmolVLM connector.

    Performs 4x4 (default) pixel-unshuffle followed by a bias-free linear projection.
    Reduces spatial resolution by `scale`x while increasing channel dim by `scale^2`.

    Input:  (B, H*W, C_vit)   -> e.g. (B, 1024, 768) for SigLIP-base-512
    Output: (B, (H//scale)*(W//scale), D_text) -> (B, 64, 576) by default
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

    def pixel_unshuffle(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Convert (B, H*W, C) -> (B, (H//scale)*(W//scale), scale^2*C)

        Args:
            x: (B, H*W, C)

        Returns:
            (B, L', scale^2*C) where L' = (H//scale)*(W//scale)
        """
        if x.ndim != 3:
            raise ValueError(
                "pixel_unshuffle expects (B, H*W, C) input; "
                f"got x.shape={tuple(x.shape)}"
            )

        scale = self.scale
        B = x.shape[0]
        n_patches = x.shape[-2]
        C = x.shape[-1]

        expected_patches = H * W
        if expected_patches != n_patches:
            raise ValueError(
                f"pixel_unshuffle expected N={expected_patches} patches (H={H}, W={W}, scale={scale}), "
                f"got N={n_patches} from x.shape={tuple(x.shape)}"
            )
        if H % scale != 0 or W % scale != 0:
            raise ValueError(f"H={H}, W={W} not divisible by scale={scale}")

        x = x.contiguous().view(B, H, W, C)
        x = x.view(B, H // scale, scale, W // scale, scale, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H', W', s, s, C)
        return x.view(B, -1, scale * scale * C)  # (B, H'*W', s^2*C)

    def forward(self, patch_embeds: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            patch_embeds: (B, H*W, C_vit)
            H, W: grid dimensions (e.g. 32 for 512/16)
        """
        if patch_embeds.ndim != 3:
            raise ValueError(
                "Idefics3Connector expects patch_embeds with shape (B, H*W, C_vit); "
                f"got patch_embeds.shape={tuple(patch_embeds.shape)}"
            )

        expected_patches = H * W
        n_patches = patch_embeds.shape[-2]
        if n_patches != expected_patches:
            raise ValueError(
                f"Idefics3Connector expected N={expected_patches} patches (H={H}, W={W}, scale={self.scale}), "
                f"got N={n_patches} from patch_embeds.shape={tuple(patch_embeds.shape)}"
            )

        tokens = self.pixel_unshuffle(patch_embeds, H=H, W=W)  # (B, L', scale^2*C)

        # Validate output length
        expected_L = (H // self.scale) * (W // self.scale)
        if tokens.shape[-2] != expected_L:
            raise RuntimeError(
                f"Output length mismatch: got {tokens.shape[-2]}, expected {expected_L} "
                f"(tokens.shape={tuple(tokens.shape)})"
            )

        if tokens.shape[-1] != self.in_features:
            raise RuntimeError(
                f"Connector input dim mismatch: {tokens.shape[-1]} vs {self.in_features}"
            )

        return self.proj(tokens)  # (B, L', D_text)
