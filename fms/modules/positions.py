import math
from typing import Tuple

import torch
from torch import nn


class PositionEncoder:
    """
    Provides the ability to insert position-encoding logic into MHA.
    """

    # Override to adjust the mask e.g. for Alibi
    def adjusted_mask(
        self,
        mask: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        past_kv_state: torch.Tensor,
        use_cache=False,
    ) -> torch.Tensor:
        return mask

    # Override to adjust q/k's e.g. for rotary embeddings
    def adjusted_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        past_kv_state: torch.Tensor,
        use_cache=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return q, k


class Alibi(PositionEncoder):
    """
    Attention Linear Bias layer for sequence models, as in https://arxiv.org/pdf/2108.12409.pdf.
    ...
    Args
    ----
    nheads : int
        Number of attention heads (and thus position bias matrices)
    max_scale : float
        Maximum scaling factor. Defaults to 0.5 as in paper.
    min_scale : float
        Minimum scaling factor. Defaults to 2^-8 as in paper.
    """

    def __init__(self, nheads, max_scale=0.5, min_scale=1 / (2**8)):
        super(Alibi, self).__init__()
        self.nheads = nheads
        start = math.log2(max_scale)
        end = math.log2(min_scale)
        self.scales = (
            2
            ** torch.arange(
                start, end + 1e-6 * math.sign(end - start), (end - start) / (nheads - 1)
            ).view(1, nheads, 1, 1),
        )

    def adjusted_mask(
        self,
        mask: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        past_kv_state: torch.Tensor,
        use_cache=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qlen = q.size(1)
        klen = k.size(1)

        # if we are using the cache, the key length needs to be extended with the past keys length
        if use_cache and past_kv_state is not None and past_kv_state[0] is not None:
            klen += past_kv_state[0][0].size(-2)
            qlen += past_kv_state[0][1].size(-2)

        # Automatically allocates on chosen cuda
        device = self.scales.device
        q_pos = torch.arange(qlen, dtype=torch.long, device=device)
        k_pos = torch.arange(klen, dtype=torch.long, device=device)

        # rel_pos: qlen x klen
        rel_pos = k_pos[None, :] - q_pos[:, None]
        values = rel_pos.abs().neg().unsqueeze(0).unsqueeze(0)

        bias = values * self.scales

        # we need to pick the k-length row of alibi maxtrix when caching is being used and not first iteration
        if use_cache and klen != 1 and qlen == 1:
            bias = bias[:, :, -1:, :]

        attn_mask = bias
        # We expect the shapes of mask and rel_pos_bias to be at least broadcastable
        if mask is not None:
            # Can't do in-place op in case broadcast makes attn_mask bigger
            attn_mask = attn_mask.masked_fill(mask == 0, float("-inf"))

        return attn_mask


class RotaryEmbedding(PositionEncoder):
    def __init__(
        self,
        dim: int,
        ratio: int = 10_000,
        device=None
    ):
        """
        This implementation of Rotary Position Embeddings (RoPE) avoids
        complex numbers, and so can be used with torch.compile.

        https://arxiv.org/abs/2104.09864

        ...
        Args
        ----
        dim : int
            Per-head embedding dimension
        max_seq_len : int
            Maximum expected sequence length for the model, if exceeded the cached freqs will be recomputed
        ratio: int
            The ratio for the geometric progression to compute the rotation angles
        """
        super(RotaryEmbedding, self).__init__()
        self.freqs = 1.0 / (
            ratio
            ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
        )
        self.cached_freqs = {}
        self.max_seq_len_cached = {}

    def compute_freqs_cis(self, device, max_position_embeddings=2048):
        if device in self.cached_freqs and max_position_embeddings <= self.max_seq_len_cached[device]:
            return

        t = torch.arange(
            max_position_embeddings, device=device, dtype=self.freqs.dtype
        )
        freqs = torch.outer(t, self.freqs.to(device)).float()
        self.max_seq_len_cached[device] = max_position_embeddings
        self.cached_freqs[device] = torch.stack(
            [
                torch.cos(freqs),
                -torch.sin(freqs),
                torch.sin(freqs),
                torch.cos(freqs),
            ],
            dim=2,
        ).view(*freqs.shape, 2, 2)

    def reshape_for_broadcast(self, x: torch.Tensor, cur_freqs):
        ndim = x.ndim
        assert 1 < ndim, ndim
        assert cur_freqs.shape[:2] == (
            x.shape[2],
            x.shape[-2],
        ), f"for {cur_freqs.shape} and {x.shape}"
        shape = [d if i == 2 or i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
        return cur_freqs.view(*shape, 2)

    def adjusted_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        past_kv_state: torch.Tensor = None,
        use_cache=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        q : torch.Tensor
            Embedded query tensor, expected size is B x H x S x Eh
        k : torch.Tensor
            Embedded query tensor, expected size is B x H x S x Eh
        past_kv_state : torch.Tensor
            the k/v cache from the previous forward pass.
        """
        start_pos = 0
        if use_cache and past_kv_state is not None:
            # TODO: handle batched start positions?
            start_pos = past_kv_state[0].shape[-2]

        seq_len = q.shape[2]
        q_ = q.float().reshape(*q.shape[:-1], -1, 2)  # B H L D/2 2
        k_ = k.float().reshape(*k.shape[:-1], -1, 2)  # B H L D/2 2

        if isinstance(start_pos, int):
            self.compute_freqs_cis(q.device, start_pos + seq_len)
            cur_freqs = self.cached_freqs[q.device][start_pos : start_pos + seq_len]
            freqs = self.reshape_for_broadcast(q_, cur_freqs)
        else:
            # TODO: this branch currently unused
            max_start_pos = torch.max(start_pos).item()
            self.compute_freqs_cis(q.device, max_start_pos + seq_len)
            freqs_idxs = torch.arange(0, seq_len, dtype=torch.long).repeat(
                start_pos.shape[0]
            ).view(-1, seq_len) + start_pos.view(-1, 1)
            freqs = self.cached_freqs[q.device][freqs_idxs].unsqueeze(1)

        freqs = freqs.float()  # 1 1 L D/2 2 2
        q_out = freqs.mul(q_.unsqueeze(-2)).sum(5).flatten(3)
        k_out = freqs.mul(k_.unsqueeze(-2)).sum(5).flatten(3)
        return q_out.type_as(q).contiguous(), k_out.type_as(k).contiguous()
