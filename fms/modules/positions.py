import math
from typing import MutableMapping, Optional, Tuple

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
        position_ids: Optional[torch.LongTensor],
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
    ) -> torch.Tensor:
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
        self, dim: int, ratio: int = 10_000, max_seq_len=2048, ntk_scaling=False
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
        self.dim = dim
        self.ratio = ratio
        self.cached_freqs: MutableMapping[int, MutableMapping[int, torch.Tensor]] = {}
        self.max_seq_len_cached: MutableMapping[int, int] = {}
        self.ntk_scaling = ntk_scaling
        self.max_seq_len = max_seq_len

    def _alpha(self, seq_len) -> int:
        if not self.ntk_scaling:
            return 1
        else:
            alpha = seq_len / self.max_seq_len
            alpha = math.ceil(alpha)
            # for some reason math.log2 didn't `torch.compile` but
            # `math.log` does
            alpha = math.log(alpha) / math.log(2)
            alpha = math.ceil(alpha)
            alpha = 2**alpha
            alpha = int(alpha)
            return alpha

    def compute_freqs_cis(self, device, max_seq_len=2048):
        # NTK scaling.
        # https://arxiv.org/abs/2306.15595
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        #
        # we'll store the freqs for each alpha value. This means that for
        # shorter sequences, we preserve the original scale.
        # To limit the number of multiples to store we'll maintain alphas for
        # `2**i` where i is the ratio of actual vs initial max seq len. (i.e. 2,
        # 4, 8, ... as needed)
        alpha = self._alpha(max_seq_len)
        dev_idx = device.index

        if dev_idx not in self.cached_freqs:
            self.cached_freqs[dev_idx] = {}
        if dev_idx not in self.max_seq_len_cached:
            self.max_seq_len_cached[dev_idx] = 0

        # This condition can be combined with the model using Rotary calling this method
        # on model init when device is known to avoid a graph break (see llama.py)
        if self.ntk_scaling:
            max_seq_len = max(max_seq_len, self.max_seq_len * alpha)
        else:
            if self.max_seq_len_cached[dev_idx] > 0:
                return alpha
            max_seq_len = max(max_seq_len, self.max_seq_len)

        if (
            alpha in self.cached_freqs[dev_idx]
            and max_seq_len <= self.max_seq_len_cached[dev_idx]
        ):
            return alpha

        ratio = self.ratio
        dim = self.dim

        if self.ntk_scaling:
            ratio = ratio * alpha ** (dim / (dim - 2))

        freqs = 1.0 / (
            ratio
            ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
        )

        t = torch.arange(max_seq_len, device=device, dtype=freqs.dtype)
        freqs = torch.outer(t, freqs).float()
        self.max_seq_len_cached[dev_idx] = max_seq_len
        self.cached_freqs[dev_idx][alpha] = torch.stack(
            [
                torch.cos(freqs),
                -torch.sin(freqs),
                torch.sin(freqs),
                torch.cos(freqs),
            ],
            dim=2,
        ).view(*freqs.size(), 2, 2)

        return alpha

    def reshape_for_broadcast(self, x: torch.Tensor, cur_freqs):
        ndim = x.ndim
        assert 1 < ndim, ndim
        assert cur_freqs.size()[:2] == (
            x.size(2),
            x.size(-2),
        ), f"for {cur_freqs.size()} and {x.size()}"
        shape = [d if i == 2 or i >= ndim - 2 else 1 for i, d in enumerate(x.size())]
        return cur_freqs.view(*shape, 2)

    def adjusted_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_kv_state: Optional[torch.Tensor] = None,
        use_cache=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        q : torch.Tensor
            Embedded query tensor, expected size is B x H x S x Eh
        k : torch.Tensor
            Embedded query tensor, expected size is B x H x S x Eh
        position_ids : Optional[torch.LongTensor]
            The position of each of the tokens encoded in q and k. This is important in
            kv-caching and left-padding situations, for which the rotation to be applied might
            not always be the pre-cached position 0...S. For kv-caching without dynamic batching
            or variable per-row left padding position_ids is shared for all the batch.
        """
        assert len(q.size()) == 4
        assert len(k.size()) == 4
        if position_ids is None:
            # Compute position_ids based on cache config
            position_ids = torch.arange(
                0, q.size(2), dtype=torch.long, device=q.device
            ).repeat(q.size(0), 1)
            if use_cache and past_kv_state is not None:
                position_ids += past_kv_state[0].size(2)

        seq_len = q.size(2)
        q_ = q.float().reshape(*q.size()[:-1], -1, 2)  # B H L D/2 2
        k_ = k.float().reshape(*k.size()[:-1], -1, 2)  # B H L D/2 2

        # the max start position should be based on the max first position of each sequence
        max_start_pos = torch.max(position_ids[:, 0])
        alpha = self.compute_freqs_cis(q.device, max_start_pos + seq_len)
        freqs = self.cached_freqs[q.device.index][alpha][position_ids].unsqueeze(1)

        freqs = freqs.float()  # 1 1 L D/2 2 2
        q_out = freqs.mul(q_.unsqueeze(-2)).sum(5).flatten(3)
        k_out = freqs.mul(k_.unsqueeze(-2)).sum(5).flatten(3)

        return q_out.type_as(q).contiguous(), k_out.type_as(k).contiguous()
