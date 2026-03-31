from collections import defaultdict
import copy
import math
from typing import MutableMapping, Optional, Tuple

import torch


class PositionEncoder:
    """
    Provides the ability to insert position-encoding logic into MHA.
    """

    # Override to adjust the mask e.g. for Alibi
    def adjusted_mask(
        self,
        mask: Optional[torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        past_kv_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_cache=False,
    ) -> Optional[torch.Tensor]:
        return mask

    # Override to adjust q/k's e.g. for rotary embeddings. Note that some
    # implementations may not use cache related kwargs, e.g., pixtral's 2D
    # rope for images, which will not have a past kv states.
    def adjusted_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        past_kv_state: Optional[Tuple[torch.Tensor | None, torch.Tensor | None]],
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
        mask: Optional[torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        past_kv_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        use_cache=False,
    ) -> Optional[torch.Tensor]:
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


class RopeNoScalingImpl:
    def __init__(
        self,
        dim: int,
        ratio: float = 10_000.0,
        orig_max_seq_len: int = 2048,
        scaling_info: dict = defaultdict(),
    ):
        self.dim = dim
        self.ratio = ratio
        self.orig_max_seq_len = orig_max_seq_len
        self.scaling_info = scaling_info

    def get_alpha(self, current_max_seq_len: int) -> int:
        return 1

    def scaled_max_seq_len(self, current_max_seq_len: int, alpha: int):
        return max(current_max_seq_len, self.orig_max_seq_len)

    def compute_scaled_freqs(self, device: str, alpha: int):
        ratio = self.ratio
        dim = self.dim

        freqs = 1.0 / (
            ratio
            ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
        )
        return freqs


class RopeNtkScalingImpl(RopeNoScalingImpl):
    # NTK scaling.
    # https://arxiv.org/abs/2306.15595
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    #
    # we'll store the freqs for each alpha value. This means that for
    # shorter sequences, we preserve the original scale.
    # To limit the number of multiples to store we'll maintain alphas for
    # `2**i` where i is the ratio of actual vs initial max seq len. (i.e. 2,
    # 4, 8, ... as needed)

    def get_alpha(self, current_max_seq_len: int) -> int:
        alpha = current_max_seq_len / self.orig_max_seq_len
        alpha = math.ceil(alpha)
        # for some reason math.log2 didn't `torch.compile` but
        # `math.log` does
        alpha = math.log(alpha) / math.log(2)
        alpha = math.ceil(alpha)
        alpha = 2**alpha
        alpha = int(alpha)
        return alpha

    def scaled_max_seq_len(self, current_max_seq_len: int, alpha: int):
        return max(current_max_seq_len, self.orig_max_seq_len * alpha)

    def compute_scaled_freqs(self, device: str, alpha: int):
        dim = self.dim
        ratio = self.ratio * alpha ** (dim / (dim - 2))

        freqs = 1.0 / (
            ratio
            ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
        )
        return freqs


class RopeLlama3ScalingImpl(RopeNoScalingImpl):
    def compute_scaled_freqs(self, device: str, alpha: int):
        freqs = super().compute_scaled_freqs(device, alpha)

        factor = self.scaling_info["factor"]
        low_freq_factor = self.scaling_info["low_freq_factor"]
        high_freq_factor = self.scaling_info["high_freq_factor"]
        old_context_len = self.scaling_info["original_max_position_embeddings"]

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / freqs
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        freqs_llama = torch.where(wavelen > low_freq_wavelen, freqs / factor, freqs)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_freqs = (
            1 - smooth_factor
        ) * freqs_llama / factor + smooth_factor * freqs_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        freqs = torch.where(is_medium_freq, smoothed_freqs, freqs_llama)
        return freqs


class YarnRopeImpl(RopeNoScalingImpl):
    def __init__(self, dim, ratio, orig_max_seq_len, scaling_info):
        super().__init__(dim, ratio, orig_max_seq_len, scaling_info)
        self.concentration: Optional[float] = None

    def compute_scaled_freqs(self, device: str, alpha: int):
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        scaling_factor = self.scaling_info["scaling_factor"]
        ntk_beta = self.scaling_info["ntk_beta"]
        ntk_alpha = self.scaling_info["ntk_alpha"]
        freq = self.ratio ** (
            torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim
        )
        if scaling_factor > 1.0:
            self.concentration = (
                0.1 * math.log(scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.orig_max_seq_len / (ntk_beta * 2 * math.pi))
                / math.log(self.ratio)
            )
            high = (
                d_half
                * math.log(self.orig_max_seq_len / (ntk_alpha * 2 * math.pi))
                / math.log(self.ratio)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            self.concentration = 1.0
            inv_freq = 1.0 / freq

        return inv_freq

    def apply_yarn_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        cos = cos[position_ids].unsqueeze(-2).to(x.dtype)
        sin = sin[position_ids].unsqueeze(-2).to(x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)


_rope_scale_mapping = {
    "llama3": RopeLlama3ScalingImpl,
    "ntk": RopeNtkScalingImpl,
    "regular": RopeNoScalingImpl,
    "yarn": YarnRopeImpl,
}


class RotaryEmbedding(PositionEncoder):
    def __init__(
        self,
        dim: int,
        ratio: float = 10_000.0,
        max_seq_len: int = 2048,
        partial_rope=1.0,
        scaling={},
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
        partial_rope: int
            fraction of head dimension to apply rope to
        scaling: dict
            dictionary of information on how to scale RoPE to higher seq lens
        """
        super(RotaryEmbedding, self).__init__()
        self.partial_rope = partial_rope
        self.dim = int(partial_rope * dim)
        own_scaling = copy.deepcopy(scaling)
        if "rope_type" not in own_scaling:
            own_scaling["rope_type"] = "regular"
        self.rope_type = own_scaling["rope_type"]
        self.rope_scaling: RopeNoScalingImpl = _rope_scale_mapping[
            own_scaling["rope_type"]
        ](self.dim, ratio, max_seq_len, own_scaling)
        self.cached_freqs: MutableMapping[int, MutableMapping[int, torch.Tensor]] = {}
        self.max_seq_len_cached: MutableMapping[int, int] = {}

    def compute_freqs_cis(self, device, max_seq_len=2048):
        alpha = self.rope_scaling.get_alpha(max_seq_len)

        if device == torch.device("meta"):
            return alpha

        dev_idx = device.index

        if dev_idx not in self.cached_freqs:
            self.cached_freqs[dev_idx] = {}
        if dev_idx not in self.max_seq_len_cached:
            self.max_seq_len_cached[dev_idx] = 0

        if alpha not in self.cached_freqs[dev_idx]:
            # This avoids a graph break from computing scaled_max_seq_len if not needed
            scaled_max_seq_len = self.rope_scaling.scaled_max_seq_len(
                max_seq_len, alpha
            )

            if scaled_max_seq_len > self.max_seq_len_cached[dev_idx]:
                # This only runs if a particular combination of alpha
                # and max_seq_len hasn't been seen before
                freqs = self.rope_scaling.compute_scaled_freqs(device, alpha)
                t = torch.arange(scaled_max_seq_len, device=device, dtype=freqs.dtype)
                freqs = torch.outer(t, freqs).float()

                self.max_seq_len_cached[dev_idx] = scaled_max_seq_len

                if self.rope_type == "yarn":
                    cos = freqs.cos() * self.rope_scaling.concentration
                    sin = freqs.sin() * self.rope_scaling.concentration

                    self.cached_freqs[dev_idx][alpha] = (cos, sin)
                else:
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
        past_kv_state: Optional[Tuple[torch.Tensor | None, torch.Tensor | None]] = None,
        use_cache=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function applies 1D rotary embeddings to the queries and keys.
        Args
        ----
        q : torch.Tensor
            Embedded query tensor, expected size is B x S x H x Eh
        k : torch.Tensor
            Embedded query tensor, expected size is B x S x H x Eh
        position_ids : Optional[torch.LongTensor]
            The position of each of the tokens encoded in q and k. This is important in
            kv-caching and left-padding situations, for which the rotation to be applied might
            not always be the pre-cached position 0...S. For kv-caching without dynamic batching
            or variable per-row left padding position_ids is shared for all the batch.
        """
        assert len(q.size()) == 4
        assert len(k.size()) == 4

        seq_len = max(k.size(1), q.size(1))
        if position_ids is None:
            # Compute position_ids based on cache config
            position_ids = torch.arange(
                0, seq_len, dtype=torch.long, device=q.device
            ).repeat(k.size(0), 1)
            if (
                use_cache
                and past_kv_state is not None
                and past_kv_state[0] is not None
                and past_kv_state[0].numel() > 0
            ):
                position_ids += past_kv_state[0].size(2)

        # the max start position should be based on the max first position of each sequence
        max_start_pos = torch.max(position_ids[:, 0])

        if self.partial_rope != 1.0:
            q_rope = q[..., : self.dim]
            k_rope = k[..., : self.dim]
        else:
            q_rope = q
            k_rope = k
        q_ = q_rope.float().view(*q.size()[:-1], -1, 2)  # B L H D/2 2
        k_ = k_rope.float().view(*k.size()[:-1], -1, 2)  # B L H D/2 2

        # the max start position should be based on the max first position of each sequence
        max_start_pos = torch.max(position_ids[:, 0])

        alpha = self.compute_freqs_cis(q.device, max_start_pos + seq_len)

        if self.rope_type == "yarn":
            # Compute cos/sin for max position
            freqs = self.cached_freqs[q.device.index][alpha]

            cos = freqs[0]
            sin = freqs[1]

            query_shape = q.shape
            key_shape = k.shape

            query = self.rope_scaling.apply_yarn_rotary_emb(q, cos, sin, position_ids)  # type: ignore
            key = self.rope_scaling.apply_yarn_rotary_emb(k, cos, sin, position_ids)  # type: ignore

            return query.reshape(query_shape), key.reshape(key_shape)

        freqs = self.cached_freqs[q.device.index][alpha][position_ids]

        position_ids = position_ids.clamp(max=freqs.size(0) - 1)

        freqs = freqs.float()  # 1 L D/2 2 2

        q_out = (
            freqs[:, -q.size(1) :, None, :, :, :]
            .mul(q_.unsqueeze(-2))
            .sum(5)
            .flatten(3)
        ).type_as(q)
        k_out = (
            freqs[:, -k.size(1) :, None, :, :, :]
            .mul(k_.unsqueeze(-2))
            .sum(5)
            .flatten(3)
        ).type_as(k)

        if self.partial_rope != 1.0:
            q_out = torch.cat([q_out.view_as(q_rope), q[..., self.dim :]], dim=-1)
            k_out = torch.cat([k_out.view_as(k_rope), k[..., self.dim :]], dim=-1)
        else:
            q_out = q_out.view_as(q_rope)
            k_out = k_out.view_as(k_rope)
        return q_out, k_out


class PixtralRotaryEmbedding(PositionEncoder):
    def __init__(
        self,
        dim: int,
        theta: float,
        image_size: int,
        patch_size: int,
    ):
        """
        This implements PixtralRotaryEmbedding, which handles frequency
        for each pixel positions The key difference from standard RoPe is
        that the frequencies are pre-computed for a 2D grid of maximal
        height x width patches.
        """

        super().__init__()
        self.max_patches_per_side = image_size // patch_size
        self.dim = dim
        self.theta = theta
        # NOTE: pixtral does not do rope scaling, i.e., alpha is always 1.
        # For simplicity and to keep the implementation readable, we just
        # map the device index to the freqs directly.
        self.cached_freqs: dict[int, torch.Tensor] = {}

    def compute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """
        Computes the rotational transforms for Pixtral, which encodes
        image patches using 2D positional IDs. If the rotation matrices
        have already been computed for this device index, the cached
        results are returned.

        NOTE: This implementation is similar in implementation to that in
        mistral inference (see link below) without complex numbers. This is
        easier to compare to for numeric correctness than the HF implementation
        because it uses the same weight permutation for interleave RoPE rather
        than permuting the weights to use rotate half.

        https://github.com/mistralai/mistral-inference/blob/v1.6.0/src/mistral_inference/rope.py

        Args:
            device: device to compute frequencies on
        """
        if device == torch.device("meta"):
            raise AssertionError("Attempted to init pixtral freqs on meta device")

        freqs = 1.0 / (
            self.theta
            ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )

        # Create position indices for height and width
        h = torch.arange(self.max_patches_per_side, device=device)
        w = torch.arange(self.max_patches_per_side, device=device)

        # Compute product: position * frequency for each dimension; use the
        # even indices for the height, and the odd indices for the width.
        freqs_h = torch.outer(h, freqs[::2]).float()
        freqs_w = torch.outer(w, freqs[1::2]).float()

        # Frequencies are 2 dimensional
        freqs_2d = torch.cat(
            [
                freqs_h[:, None, :].repeat(1, self.max_patches_per_side, 1),
                freqs_w[None, :, :].repeat(self.max_patches_per_side, 1, 1),
            ],
            dim=-1,
        )

        # [max_size, max_size, dim, 4]
        rot_matrices = torch.stack(
            [
                torch.cos(freqs_2d),
                -torch.sin(freqs_2d),
                torch.sin(freqs_2d),
                torch.cos(freqs_2d),
            ],
            dim=-1,
        )
        # Create the 2x2 rotation matrices -> [max_size, max_size, dim, 2, 2]
        rot_matrices = rot_matrices.view(*freqs_2d.size(), 2, 2)
        self.cached_freqs[device.index] = rot_matrices
        return rot_matrices

    def adjusted_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function applies 2D rotary embeddings to the queries and keys.
        NOTE: as this is only used by Pixtral (vision encoder), we do not
        need to consider caching related args/kwargs, because it will only
        be used at prefill time.

        Args
        ----
        q : torch.Tensor
            Embedded query tensor, expected size is B x S x H x Eh
        k : torch.Tensor
            Embedded query tensor, expected size is B x S x H x Eh
        position_ids : torch.LongTensor
            2D positional IDs (patch coordinates).
        """
        if position_ids is None or position_ids.ndim < 2 or position_ids.shape[-1] != 2:
            raise ValueError("Position IDs for Pixtral must be 2D (H, W) coordinates.")

        q_ = q.float().view(*q.size()[:-1], -1, 2)  # [B, L, H, D/2, 2]
        k_ = k.float().view(*k.size()[:-1], -1, 2)  # [B, L, H, D/2, 2]

        # Positionally encode the 2D positional IDs.
        # position_ids has shape [B, L, 2] with batch dimension included
        self.compute_freqs_cis(q.device)
        freqs = self.cached_freqs[q.device.index][
            position_ids[:, :, 0], position_ids[:, :, 1]
        ]

        freqs = freqs.float()  # [B, L, D/2, 2, 2]

        # [B, L, 1, D/2, 2, 2] x [B, L, N, D/2, 1, 2]
        # Which broadcasts to [B, L, N, D/2, 2, 2]
        mulq = freqs[:, -q.size(1) :, None, :, :, :].mul(q_.unsqueeze(-2))
        mulk = freqs[:, -k.size(1) :, None, :, :, :].mul(k_.unsqueeze(-2))

        # Sum the last dimension out, creating a [B, L, N, D/2, 2], then flatten
        # the (new) 3rd dimension to create the [B, L, N, D] output.
        q_out = mulq.sum(5).flatten(3).type_as(q)
        k_out = mulk.sum(5).flatten(3).type_as(k)
        return q_out, k_out


class CachedYarnRotaryEmbedding(PositionEncoder):
    def __init__(
        self,
        dim: int,  # Rotary dimension
        original_max_position_embeddings: int,
        base: float,  # Rope theta
        scaling_factor: float,  # factor
        *,
        extrapolation_factor: float = 1.0,
        attn_factor: Optional[float] = None,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 1.0,
        llama_4_scaling_beta: Optional[float] = None,
        **kwargs,
    ):
        """
        This implements Yarn scaling rotary embedding.

        Credits to Peng et al. github.com/jquesnelle/yarn
        Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py
        Ref: https://github.com/mistralai/vllm-release/blob/3e21dacb79471ebf946e72e67a5ca14ebcc598c1/vllm/model_executor/layers/rotary_embedding.py#L268
        """

        super().__init__()
        self.dim = dim
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.beta_fast = beta_fast  # low
        self.beta_slow = beta_slow  # high
        self.llama_4_scaling_beta = (
            llama_4_scaling_beta if llama_4_scaling_beta is not None else 1
        )

        self.cached_freqs: dict[int, torch.Tensor] = {}
        self.max_seq_len_cached: MutableMapping[int, int] = {}

        # magnitude scaling factor
        self.mscale = float(self._yarn_get_mscale(mscale))

        self.mscale_all_dim = float(self._yarn_get_mscale(mscale_all_dim))

        # NOTE: We are not computing attn_factor based on mscale here, since its not requried for ministral3
        if attn_factor is None:
            attn_factor = float(self.mscale / self.mscale_all_dim)

        self.attn_factor = attn_factor

    def _yarn_get_mscale(self, mscale: float = 1) -> float:
        if self.scaling_factor <= 1:
            return 1.0
        return 0.1 * mscale * math.log(self.scaling_factor) + 1.0

    def _compute_cos_sin_cache(
        self, inv_freq: torch.Tensor, device: torch.device, max_seq_len: int
    ) -> torch.Tensor:
        """
        Compute the rotation matrix cache for the rotary embedding to avoid computing
        while doing the forward pass.
        Args:
            inv_freq: The precomputed inverse frequency tensor
            device: The device to compute on
            max_seq_len: Maximum sequence length to cache
        Returns:
            Rotation matrices with shape [max_pos, dim/2, 2, 2]
        """
        t = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )
        freqs = torch.outer(t, inv_freq).float()

        # Apply mscale and compute cos/sin
        cos = freqs.cos() * self.attn_factor
        sin = freqs.sin() * self.attn_factor

        # Construct rotation matrices: [max_pos, dim/2, 2, 2]
        # Matrix form: [[cos, -sin], [sin, cos]]
        freqs_cis = torch.stack([cos, -sin, sin, cos], dim=-1).view(*cos.shape, 2, 2)

        return freqs_cis

    def _get_llama_4_attn_scale(self, positions_ids: torch.Tensor) -> torch.Tensor:
        scaling = 1 + self.llama_4_scaling_beta * torch.log(
            1 + torch.floor(positions_ids / self.original_max_position_embeddings)
        )
        return scaling.unsqueeze(-1)

    def compute_freqs_cis(self, device: torch.device, max_seq_len) -> None:
        """
        Compute and cache rotation matrices for the target device.

        This method computes cos/sin rotation matrices and caches them per device.
        If the requested max_seq_len exceeds the cached length, it recomputes with
        the new length.

        Args:
            device: target device to compute rotation matrices on
            max_seq_len: maximum sequence length for the model, if exceeded the cached freqs will be recomputed
        """

        if device == torch.device("meta"):
            return

        dev_idx = device.index

        # Initialize cache entries for this device if not present
        if dev_idx not in self.cached_freqs:
            self.cached_freqs[dev_idx] = None
        if dev_idx not in self.max_seq_len_cached:
            self.max_seq_len_cached[dev_idx] = 0

        # Check if cache is empty (first time)
        if self.cached_freqs[dev_idx] is None:
            # Use scaled max_seq_len for cache size
            # This avoids a graph break from computing scaled_max_seq_len if not needed
            scaled_max_seq_len = int(self.original_max_position_embeddings * self.scaling_factor)
            cache_size = max(max_seq_len, scaled_max_seq_len)

            # Only recompute if we need a longer sequence than what's cached
            if cache_size > self.max_seq_len_cached[dev_idx]:
                freqs = self.base ** (
                    torch.arange(0, self.dim, 2, device=device).float() / self.dim
                )

                inv_freq_extrapolation = 1.0 / freqs
                inv_freq_interpolation = 1.0 / (self.scaling_factor * freqs)

                # NOTE: math.floor and math.ceil being used here are referred to as "truncate" option
                low = math.floor(
                    self.dim
                    * math.log(
                        self.original_max_position_embeddings / (self.beta_fast * 2 * math.pi)
                    )
                ) / (2 * math.log(self.base))
                high = math.ceil(
                    self.dim
                    * math.log(
                        self.original_max_position_embeddings / (self.beta_slow * 2 * math.pi)
                    )
                ) / (2 * math.log(self.base))

                # Make sure values are not going outside range
                low = max(low, 0)
                high = min(high, self.dim - 1)

                if low == high:
                    high += 0.001  # Prevent singularity

                # Get n-dimensional rotational scaling corrected for extrapolation
                linear_func = (
                    torch.arange(self.dim // 2, dtype=torch.float32, device=device) - low
                ) / (high - low)

                # Compute ramp function (clamped linear interpolation)
                ramp_func = torch.clamp(linear_func, 0, 1)

                # inv_freq_extrapolation_factor is the weight for extrapolation
                # (1 - ramp_func) means: use extrapolation for low frequencies (< low)
                # ramp_func means: use interpolation for high frequencies (> high)
                inv_freq_extrapolation_factor = 1 - ramp_func

                # Blend between interpolation and extrapolation
                # Note: extrapolation_factor is applied to the extrapolation frequencies
                inv_freq = (
                    inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
                    + inv_freq_extrapolation
                    * inv_freq_extrapolation_factor
                    * self.extrapolation_factor
                )

                # Cache the computed rotation matrices for this device
                freqs_cis = self._compute_cos_sin_cache(inv_freq, device, cache_size)
                self.cached_freqs[dev_idx] = freqs_cis
                self.max_seq_len_cached[dev_idx] = cache_size

    def adjusted_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_kv_state: Optional[Tuple[torch.Tensor | None, torch.Tensor | None]] = None,
        use_cache=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function applies 1D rotary embeddings to the queries and keys using interleaved rotation.
        Args
        ----
        q : torch.Tensor
            Embedded query tensor, expected size is B x S x H x D
            where B=batch, S=sequence length, H=num heads, D=head dimension
        k : torch.Tensor
            Embedded key tensor, expected size is B x S x H x D
            where B=batch, S=sequence length, H=num heads, D=head dimension
        position_ids : Optional[torch.LongTensor]
            The position of each of the tokens encoded in q and k. This is important in
            kv-caching and left-padding situations, for which the rotation to be applied might
            not always be the pre-cached position 0...S. For kv-caching without dynamic batching
            or variable per-row left padding position_ids is shared for all the batch.
        past_kv_state : Optional[Tuple[torch.Tensor | None, torch.Tensor | None]]
            Past key-value states for caching
        use_cache : bool
            Whether to use KV caching
        """

        assert len(q.size()) == 4
        assert len(k.size()) == 4

        seq_len = max(k.size(1), q.size(1))
        if position_ids is None:
            # Compute position_ids based on cache config
            position_ids = torch.arange(
                0, seq_len, dtype=torch.long, device=q.device
            ).repeat(k.size(0), 1)
            if (
                use_cache
                and past_kv_state is not None
                and past_kv_state[0] is not None
                and past_kv_state[0].numel() > 0
            ):
                position_ids += past_kv_state[0].size(2)

        # the max start position should be based on the max first position of each sequence
        max_start_pos = torch.max(position_ids[:, 0])

        # Fetch the rotation matrices from cache
        self.compute_freqs_cis(q.device, max_start_pos + seq_len)

        # Get device index for cache lookup, None if on CPU
        dev_idx = q.device.index

        # Index by position_ids: [B, L] -> [B, L, rotary_dim/2, 2, 2]
        freqs = self.cached_freqs[dev_idx][position_ids]
        freqs = freqs.float()

        # Only apply rotation to the first self.dim dimensions
        # Extract the rotary portion
        q_rope = q
        k_rope = k

        # Reshape for interleaved rotation
        # From [B, L, H, rotary_dim] to [B, L, H, rotary_dim/2, 2] for interleaved pairs
        q_ = q_rope.float().view(*q_rope.size()[:-1], -1, 2)  # B L H rotary_dim/2 2
        k_ = k_rope.float().view(*k_rope.size()[:-1], -1, 2)  # B L H rotary_dim/2 2

        # Apply rotation using matrix multiplication
        # freqs: [B, L, rotary_dim/2, 2, 2]
        # q_, k_: [B, L, H, rotary_dim/2, 2]
        q_out = (
            freqs[:, -q.size(1) :, None, :, :, :]
            .mul(q_.unsqueeze(-2))
            .sum(5)
            .flatten(3)
        ).type_as(q)
        k_out = (
            freqs[:, -k.size(1) :, None, :, :, :]
            .mul(k_.unsqueeze(-2))
            .sum(5)
            .flatten(3)
        ).type_as(k)

        # Apply llama_4_scaling
        if self.llama_4_scaling_beta:
            cache_position = torch.arange(
                q_out.shape[2], device=q_out.device, dtype=q_out.dtype
            )
            q_out = q_out * self._get_llama_4_attn_scale(cache_position)

        q_out = q_out.view_as(q_rope)
        k_out = k_out.view_as(k_rope)

        return q_out, k_out
