from typing import Tuple

import torch


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts, with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible by block_size for proper block matrix operations.
    """

    # First count how many tokens go to each expert
    cnts = torch.zeros(
        topk_ids.shape[0], num_experts, dtype=topk_ids.dtype, device=topk_ids.device
    )
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)

    # Then pad the amount to the block size of work
    tokens_per_expert_block_padded = (
        torch.floor_divide(tokens_per_expert + block_size - 1, block_size) * block_size
    )

    # Count how many tokens in total are we computing in the GPU
    cumsum = tokens_per_expert_block_padded.cumsum(0)
    total_padded_tokens = cumsum[-1]

    # For allocation purposes, compute the worst case scenario for how
    # many tokens we could be doing work for if the RNG gods are bad
    max_total_padded_tokens = (
        (topk_ids.numel() + num_experts * (block_size - 1))
        if topk_ids.numel() > num_experts
        else (topk_ids.numel() + 1) * block_size
    )

    # Compute what MoE expert corresponds to each block of work
    # A block of work consists of tokens that all go to the same expert
    # There are total_padded_tokens // block_size blocks of work, but to
    # simplify kernel launches, we allocate based on worst case and use
    # max_total_padded_tokens instead.
    expert_block_mapping = torch.zeros(
        max((max_total_padded_tokens + block_size - 1) // block_size + 1, num_experts),
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )

    num_blocks_per_expert = cumsum.div(block_size, rounding_mode="floor")
    ones = torch.ones_like(expert_block_mapping)
    expert_block_mapping.scatter_add_(0, num_blocks_per_expert, ones)
    expert_block_mapping = expert_block_mapping.cumsum(0)

    # Create the mapping between token idxs in the input tensor and
    # the tokens that will go in each work block

    # Count how many pad tokens need adding to the list of tokens in
    # topk_ids, then create a tensor that will fill in the right spots
    # when doing an argsort later to compute padded_token_ids_per_block
    cum_pad_tokens = (tokens_per_expert_block_padded - tokens_per_expert).cumsum(0)
    padded_tokens = torch.zeros(
        max_total_padded_tokens - topk_ids.numel(),
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )
    ones = torch.ones_like(padded_tokens)
    padded_tokens.scatter_add_(0, cum_pad_tokens[:-1], ones)
    padded_tokens = padded_tokens.cumsum(0)
    padded_token_ids_per_block = torch.cat([topk_ids.view(-1), padded_tokens]).argsort()

    return padded_token_ids_per_block, expert_block_mapping, total_padded_tokens


torch.library.define(
    "moe::moe_mm",
    "(Tensor input, Tensor moe_matrix, Tensor token_expert_mapping, Tensor padded_token_ids_per_block, Tensor expert_block_mapping, Tensor total_padded_tokens, int topk, int padding_size) -> Tensor",
)


# All that's needed for torch.compile support
@torch.library.impl_abstract("moe::moe_mm")
def moe_mm_meta(
    input: torch.Tensor,
    moe_matrix: torch.Tensor,
    token_expert_mapping: torch.Tensor,
    padded_token_ids_per_block: torch.Tensor,
    expert_block_mapping: torch.Tensor,
    total_padded_tokens: torch.Tensor,
    topk: int,
    padding_size,
):
    M, A = token_expert_mapping.shape
    _, N, _ = moe_matrix.shape
    return torch.empty((M, A, N), device=input.device, dtype=input.dtype)


@torch.library.impl("moe::moe_mm", "CUDA")
def moe_mm(
    input: torch.Tensor,
    moe_matrix: torch.Tensor,
    token_expert_mapping: torch.Tensor,
    padded_token_ids_per_block: torch.Tensor,
    expert_block_mapping: torch.Tensor,
    total_padded_tokens: torch.Tensor,
    topk: int,
    padding_size,
):
    from fms.triton.moe_kernel import (
        BEST_MOE_CONFIGS,
        _autotune,
        _create_best_configs_key,
        _load_best_configs,
        _save_best_configs,
        invoke_fused_moe_kernel,
    )

    M, A = token_expert_mapping.shape
    E, N, _ = moe_matrix.shape
    output = torch.zeros((M, A, N), device=input.device, dtype=input.dtype)

    if BEST_MOE_CONFIGS is None:
        BEST_MOE_CONFIGS = _load_best_configs()
    # Loading must have not been successful. Let's create a new dictionary.
    if BEST_MOE_CONFIGS is None:
        BEST_MOE_CONFIGS = {}
    key = _create_best_configs_key(input, moe_matrix, token_expert_mapping)
    if key not in BEST_MOE_CONFIGS:
        import functools

        # TODO: Add more configs?
        configs = [
            {
                "block_m": 64,
                "block_n": 64,
                "block_k": 32,
            },
            {
                "block_m": 16,
                "block_n": 32,
                "block_k": 64,
            },
        ]

        configs = [config for config in configs if config["block_m"] == padding_size]
        best, best_config = _autotune(
            configs,
            functools.partial(
                invoke_fused_moe_kernel,
                input,
                moe_matrix,
                output,
                token_expert_mapping,
                padded_token_ids_per_block,
                expert_block_mapping,
                total_padded_tokens,
                topk,
            ),
        )
        BEST_MOE_CONFIGS[key] = best_config
        _save_best_configs(BEST_MOE_CONFIGS)
    best_config = BEST_MOE_CONFIGS[key]
    if best_config is None:
        return torch.tensor([])

    invoke_fused_moe_kernel(
        input,
        moe_matrix,
        output,
        token_expert_mapping,
        padded_token_ids_per_block,
        expert_block_mapping,
        total_padded_tokens,
        topk,
        best_config,
    )

    return output


@torch.library.impl("moe::moe_mm", ["CPU", "MPS"])
def moe_mm_cpu(
    input: torch.Tensor,
    moe_matrix: torch.Tensor,
    token_expert_mapping: torch.Tensor,
    padded_token_ids_per_block: torch.Tensor,
    expert_block_mapping: torch.Tensor,
    total_padded_tokens: torch.Tensor,
    topk: int,
    padding_size,
):
    T, D = input.shape
    M, A = token_expert_mapping.shape

    a = input.view(T, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(T * topk, moe_matrix.shape[1], dtype=a.dtype, device=a.device)

    token_expert_mapping = token_expert_mapping.view(-1)
    for i in range(moe_matrix.shape[0]):
        mask = token_expert_mapping == i
        if mask.sum():
            out[mask] = a[mask] @ moe_matrix[i].transpose(0, 1)
    return out.view(M, A, moe_matrix.shape[1])
