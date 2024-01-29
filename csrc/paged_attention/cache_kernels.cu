#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

namespace vllm {

template<typename scalar_t>
__global__ void reshape_and_cache_kernel(
  const scalar_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
  scalar_t* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                                + head_idx * (head_size / x) * block_size * x
                                + x_idx * block_size * x
                                + block_offset * x
                                + x_offset;
    const int64_t tgt_value_idx = block_idx * num_heads * head_size * block_size
                                  + head_idx * head_size * block_size
                                  + head_offset * block_size
                                  + block_offset;
    key_cache[tgt_key_idx] = key[src_key_idx];
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}

} // namespace vllm

void reshape_and_cache(
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,   // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& slot_mapping)  // [num_tokens]
{
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    key.scalar_type(),
    "reshape_and_cache_kernel",
    [&] {
      vllm::reshape_and_cache_kernel<scalar_t><<<grid, block, 0, stream>>>(
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int64_t>(),
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        x);
    });
}

namespace vllm {

template<typename scalar_t>
__global__ void reshape_and_cache_key_kernel(
  const scalar_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int64_t tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                                + head_idx * (head_size / x) * block_size * x
                                + x_idx * block_size * x
                                + block_offset * x
                                + x_offset;
    key_cache[tgt_key_idx] = key[src_key_idx];
  }
}

} // namespace vllm

void reshape_and_cache_key(
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& slot_mapping)  // [num_tokens]
{
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    key.scalar_type(),
    "reshape_and_cache_key_kernel",
    [&] {
      vllm::reshape_and_cache_key_kernel<scalar_t><<<grid, block, 0, stream>>>(
        key.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int64_t>(),
        key_stride,
        num_heads,
        head_size,
        block_size,
        x);
    });
}

namespace vllm {

template<typename scalar_t>
__global__ void reshape_and_cache_value_kernel(
  const scalar_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;

    const int64_t tgt_value_idx = block_idx * num_heads * head_size * block_size
                                  + head_idx * head_size * block_size
                                  + head_offset * block_size
                                  + block_offset;
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}

} // namespace vllm

void reshape_and_cache_value(
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  torch::Tensor& value_cache,   // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& slot_mapping)  // [num_tokens]
{
  int num_tokens = value.size(0);
  int num_heads = value.size(1);
  int head_size = value.size(2);
  int block_size = value_cache.size(3);

  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    key.scalar_type(),
    "reshape_and_cache_value_kernel",
    [&] {
      vllm::reshape_and_cache_value_kernel<scalar_t><<<grid, block, 0, stream>>>(
        value.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int64_t>(),
        value_stride,
        num_heads,
        head_size,
        block_size);
    });
}