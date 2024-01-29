#pragma once

#include <torch/extension.h>

#include <map>
#include <vector>


void reshape_and_cache(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);

void reshape_and_cache_key(
  torch::Tensor& key,
  torch::Tensor& key_cache,
  torch::Tensor& slot_mapping);

void reshape_and_cache_value(
  torch::Tensor& value,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);
