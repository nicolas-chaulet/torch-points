#pragma once

#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

std::vector<torch::Tensor> cubic_feature_sampling(int scale, int neighborhood_size,
                                                  torch::Tensor ptcloud,
                                                  torch::Tensor cubic_features);

std::vector<torch::Tensor> cubic_feature_sampling_grad(int scale, int neighborhood_size,
                                                       torch::Tensor grad_point_features,
                                                       torch::Tensor grid_pt_indexes);

std::vector<torch::Tensor> cubic_feature_sampling_kernel_wrapper(int scale, int neighborhood_size,
                                                                 torch::Tensor ptcloud,
                                                                 torch::Tensor cubic_features,
                                                                 cudaStream_t stream);

std::vector<torch::Tensor>
cubic_feature_sampling_grad_kernel_wrapper(int scale, int neighborhood_size,
                                           torch::Tensor grad_point_features,
                                           torch::Tensor grid_pt_indexes, cudaStream_t stream);
