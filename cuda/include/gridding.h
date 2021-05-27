#pragma once

#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

std::vector<torch::Tensor> gridding_kernel_warpper(float min_x, float max_x, float min_y,
                                                   float max_y, float min_z, float max_z,
                                                   torch::Tensor ptcloud, cudaStream_t stream);

torch::Tensor gridding_grad_kernel_warpper(torch::Tensor grid_pt_weights,
                                           torch::Tensor grid_pt_indexes, torch::Tensor grad_grid,
                                           cudaStream_t stream);

std::vector<torch::Tensor> gridding(float min_x, float max_x, float min_y, float max_y, float min_z,
                                    float max_z, torch::Tensor ptcloud);

torch::Tensor gridding_grad(torch::Tensor grid_pt_weights, torch::Tensor grid_pt_indexes,
                            torch::Tensor grad_grid);
