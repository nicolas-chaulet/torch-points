#pragma once

#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

torch::Tensor gridding_reverse_kernel_warpper(int scale, torch::Tensor grid, cudaStream_t stream);

torch::Tensor gridding_reverse_grad_kernel_warpper(torch::Tensor ptcloud, torch::Tensor grid,
                                                   torch::Tensor grad_ptcloud, cudaStream_t stream);

torch::Tensor gridding_reverse(int scale, torch::Tensor grid);

torch::Tensor gridding_reverse_grad(torch::Tensor ptcloud, torch::Tensor grid,
                                    torch::Tensor grad_ptcloud);
