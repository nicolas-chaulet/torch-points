#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> chamfer_dist(torch::Tensor xyz1, torch::Tensor xyz2);

std::vector<torch::Tensor> chamfer_dist_grad(torch::Tensor xyz1, torch::Tensor xyz2,
                                             torch::Tensor idx1, torch::Tensor idx2,
                                             torch::Tensor grad_dist1, torch::Tensor grad_dist2);

std::vector<torch::Tensor> chamfer_dist_kernel_wrapper(torch::Tensor xyz1, torch::Tensor xyz2);

std::vector<torch::Tensor> chamfer_dist_grad_kernel_wrapper(torch::Tensor xyz1, torch::Tensor xyz2,
                                                            torch::Tensor idx1, torch::Tensor idx2,
                                                            torch::Tensor grad_dist1,
                                                            torch::Tensor grad_dist2);
