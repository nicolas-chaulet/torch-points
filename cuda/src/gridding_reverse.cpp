#include "gridding_reverse.h"
#include "utils.h"

torch::Tensor gridding_reverse(int scale, torch::Tensor grid)
{
    CHECK_CUDA(grid);
    CHECK_CONTIGUOUS(grid);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    return gridding_reverse_kernel_warpper(scale, grid, stream);
}

torch::Tensor gridding_reverse_grad(torch::Tensor ptcloud, torch::Tensor grid,
                                    torch::Tensor grad_ptcloud)
{
    CHECK_CUDA(ptcloud);
    CHECK_CONTIGUOUS(ptcloud);
    CHECK_CUDA(grid);
    CHECK_CONTIGUOUS(grid);
    CHECK_CUDA(grad_ptcloud);
    CHECK_CONTIGUOUS(grad_ptcloud);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    return gridding_reverse_grad_kernel_warpper(ptcloud, grid, grad_ptcloud, stream);
}
