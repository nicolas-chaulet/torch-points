import torch

if torch.cuda.is_available():
    import torch_points_kernels.points_cuda as tpcuda


class GriddingReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, scale):
        if not torch.cuda.is_available():
            raise NotImplementedError("CPU version is not available for Gridding Reverse")

        ptcloud = tpcuda.gridding_reverse(scale, grid)
        ctx.save_for_backward(torch.Tensor([scale]), grid, ptcloud)
        return ptcloud

    @staticmethod
    def backward(ctx, grad_ptcloud):
        scale, grid, ptcloud = ctx.saved_tensors
        scale = int(scale.item())
        grad_grid = tpcuda.gridding_reverse_grad(ptcloud, grid, grad_ptcloud)
        grad_grid = grad_grid.view(-1, scale, scale, scale)
        return grad_grid, None


def gridding_reverse(grid, scale):
    r"""
    Converts the input point clouds into 3D grids by trilinear interpolcation.
    Please refer to https://arxiv.org/pdf/2006.03761 for more information

    Parameters
    ----------
    grid: torch.Tensor (dtype=torch.float32)
        (B, scale, scale, scale): the 3D grid of the resolution of scale * scale * scale
    scale : Int
        the resolution of the 3D grid

    Returns
    -------
    grid: torch.Tensor (dtype=torch.float32)
        (B, n_pts, 3) B point clouds containing n_pts points
    """
    ptcloud = GriddingReverseFunction.apply(grid, scale)
    return ptcloud / scale * 2
