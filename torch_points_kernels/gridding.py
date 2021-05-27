import torch

if torch.cuda.is_available():
    import torch_points_kernels.points_cuda as tpcuda


class GriddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ptcloud, scale):
        if not torch.cuda.is_available():
            raise NotImplementedError("CPU version is not available for Gridding")

        grid, grid_pt_weights, grid_pt_indexes = tpcuda.gridding(
            -scale, scale - 1, -scale, scale - 1, -scale, scale - 1, ptcloud
        )
        # print(grid.size())             # torch.Size(batch_size, n_grid_vertices)
        # print(grid_pt_weights.size())  # torch.Size(batch_size, n_pts, 8, 3)
        # print(grid_pt_indexes.size())  # torch.Size(batch_size, n_pts, 8)
        ctx.save_for_backward(grid_pt_weights, grid_pt_indexes)

        return grid

    @staticmethod
    def backward(ctx, grad_grid):
        grid_pt_weights, grid_pt_indexes = ctx.saved_tensors
        grad_ptcloud = tpcuda.gridding_grad(grid_pt_weights, grid_pt_indexes, grad_grid)
        # print(grad_ptcloud.size())   # torch.Size(batch_size, n_pts, 3)

        return grad_ptcloud, None


def gridding(ptcloud, scale):
    r"""
    Converts the input point clouds into 3D grids by trilinear interpolcation.
    Please refer to https://arxiv.org/pdf/2006.03761 for more information

    Parameters
    ----------
    ptcloud : torch.Tensor (dtype=torch.float32)
        (B, n_pts, 3) B point clouds containing n_pts points
    scale : Int
        the resolution of the 3D grid

    Returns
    -------
    grid: torch.Tensor
        (B, scale, scale, scale): the grid of the resolution of scale * scale * scale
    """
    if len(ptcloud.shape) != 3 or ptcloud.size(2) != 3:
        raise ValueError("The input point cloud should be of size (B, n_pts, 3)")

    ptcloud = ptcloud * scale
    _ptcloud = torch.split(ptcloud, 1, dim=0)
    grids = []
    for p in _ptcloud:
        non_zeros = torch.sum(p, dim=2).ne(0)
        p = p[non_zeros].unsqueeze(dim=0)
        grids.append(GriddingFunction.apply(p, scale))

    return torch.cat(grids, dim=0).contiguous()
