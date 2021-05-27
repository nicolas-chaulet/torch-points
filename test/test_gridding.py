import numpy as np
import os
import sys
import torch
import unittest

from torch.autograd import gradcheck

from . import run_if_cuda


ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points_kernels.gridding import GriddingFunction


class TestGridding(unittest.TestCase):
    @run_if_cuda
    def test_gridding_function_32pts(self):
        x = torch.rand(1, 32, 3)
        x.requires_grad = True
        self.assertTrue(gradcheck(GriddingFunction.apply, [x.double().cuda(), 4]))

    @run_if_cuda
    def test_gridding_function_64pts(self):
        x = torch.rand(1, 64, 3)
        x.requires_grad = True
        self.assertTrue(gradcheck(GriddingFunction.apply, [x.double().cuda(), 8]))
