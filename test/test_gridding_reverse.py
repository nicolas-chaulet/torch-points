import numpy as np
import os
import sys
import torch
import unittest

from torch.autograd import gradcheck

from . import run_if_cuda


ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)

from torch_points_kernels.gridding_reverse import GriddingReverseFunction


class TestGridding(unittest.TestCase):
    @run_if_cuda
    def test_gridding_reverse_function_r8(self):
        x = torch.rand(4, 8, 8, 8)
        x.requires_grad = True
        self.assertTrue(gradcheck(GriddingReverseFunction.apply, [x.double().cuda(), 8]))
