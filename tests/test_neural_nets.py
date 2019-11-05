import unittest
from unittest.mock import Mock
import random
import warnings

import numpy as np

import torch
import torch.nn as nn

from lgf.models.components.networks import get_resnet, get_glow_cnn, get_ar_mlp


class TestGetResnet(unittest.TestCase):
    def test_shape(self):
        in_channels = 10
        in_shape = (in_channels, 5, 2)
        out_channels = 3
        out_shape = (out_channels, 5, 2)

        resnet = get_resnet(
            num_input_channels=in_channels,
            hidden_channels=[10]*5,
            num_output_channels=out_channels
        )

        batch_size = 5
        inputs = torch.randn(batch_size, *in_shape)
        outputs = resnet(inputs)

        self.assertEqual(outputs.shape, (batch_size, *out_shape))


class TestGlowCNN(unittest.TestCase):
    def _test_shape(self, num_input_channels):
        in_channels = num_input_channels
        out_channels = num_input_channels
        hidden_channels = [512] * 2

        in_shape = (in_channels, 32, 38)
        out_shape = in_shape

        glow_cnn = get_glow_cnn(
            num_input_channels=in_channels,
            num_hidden_channels=hidden_channels[0],
            num_output_channels=out_channels
        )

        batch_size = 5
        inputs = torch.randn(batch_size, *in_shape)
        outputs = glow_cnn(inputs)

        self.assertEqual(outputs.shape, (batch_size, *out_shape))

    def test_shape_one_channel(self):
        self._test_shape(num_input_channels=1)

    def test_shape_multi_channel(self):
        self._test_shape(num_input_channels=3)


class TestAutoregressiveMLP(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1000
        self.num_input_channels = 10
        self.num_outputs_per_input = 4
        self.ar_mlp = get_ar_mlp(
            num_input_channels=self.num_input_channels,
            hidden_channels=[11, 12, 10, 14],
            num_outputs_per_input=self.num_outputs_per_input,
            activation=nn.Tanh
        )

    def test_output_format(self):
        x = torch.randn(self.batch_size, self.num_input_channels)
        f_x = self.ar_mlp(x)

        batch_size, dim = f_x.shape
        self.assertEqual(batch_size, self.batch_size)
        self.assertEqual(dim, self.num_outputs_per_input*self.num_input_channels)

    def test_first_coord_autoreg(self):
        x, f_x, y, f_y = self.perturb_inputs(0)

        self.assertTrue((f_x[:, 0] == f_y[:, 0]).all())
        self.assertFalse((f_x[:, 1:] == f_y[:, 1:]).all())

    def test_middle_coords_autoreg(self):
        for coord in range(1, self.num_input_channels - 1):
            self.assert_middle_coord_autoreg(coord)

    def assert_middle_coord_autoreg(self, coord):
        self.assertGreater(coord, 0)

        x, f_x, y, f_y = self.perturb_inputs(coord)

        self.assertTrue((x[:, :coord] == y[:, :coord]).all())
        self.assertTrue((f_x[:, :coord+1] == f_y[:, :coord+1]).all())
        self.assertFalse((f_x[:, coord+1:] == f_y[:, coord+1:]).all())

    def test_last_coord_autoreg(self):
        x, f_x, y, f_y = self.perturb_inputs(-1)

        self.assertTrue((x[:, :-1] == y[:, :-1]).all())
        self.assertTrue((f_x == f_y).all())

    def perturb_inputs(self, coord):
        x = torch.randn(self.batch_size, self.num_input_channels)
        f_x = self.ar_mlp(x)

        noise = torch.zeros_like(x)
        noise[:, coord:] = torch.randn_like(noise[:, coord:])
        y = x + noise
        f_y = self.ar_mlp(y)

        return x, f_x, y, f_y

    def test_no_nans(self):
        inputs = torch.randn(self.batch_size, self.num_input_channels)
        result = self.ar_mlp(inputs)
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
