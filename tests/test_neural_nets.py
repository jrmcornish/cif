import unittest
from unittest.mock import Mock
import random
import warnings

import numpy as np

import torch
import torch.nn as nn

from lgf.models.components.networks import get_resnet, get_glow_cnn, AutoregressiveMLP


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
            num_output_channels=out_channels,
            zero_init_output=True
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
        self.num_output_heads = 4
        self.ar_mlp = AutoregressiveMLP(
            num_input_channels=self.num_input_channels,
            hidden_channels=[11, 12, 10, 14],
            num_output_heads=self.num_output_heads,
            activation=nn.Tanh
        )

    def test_output_format(self):
        x = torch.randn(self.batch_size, self.num_input_channels)
        f_x = self.ar_mlp(x)

        assert f_x.shape == (self.batch_size, self.num_output_heads, self.num_input_channels)

    def test_first_coord_autoreg(self):
        x, f_x, y, f_y = self.perturb_inputs(0)

        for i in range(self.num_output_heads):
            assert (f_x[:, i, 0] == f_y[:, i, 0]).all()

    def test_middle_coords_autoreg(self):
        for coord in range(1, self.num_input_channels - 1):
            self.assert_middle_coord_autoreg(coord)

    def assert_middle_coord_autoreg(self, coord):
        assert coord > 0

        x, f_x, y, f_y = self.perturb_inputs(coord)

        for i in range(self.num_output_heads):
            assert (x[:, :coord] == y[:, :coord]).all()
            assert (f_x[:, i, :coord+1] == f_y[:, i, :coord+1]).all()

    def test_last_coord_autoreg(self):
        x, f_x, y, f_y = self.perturb_inputs(-1)

        assert (x[:, :-1] == y[:, :-1]).all()
        assert (f_x == f_y).all()

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
        assert torch.isfinite(result).all()


if __name__ == "__main__":
    unittest.main()
