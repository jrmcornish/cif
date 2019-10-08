import unittest
from unittest.mock import Mock
import random
import warnings

import numpy as np

import torch
import torch.nn as nn

from lgf.models.components.networks import get_resnet, get_glow_cnn


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


if __name__ == "__main__":
    unittest.main()
