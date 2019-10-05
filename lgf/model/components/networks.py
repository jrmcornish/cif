import torch
import torch.nn as nn

from .helpers import ScaledTanh2dModule


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = self._get_conv3x3(num_channels)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = self._get_conv3x3(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        out = self.bn1(inputs)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + inputs

        return out

    def _get_conv3x3(self, num_channels):
        return nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )


def get_resnet(
        num_input_channels,
        num_blocks,
        num_hidden_channels_per_block,
        num_output_channels
):
    layers = (
        [
            nn.Conv2d(
                in_channels=num_input_channels,
                out_channels=num_hidden_channels_per_block,
                kernel_size=3,
                padding=1,
                bias=False
            )
        ]
        + [ResidualBlock(num_hidden_channels_per_block) for _ in range(num_blocks)]
        + [
            nn.BatchNorm2d(num_hidden_channels_per_block),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_hidden_channels_per_block,
                out_channels=num_output_channels,
                kernel_size=1
            )
        ]
    )

    return ScaledTanh2dModule(
        module=nn.Sequential(*layers),
        num_channels=num_output_channels
    )


def get_mlp(num_inputs, hidden_units, output_dim, activation, log_softmax_outputs=False):
    layers = []
    prev_dim = num_inputs
    for dim in hidden_units:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(activation())
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, output_dim))

    if log_softmax_outputs:
        layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)
