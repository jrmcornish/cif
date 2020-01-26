import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[3] / "gitmodules"))
try:
    from residual_flows.lib.layers.base import (
        Swish,
        InducedNormLinear,
        InducedNormConv2d
    )
finally:
    sys.path.pop(0)


class ConstantNetwork(nn.Module):
    def __init__(self, value, fixed):
        super().__init__()
        if fixed:
            self.register_buffer("value", value)
        else:
            self.value = nn.Parameter(value)

    def forward(self, inputs):
        return self.value.expand(inputs.shape[0], *self.value.shape)


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv1 = self._get_conv3x3(num_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv2 = self._get_conv3x3(num_channels, bias=True)

    def forward(self, inputs):
        out = self.bn1(inputs)
        out = torch.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)

        out = out + inputs

        return out

    def _get_conv3x3(self, num_channels, bias):
        return nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )


class ScaledTanh2dModule(nn.Module):
    def __init__(self, module, num_channels):
        super().__init__()
        self.module = module
        self.weights = nn.Parameter(torch.ones(num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_channels, 1, 1))

    def forward(self, inputs):
        out = self.module(inputs)
        out = self.weights * torch.tanh(out) + self.bias
        return out


def get_resnet(
        num_input_channels,
        hidden_channels,
        num_output_channels
):
    num_hidden_channels = hidden_channels[0] if hidden_channels else num_output_channels

    layers = [
        nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=num_hidden_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
    ]

    for num_hidden_channels in hidden_channels:
        layers.append(ResidualBlock(num_hidden_channels))

    layers += [
        nn.BatchNorm2d(num_hidden_channels),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=num_hidden_channels,
            out_channels=num_output_channels,
            kernel_size=1
        )
    ]

    return ScaledTanh2dModule(
        module=nn.Sequential(*layers),
        num_channels=num_output_channels
    )


def get_glow_cnn(num_input_channels, num_hidden_channels, num_output_channels):
    conv1 = nn.Conv2d(
        in_channels=num_input_channels,
        out_channels=num_hidden_channels,
        kernel_size=3,
        padding=1,
        bias=False
    )

    bn1 = nn.BatchNorm2d(num_hidden_channels)

    conv2 = nn.Conv2d(
        in_channels=num_hidden_channels,
        out_channels=num_hidden_channels,
        kernel_size=1,
        padding=0,
        bias=False
    )

    bn2 = nn.BatchNorm2d(num_hidden_channels)

    conv3 = nn.Conv2d(
        in_channels=num_hidden_channels,
        out_channels=num_output_channels,
        kernel_size=3,
        padding=1
    )
    conv3.weight.data.zero_()
    conv3.bias.data.zero_()

    relu = nn.ReLU()

    return nn.Sequential(conv1, bn1, relu, conv2, bn2, relu, conv3)


def get_mlp(
        num_input_channels,
        hidden_channels,
        num_output_channels,
        activation,
        log_softmax_outputs=False
):
    layers = []
    prev_num_hidden_channels = num_input_channels
    for num_hidden_channels in hidden_channels:
        layers.append(nn.Linear(prev_num_hidden_channels, num_hidden_channels))
        layers.append(activation())
        prev_num_hidden_channels = num_hidden_channels
    layers.append(nn.Linear(prev_num_hidden_channels, num_output_channels))

    if log_softmax_outputs:
        layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


class MaskedLinear(nn.Module):
    def __init__(self, input_degrees, output_degrees):
        super().__init__()

        assert len(input_degrees.shape) == len(output_degrees.shape) == 1

        num_input_channels = input_degrees.shape[0]
        num_output_channels = output_degrees.shape[0]

        self.linear = nn.Linear(num_input_channels, num_output_channels)

        mask = output_degrees.view(-1, 1) >= input_degrees
        self.register_buffer("mask", mask.to(self.linear.weight.dtype))

    def forward(self, inputs):
        return F.linear(inputs, self.mask*self.linear.weight, self.linear.bias)


class AutoregressiveMLP(nn.Module):
    def __init__(
            self,
            num_input_channels,
            hidden_channels,
            num_output_heads,
            activation
    ):
        super().__init__()
        self.flat_ar_mlp = self._get_flat_ar_mlp(num_input_channels, hidden_channels, num_output_heads, activation)
        self.num_input_channels = num_input_channels
        self.num_output_heads = num_output_heads

    def _get_flat_ar_mlp(
            self,
            num_input_channels,
            hidden_channels,
            num_output_heads,
            activation
    ):
        assert num_input_channels >= 2
        assert all([num_input_channels <= d for d in hidden_channels]), "Random initialisation not yet implemented"

        prev_degrees = torch.arange(1, num_input_channels + 1, dtype=torch.int64)
        layers = []

        for hidden_channels in hidden_channels:
            degrees = torch.arange(hidden_channels, dtype=torch.int64) % (num_input_channels - 1) + 1

            layers.append(MaskedLinear(prev_degrees, degrees))
            layers.append(activation())

            prev_degrees = degrees

        degrees = torch.arange(num_input_channels, dtype=torch.int64).repeat(num_output_heads)
        layers.append(MaskedLinear(prev_degrees, degrees))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        assert inputs.shape[1:] == (self.num_input_channels,)
        result = self.flat_ar_mlp(inputs)
        result = result.view(inputs.shape[0], self.num_output_heads, self.num_input_channels)
        return result


def get_lipschitz_mlp(
        num_input_channels,
        hidden_channels,
        num_output_channels,
        lipschitz_constant
):
    layers = []
    prev_num_channels = num_input_channels
    for i, num_channels in enumerate(hidden_channels + [num_output_channels]):
        layers += [
            Swish(),
            _get_lipschitz_linear_layer(
                num_input_channels=prev_num_channels,
                num_output_channels=num_channels,
                lipschitz_constant=lipschitz_constant,

                # Zero the weight matrix of the final layer. Done to align with
                # `train_toy.py`.
                zero_init=(i == len(hidden_channels))
            )
        ]

        prev_num_channels = num_channels

    return nn.Sequential(*layers)


def _get_lipschitz_linear_layer(
        num_input_channels,
        num_output_channels,
        lipschitz_constant,
        zero_init
):
    return InducedNormLinear(
        in_features=num_input_channels,
        out_features=num_output_channels,

        # Corresponds to kappa in "Residual Flows" paper or c in original iResNet paper
        coeff=lipschitz_constant,

        # p-norms to use for the domain and codomain when enforcing Lipschitz constraint.
        # We set these to 2 for simplicity in line with the discussion in Appendix D of 
        # ResFlows paper.
        domain=2,
        codomain=2,

        # Parameters to determine number of power iterations used when estimating the
        # Lipschitz constant. These can all be set directly when calling `compute_weight`,
        # so we make None here.
        n_iterations=None,
        atol=None,
        rtol=None,

        # (Approximately) zeros the weight matrix
        zero_init=zero_init
    )


def get_lipschitz_cnn(
        num_input_channels,
        num_hidden_channels,
        num_output_channels,
        lipschitz_constant
):
    conv1 = _get_lipschitz_conv_layer(
        num_input_channels=num_input_channels,
        num_output_channels=num_hidden_channels,
        kernel_size=3,
        padding=1,
        lipschitz_constant=lipschitz_constant
    )

    conv2 = _get_lipschitz_conv_layer(
        num_input_channels=num_hidden_channels,
        num_output_channels=num_hidden_channels,
        kernel_size=1,
        padding=0,
        lipschitz_constant=lipschitz_constant
    )

    conv3 = _get_lipschitz_conv_layer(
        num_input_channels=num_hidden_channels,
        num_output_channels=num_output_channels,
        kernel_size=3,
        padding=1,
        lipschitz_constant=lipschitz_constant
    )

    lipswish = Swish()

    return nn.Sequential(lipswish, conv1, lipswish, conv2, lipswish, conv3)


def _get_lipschitz_conv_layer(
        num_input_channels,
        num_output_channels,
        kernel_size,
        padding,
        lipschitz_constant
):
    return InducedNormConv2d(
        in_channels=num_input_channels,
        out_channels=num_hidden_channels,

        kernel_size=kernel_size,
        stride=1,
        padding=padding,

        # We always add bias since we don't use batch norm in our Lipschitz CNNs
        bias=True,

        coeff=lipschitz_constant,

        # See note in `_get_lipschitz_linear_layer`
        domain=2,
        codomain=2,

        # See note in `_get_lipschitz_linear_layer`
        n_iterations=None,
        atol=None,
        rtol=None
    )
