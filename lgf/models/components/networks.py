import torch
import torch.nn as nn
import torch.nn.functional as F


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


# TODO: Add a bias
class ScaledTanh2dModule(nn.Module):
    def __init__(self, module, num_channels):
        super().__init__()
        self.module = module
        self.weights = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, inputs):
        out = self.module(inputs)
        out = self.weights * torch.tanh(out)
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


def get_ar_mlp(
        num_input_channels,
        hidden_channels,
        num_outputs_per_input,
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

    degrees = torch.arange(num_input_channels, dtype=torch.int64).repeat(num_outputs_per_input)
    layers.append(MaskedLinear(prev_degrees, degrees))

    return nn.Sequential(*layers)
