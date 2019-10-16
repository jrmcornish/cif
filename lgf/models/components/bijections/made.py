import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bijection import Bijection

from ..couplers import SharedCoupler


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


class MADEBijection(Bijection):
    def __init__(
            self,
            num_input_channels,
            hidden_channels,
            activation
    ):
        super().__init__(x_shape=(num_input_channels,), z_shape=(num_input_channels,))

        self.ar_map = self._get_ar_map(
            num_input_channels=num_input_channels,
            hidden_channels=hidden_channels,
            activation=activation
        )

    def _z_to_x(self, z, **kwargs):
        # Important to zero here initially, because torch.empty_like() can (and does)
        # sometimes initialise to nan, which causes problems for our MaskedLinear layers
        x = torch.zeros_like(z)

        for dim in range(z.size(1)):
            result = self.ar_map(x)
            means = result["shift"]
            log_stds = result["log-scale"]

            x[:, dim] = z[:, dim] * torch.exp(log_stds[:, dim]) + means[:, dim]

        return {"x": x, "log-jac": self._log_jac_z_to_x(log_stds)}

    def _x_to_z(self, x, **kwargs):
        result = self.ar_map(x)
        means = result["shift"]
        log_stds = result["log-scale"]

        z = (x - means) * torch.exp(-log_stds)

        return {"z": z, "log-jac": self._log_jac_x_to_z(log_stds)}

    def _log_jac_x_to_z(self, log_stds):
        return -log_stds.sum(dim=-1, keepdim=True)

    def _log_jac_z_to_x(self, log_stds):
        return -self._log_jac_x_to_z(log_stds)

    def _get_ar_map(
            self,
            num_input_channels,
            hidden_channels,
            activation
    ):
        return SharedCoupler(
            shift_log_scale_net=self._get_ar_mlp(
                num_input_channels=num_input_channels,
                hidden_channels=hidden_channels,
                num_outputs_per_input=2,
                activation=activation
            )
        )

    def _get_ar_mlp(
            self,
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
