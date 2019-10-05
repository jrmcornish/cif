import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bijection import Bijection

from ..helpers import SplittingModule


class MaskedLinear(nn.Module):
    def __init__(self, input_degrees, output_degrees):
        super().__init__()

        assert len(input_degrees.shape) == len(output_degrees.shape) == 1

        num_inputs = input_degrees.shape[0]
        num_outputs = output_degrees.shape[0]

        self.linear = nn.Linear(num_inputs, num_outputs)

        mask = output_degrees.view(-1, 1) >= input_degrees
        self.register_buffer("mask", mask.to(self.linear.weight.dtype))

    def forward(self, inputs):
        return F.linear(inputs, self.mask*self.linear.weight, self.linear.bias)


class MADEBijection(Bijection):
    def __init__(
            self,
            num_inputs,
            hidden_units,
            activation
    ):
        super().__init__(x_shape=(num_inputs,), z_shape=(num_inputs,))

        self.ar_map = self._get_ar_map(
            num_inputs=num_inputs,
            hidden_units=hidden_units,
            activation=activation
        )

    def _z_to_x(self, z, **kwargs):
        # Important to zero here initially, because torch.empty_like() can (and does)
        # sometimes initialise to nan, which causes problems for our MaskedLinear layers
        x = torch.zeros_like(z)

        for dim in range(z.size(1)):
            result = self.ar_map(x)
            means = result["mean"]
            log_stds = result["log-std"]

            x[:, dim] = z[:, dim] * torch.exp(log_stds[:, dim]) + means[:, dim]

        return {"x": x, "log-jac": self._log_jac_z_to_x(log_stds)}

    def _x_to_z(self, x, **kwargs):
        result = self.ar_map(x)
        means = result["mean"]
        log_stds = result["log-std"]

        z = (x - means) * torch.exp(-log_stds)

        return {"z": z, "log-jac": self._log_jac_x_to_z(log_stds)}

    def _log_jac_x_to_z(self, log_stds):
        return -log_stds.sum(dim=-1, keepdim=True)

    def _log_jac_z_to_x(self, log_stds):
        return -self._log_jac_x_to_z(log_stds)

    def _get_ar_map(
            self,
            num_inputs,
            hidden_units,
            activation
    ):
        output_names = ["mean", "log-std"]

        return SplittingModule(
            module=self._get_ar_mlp(
                num_inputs=num_inputs,
                hidden_units=hidden_units,
                num_outputs_per_input=len(output_names),
                activation=activation
            ),
            output_names=output_names,
            dim=1
        )

    def _get_ar_mlp(
            self,
            num_inputs,
            hidden_units,
            num_outputs_per_input,
            activation
    ):
        assert num_inputs >= 2
        assert all([num_inputs <= d for d in hidden_units]), "Random initialisation not yet implemented"

        prev_degrees = torch.arange(1, num_inputs + 1, dtype=torch.int64)
        layers = []

        for hidden_units in hidden_units:
            degrees = torch.arange(hidden_units, dtype=torch.int64) % (num_inputs - 1) + 1

            layers.append(MaskedLinear(prev_degrees, degrees))
            layers.append(activation())

            prev_degrees = degrees

        degrees = torch.arange(num_inputs, dtype=torch.int64).repeat(num_outputs_per_input)
        layers.append(MaskedLinear(prev_degrees, degrees))

        return nn.Sequential(*layers)
