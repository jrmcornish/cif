import numpy as np

import torch
import torch.nn as nn

from .bijection import Bijection

from ..couplers import IndexedSharedCoupler
from ..networks import AutoregressiveMLP


class MADEBijection(Bijection):
    def __init__(
            self,
            num_input_channels,
            hidden_channels,
            activation
    ):
        super().__init__(x_shape=(num_input_channels,), z_shape=(num_input_channels,))

        self.ar_coupler = self._get_ar_coupler(
            num_input_channels=num_input_channels,
            hidden_channels=hidden_channels,
            activation=activation
        )

    def _z_to_x(self, z, **kwargs):
        # Important to zero here initially, because torch.empty_like() can (and does)
        # sometimes initialise to nan, which causes problems for our MaskedLinear layers
        x = torch.zeros_like(z)

        for dim in range(z.size(1)):
            result = self.ar_coupler(x)
            means = result["shift"]
            log_stds = result["log-scale"]

            x[:, dim] = z[:, dim] * torch.exp(log_stds[:, dim]) + means[:, dim]

        return {"x": x, "log-jac": self._log_jac_z_to_x(log_stds)}

    def _x_to_z(self, x, **kwargs):
        result = self.ar_coupler(x)
        means = result["shift"]
        log_stds = result["log-scale"]

        z = (x - means) * torch.exp(-log_stds)

        return {"z": z, "log-jac": self._log_jac_x_to_z(log_stds)}

    def _log_jac_x_to_z(self, log_stds):
        return -log_stds.sum(dim=-1, keepdim=True)

    def _log_jac_z_to_x(self, log_stds):
        return -self._log_jac_x_to_z(log_stds)

    def _get_ar_coupler(
            self,
            num_input_channels,
            hidden_channels,
            activation
    ):
        return IndexedSharedCoupler(
            shift_log_scale_net=AutoregressiveMLP(
                num_input_channels=num_input_channels,
                hidden_channels=hidden_channels,
                num_output_heads=2,
                activation=activation
            )
        )
