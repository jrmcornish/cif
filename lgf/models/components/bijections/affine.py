import numpy as np

import torch
import torch.nn as nn

from .bijection import Bijection


# TODO: Should we parameterise this as x * scale + shift (current setup) or as (x + shift) * scale?
class AffineBijection(Bijection):
    def __init__(self, x_shape, per_channel):
        super().__init__(x_shape=x_shape, z_shape=x_shape)

        if per_channel:
            param_shape = (x_shape[0], *[1 for _ in x_shape[1:]])
            self.log_jac_factor = np.prod(x_shape[1:])
        else:
            param_shape = x_shape
            self.log_jac_factor = 1

        self.shift = nn.Parameter(torch.zeros(param_shape))
        self.log_scale = nn.Parameter(torch.zeros(param_shape))

    def _x_to_z(self, x, **kwargs):
        return {
            "z": x * torch.exp(self.log_scale) + self.shift,
            "log-jac": self._log_jac_x_to_z(x.shape[0])
        }

    def _z_to_x(self, z, **kwargs):
        return {
            "x": (z - self.shift) * torch.exp(-self.log_scale),
            "log-jac": -self._log_jac_x_to_z(z.shape[0])
        }

    def _log_jac_x_to_z(self, batch_size):
        log_jac_single = self.log_jac_factor * torch.sum(self.log_scale)
        return log_jac_single.view(1, 1).expand(batch_size, 1)


# TODO: Potentially we can merge with AffineBijection
# TODO: Potentially a per-channel version of this would do better for images
class ConditionalAffineBijection(Bijection):
    def __init__(
            self,
            x_shape,
            coupler
    ):
        super().__init__(x_shape, x_shape)
        self.coupler = coupler

    def _x_to_z(self, x, **kwargs):
        shift, log_scale = self._shift_log_scale(kwargs["u"])
        z = (x + shift) * torch.exp(log_scale)
        return {"z": z, "log-jac": self._log_jac_x_to_z(log_scale)}

    def _z_to_x(self, z, **kwargs):
        shift, log_scale = self._shift_log_scale(kwargs["u"])
        x = z * torch.exp(-log_scale) - shift
        return {"x": x, "log-jac": self._log_jac_z_to_x(log_scale)}

    def _shift_log_scale(self, u):
        shift_log_scale = self.coupler(u)
        return shift_log_scale["shift"], shift_log_scale["log-scale"]

    def _log_jac_x_to_z(self, log_scale):
        return log_scale.flatten(start_dim=1).sum(dim=1, keepdim=True)

    def _log_jac_z_to_x(self, log_scale):
        return -self._log_jac_x_to_z(log_scale)
