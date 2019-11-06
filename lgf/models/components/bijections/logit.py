import numpy as np

import torch

from .bijection import Bijection


class LogitBijection(Bijection):
    def __init__(self, x_shape):
        super().__init__(x_shape=x_shape, z_shape=x_shape)

    def _x_to_z(self, x, **kwargs):
        z = torch.log(x) - torch.log(1-x)
        return {"z": z, "log-jac": self._log_jac_x_to_z(x)}

    def _z_to_x(self, z, **kwargs):
        x = torch.sigmoid(z)
        return {"x": x, "log-jac": self._log_jac_z_to_x(x)}

    def _log_jac_x_to_z(self, x):
        log_derivs = -torch.log(x) - torch.log(1-x)
        return log_derivs.flatten(start_dim=1).sum(dim=1, keepdim=True)

    def _log_jac_z_to_x(self, z):
        return -self._log_jac_x_to_z(z)
