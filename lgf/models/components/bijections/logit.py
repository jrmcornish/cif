import numpy as np

import torch

from .bijection import Bijection


class LogitTransformBijection(Bijection):
    def __init__(self, x_shape, lam, scale):
        super().__init__(x_shape=x_shape, z_shape=x_shape)
        self.lam = lam
        self.scale = scale

    def _x_to_z(self, x, **kwargs):
        p = self.lam + (1 - 2*self.lam)*x/self.scale
        z = torch.log(p) - torch.log(1-p)
        return {"z": z, "log-jac": self._log_jac_x_to_z(p)}

    def _z_to_x(self, z, **kwargs):
        p = torch.sigmoid(z)
        x = self.scale * (p - self.lam) / (1 - 2*self.lam)
        return {"x": x, "log-jac": self._log_jac_z_to_x(p)}

    def _log_jac_x_to_z(self, p):
        log_derivs = -torch.log(p) - torch.log(1-p) + np.log(1 - 2*self.lam) - np.log(self.scale)
        return log_derivs.flatten(start_dim=1).sum(dim=1, keepdim=True)

    def _log_jac_z_to_x(self, p):
        return -self._log_jac_x_to_z(p)
