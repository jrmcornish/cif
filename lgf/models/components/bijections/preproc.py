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


class ScalarMultiplicationBijection(Bijection):
    def __init__(self, x_shape, value):
        assert np.isscalar(value)
        assert value != 0., "Scalar multiplication by zero is not a bijection"

        super().__init__(x_shape=x_shape, z_shape=x_shape)

        self.value = value
        self.dim = np.prod(x_shape)

    def _x_to_z(self, x, **kwargs):
        return {
            "z": self.value * x,
            "log-jac": self._log_jac_x_to_z(x)
        }

    def _z_to_x(self, z, **kwargs):
        return {
            "x": z / self.value,
            "log-jac": self._log_jac_z_to_x(z)
        }

    def _log_jac_x_to_z(self, x):
        return torch.full(
            size=(x.shape[0], 1),
            fill_value=self.dim*np.log(np.abs(self.value)),
            dtype=x.dtype,
            device=x.device
        )

    def _log_jac_z_to_x(self, z):
        return -self._log_jac_x_to_z(z)


class ScalarAdditionBijection(Bijection):
    def __init__(self, x_shape, value):
        assert np.isscalar(value)

        super().__init__(x_shape=x_shape, z_shape=x_shape)

        self.value = value

    def _x_to_z(self, x, **kwargs):
        return {
            "z": x + self.value,
            "log-jac": self._log_jac_like(x)
        }

    def _z_to_x(self, z, **kwargs):
        return {
            "x": z - self.value,
            "log-jac": self._log_jac_like(z)
        }

    def _log_jac_like(self, inputs):
        return torch.zeros(
            size=(inputs.shape[0], 1),
            dtype=inputs.dtype,
            device=inputs.device
        )
