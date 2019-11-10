import numpy as np

import torch
import torch.nn.functional as F

from .bijection import Bijection


class ElementwiseBijection(Bijection):
    def __init__(self, x_shape):
        super().__init__(x_shape=x_shape, z_shape=x_shape)

    def _x_to_z(self, x, **kwargs):
        return {
            "z": self._F(x),
            "log-jac": self._log_jac_x_to_z(x)
        }

    def _z_to_x(self, z, **kwargs):
        return {
            "x": self._F_inv(z),
            "log-jac": self._log_jac_z_to_x(z)
        }

    def _log_jac_x_to_z(self, x):
        return self._log_dF(x).flatten(start_dim=1).sum(dim=1, keepdim=True)

    def _log_jac_z_to_x(self, z):
        return -self._log_jac_x_to_z(z)

    def _F(self, x):
        raise NotImplementedError

    def _F_inv(self, z):
        raise NotImplementedError

    def _log_dF(self, x):
        raise NotImplementedError


class LogitBijection(ElementwiseBijection):
    _EPS = 1e-7

    def _F(self, x):
        # TODO: Unstable
        return torch.log(x) - torch.log(1-x)

    def _F_inv(self, z):
        return torch.sigmoid(z)

    def _log_dF(self, x):
        x_clamped = x.clamp(self._EPS, 1 - self._EPS)
        return -torch.log(x_clamped) - torch.log(1 - x_clamped)


class TanhBijection(ElementwiseBijection):
    _EPS = 1e-7

    def _F(self, x):
        return torch.tanh(x)

    def _F_inv(self, z):
        # TODO: Unclear whether this is the best way to stabilise
        z_clamped = z.clamp(-1 + self._EPS, 1 - self._EPS)
        return .5 * (torch.log(1 + z_clamped)  - torch.log(1 - z_clamped))

    def _log_dF(self, x):
        return y - 2*F.softplus(y) + np.log(4)


class ScalarMultiplicationBijection(ElementwiseBijection):
    def __init__(self, x_shape, value):
        assert np.isscalar(value)
        assert value != 0., "Scalar multiplication by zero is not a bijection"

        super().__init__(x_shape=x_shape)

        self.value = value

    def _F(self, x):
        return self.value * x

    def _F_inv(self, z):
        return z / self.value

    def _log_dF(self, x):
        return torch.full_like(x, np.log(np.abs(self.value)))


class ScalarAdditionBijection(ElementwiseBijection):
    def __init__(self, x_shape, value):
        assert np.isscalar(value)

        super().__init__(x_shape=x_shape)

        self.value = value

    def _F(self, x):
        return x + self.value

    def _F_inv(self, z):
        return z - self.value

    def _log_dF(self, x):
        return torch.zeros_like(x)
