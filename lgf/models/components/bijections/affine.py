import numpy as np

import torch
import torch.nn as nn

from .bijection import Bijection


class BatchNormBijection(Bijection):
    def __init__(self, x_shape, per_channel, apply_affine, momentum=0.1, eps=1e-5):
        super().__init__(x_shape=x_shape, z_shape=x_shape)

        self.momentum = momentum
        self.eps = eps

        if per_channel:
            param_shape = (x_shape[0], *[1 for _ in x_shape[1:]])
            self.average_dims = [0] + list(range(2, len(x_shape) + 1))
            self.log_jac_factor = np.prod(x_shape[1:])
        else:
            param_shape = x_shape
            self.average_dims = [0]
            self.log_jac_factor = 1

        self.register_buffer("running_mean", torch.zeros(param_shape))
        self.register_buffer("running_var", torch.ones(param_shape))

        self.apply_affine = apply_affine

        if apply_affine:
            self.shift = nn.Parameter(torch.zeros(param_shape))
            self.log_scale = nn.Parameter(torch.zeros(param_shape))

    def _x_to_z(self, x, **kwargs):
        if self.training:
            mean = self._average(x)
            var = self._average((x - mean)**2)

            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.data)
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.data)

        else:
            mean = self.running_mean
            var = self.running_var

        z = (x - mean) / torch.sqrt(var + self.eps)

        if self.apply_affine:
            z = z * torch.exp(self.log_scale) + self.shift

        return {
            "z": z,
            "log-jac": self._log_jac_x_to_z(var, x.shape[0])
        }

    def _z_to_x(self, z, **kwargs):
        assert not self.training

        if self.apply_affine:
            z = (z - self.shift) * torch.exp(-self.log_scale)

        x = z * torch.sqrt(self.running_var + self.eps) + self.running_mean

        return {
            "x": x,
            "log-jac": -self._log_jac_x_to_z(self.running_var, z.shape[0])
        }

    def _average(self, data):
        # TODO: Maybe can do better by flattening?
        return torch.mean(data, dim=self.average_dims, keepdim=True).squeeze(0)

    def _log_jac_x_to_z(self, var, batch_size):
        summands = -0.5 * torch.log(var + self.eps)

        if self.apply_affine:
            summands = self.log_scale + summands

        log_jac_single = self.log_jac_factor * torch.sum(summands)

        return log_jac_single.view(1, 1).expand(batch_size, 1)


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
