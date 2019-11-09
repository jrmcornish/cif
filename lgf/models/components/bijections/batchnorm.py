import numpy as np

import torch
import torch.nn as nn

from .bijection import Bijection


class BatchNormBijection(Bijection):
    def __init__(self, x_shape, per_channel, apply_affine, momentum, eps=1e-5):
        super().__init__(x_shape=x_shape, z_shape=x_shape)

        assert 0 <= momentum <= 1
        self.momentum = momentum

        assert eps > 0
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

            if self.momentum == 1:
                self.running_mean.copy_(mean)
                self.running_var.copy_(var)

            elif self.momentum > 0:
                # TODO: Should raise an exception or do something here if we get a nan
                # since this will propagate
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
