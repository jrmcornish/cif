import torch
import torch.nn as nn

from .bijection import Bijection


class BatchNormBijection(Bijection):
    def __init__(self, x_shape, momentum=0.1, eps=1e-5):
        super().__init__(x_shape=x_shape, z_shape=x_shape)

        self.momentum = momentum
        self.eps = eps

        self.beta = nn.Parameter(torch.zeros(x_shape))
        self.gamma = nn.Parameter(torch.zeros(x_shape))

        self.register_buffer("running_mean", torch.zeros(x_shape))
        self.register_buffer("running_var", torch.ones(x_shape))

    def _log_jac_x_to_z(self, var, batch_size):
        log_jac_single = torch.sum(self.gamma - 0.5*torch.log(var + self.eps))
        return log_jac_single.view(1, 1).expand(batch_size, 1)

    def _x_to_z(self, x, **kwargs):
        if self.training:
            mean = torch.mean(x, dim=0)
            var = torch.mean((x - mean)**2, dim=0)

            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.data)
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.data)

        else:
            mean = self.running_mean
            var = self.running_var

        z = (x - mean) / torch.sqrt(var + self.eps) * torch.exp(self.gamma) + self.beta
        return {"z": z, "log-jac": self._log_jac_x_to_z(var, x.shape[0])}

    def _z_to_x(self, z, **kwargs):
        assert not self.training
        x = (z - self.beta) * torch.exp(-self.gamma) * torch.sqrt(self.running_var + self.eps) + self.running_mean
        return {"x": x, "log-jac": -self._log_jac_x_to_z(self.running_var, z.shape[0])}


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
