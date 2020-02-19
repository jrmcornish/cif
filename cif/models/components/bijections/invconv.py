import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .bijection import Bijection


class Invertible1x1ConvBijection(Bijection):
    def __init__(self, x_shape, num_u_channels=0):
        assert len(x_shape) == 1 or len(x_shape) == 3
        super().__init__(x_shape, x_shape)

        num_channels = x_shape[0]
        self.weight_shape = [num_channels, num_channels]

        self.conv_weights_shape = self.weight_shape + [1 for _ in x_shape[1:]]
        self.num_non_channel_elements = np.prod(x_shape[1:])

        self.weights_init = torch.qr(torch.randn(*self.weight_shape))[0]

        self.num_u_channels = num_u_channels
        if num_u_channels > 0:
            self.u_weights = nn.Parameter(torch.zeros(num_channels, num_u_channels))
            self.u_conv_weights_shape = [num_channels, num_u_channels] + [1 for _ in x_shape[1:]]
        
    def _convolve(self, inputs, weights, weights_shape):
        if len(weights_shape) < 3:
            return torch.matmul(inputs, weights.t())
        else:
            return F.conv2d(inputs, weights.view(*weights_shape))

    def _log_jac_single(self):
        raise NotImplementedError

    def _get_weights(self):
        raise NotImplementedError

    def _get_Vu(self, **kwargs):
        if "u" in kwargs:
            Vu = self._convolve(kwargs["u"], self.u_weights, self.u_conv_weights_shape)
        else:
            Vu = 0
            assert self.num_u_channels == 0
        return Vu

    def _x_to_z(self, x, **kwargs):
        Vu = self._get_Vu(**kwargs)
        Wx = self._convolve(x, self._get_weights(), self.conv_weights_shape)
        z = Wx + Vu

        log_jac = self._log_jac_single().expand(x.shape[0], 1)
        return {"z": z, "log-jac": log_jac}

    def _z_to_x(self, z, **kwargs):
        Vu = self._get_Vu(**kwargs)
        x = self._convolve(z - Vu, torch.inverse(self._get_weights()), self.conv_weights_shape)

        neg_log_jac = self._log_jac_single().expand(z.shape[0], 1)
        return {"x": x, "log-jac": -neg_log_jac}


class BruteForceInvertible1x1ConvBijection(Invertible1x1ConvBijection):
    def __init__(self, x_shape, num_u_channels=0):
        super().__init__(x_shape, num_u_channels)
        self.weights = nn.Parameter(self.weights_init)

    def _get_weights(self):
        return self.weights

    def _log_jac_single(self):
        return torch.slogdet(self.weights)[1] * self.num_non_channel_elements


class LUInvertible1x1ConvBijection(Invertible1x1ConvBijection):
    def __init__(self, x_shape, num_u_channels=0):
        super().__init__(x_shape, num_u_channels)

        P, lower, upper = torch.lu_unpack(*torch.lu(self.weights_init))
        s = torch.diag(upper)
        log_s = torch.log(torch.abs(s))
        upper = torch.triu(upper, 1)

        self.register_buffer('P', P)
        self.register_buffer('sign_s', torch.sign(s))
        self.register_buffer('l_mask', torch.tril(torch.ones(self.weight_shape), -1))
        self.register_buffer('eye', torch.eye(*self.weight_shape))

        self.lower = nn.Parameter(lower)
        self.log_s = nn.Parameter(log_s)
        self.upper = nn.Parameter(upper)

        self.bias = nn.Parameter(torch.zeros(x_shape[0], *x_shape[1:]))

    def _get_weights(self):
        L = self.lower * self.l_mask + self.eye
        U = self.upper * self.l_mask.transpose(0, 1).contiguous()
        U += torch.diag(self.sign_s * torch.exp(self.log_s))

        W = torch.matmul(self.P, torch.matmul(L, U))
        return W

    def _log_jac_single(self):
        return torch.sum(self.log_s) * self.num_non_channel_elements
