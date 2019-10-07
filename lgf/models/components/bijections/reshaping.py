import numpy as np

import torch

from .bijection import Bijection


class ReshapingBijection(Bijection):
    def __init__(self, x_shape, z_shape):
        assert np.prod(x_shape) == np.prod(z_shape)
        super().__init__(x_shape=x_shape, z_shape=z_shape)

    def _x_to_z(self, x, **kwargs):
        return {"z": self._reshape_x(x), "log-jac": self._log_jac_like(x)}

    def _z_to_x(self, z, **kwargs):
        return {"x": self._reshape_z(z), "log-jac": self._log_jac_like(z)}

    def _log_jac_like(self, inputs):
        return torch.zeros(inputs.shape[0], 1, dtype=inputs.dtype, device=inputs.device)

    def _reshape_x(self, x):
        raise NotImplementedError

    def _reshape_z(self, z):
        raise NotImplementedError


class FlipBijection(ReshapingBijection):
    def __init__(self, x_shape, dim):
        super().__init__(x_shape=x_shape, z_shape=x_shape)
        self.dim = dim

    def _reshape_x(self, x):
        return self._flip(x)

    def _reshape_z(self, z):
        return self._flip(z)

    def _flip(self, inputs):
        return torch.flip(inputs, dims=(self.dim,))


class ViewBijection(ReshapingBijection):
    def _reshape_x(self, x):
        return x.view(x.shape[0], *self.z_shape)

    def _reshape_z(self, z):
        return z.view(z.shape[0], *self.x_shape)


class Squeeze2dBijection(ReshapingBijection):
    def __init__(self, x_shape, factor):
        assert len(x_shape) == 3

        self.x_c, self.x_h, self.x_w = x_shape

        assert self.x_h % factor == 0
        assert self.x_w % factor == 0

        self.z_c = self.x_c * factor**2
        self.z_h = self.x_h // factor
        self.z_w = self.x_w // factor

        super().__init__(x_shape=x_shape, z_shape=(self.z_c, self.z_h, self.z_w))

        self.factor = factor

    # Adapted from https://github.com/chaiyujin/glow-pytorch/blob/487a6b149295f4ec4b36e408f63604c593ff2031/glow/modules.py#L313
    def _reshape_x(self, x, **kwargs):
        batch_size = x.shape[0]

        z = x.view(
            batch_size,
            self.x_c,
            self.x_h // self.factor,
            self.factor,
            self.x_w // self.factor,
            self.factor
        )

        return z.permute(0, 1, 3, 5, 2, 4).reshape(batch_size, *self.z_shape)

    # Adapted from https://github.com/chaiyujin/glow-pytorch/blob/487a6b149295f4ec4b36e408f63604c593ff2031/glow/modules.py#L329
    def _reshape_z(self, z, **kwargs):
        batch_size = z.shape[0]

        x = z.view(
            batch_size,
            self.z_c // self.factor**2,
            self.factor,
            self.factor,
            self.z_h,
            self.z_w
        )

        return x.permute(0, 1, 4, 2, 5, 3).reshape(batch_size, *self.x_shape)
