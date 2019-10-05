import torch
import torch.nn as nn

from .bijection import Bijection


class AffineCouplingBijection(Bijection):
    def __init__(self, x_shape, coupler):
        super().__init__(x_shape=x_shape, z_shape=x_shape)
        self.coupler = coupler

    def _couple(self, inputs, **kwargs):
        if "u" in kwargs:
            inputs = torch.cat((inputs, kwargs["u"]), dim=1)
        outputs = self.coupler(inputs)
        return outputs["shift"], outputs["scale"]

    def _log_jac_x_to_z(self, scale):
        return scale.flatten(start_dim=1).sum(dim=1, keepdim=True)

    def _log_jac_z_to_x(self, scale):
        return -self._log_jac_x_to_z(scale)


class CheckerboardMasked2dAffineCouplingBijection(AffineCouplingBijection):
    def __init__(
            self,
            x_shape,
            coupler,
            reverse_mask
    ):
        super().__init__(x_shape=x_shape, coupler=coupler)

        assert len(x_shape) == 3
        num_channels, height, width = x_shape

        self.register_buffer("mask", self._checkerboard_mask(num_channels, height, width, reverse_mask))

    def _x_to_z(self, x, **kwargs):
        shift, scale = self._couple(self.mask*x, **kwargs)
        z = self.mask*x + (1-self.mask)*((x + shift) * torch.exp(scale))
        return {"z": z, "log-jac": self._log_jac_x_to_z((1-self.mask)*scale)}

    def _z_to_x(self, z, **kwargs):
        shift, scale = self._couple(self.mask*z, **kwargs)
        x = self.mask*z + (1-self.mask)*(z * torch.exp(-scale) - shift)
        return {"x": x, "log-jac": self._log_jac_z_to_x((1-self.mask)*scale)}

    def _checkerboard_mask(self, num_channels, height, width, reverse_mask):
        mask = torch.empty((height, width))
        for i in range(height):
            for j in range(width):
                mask[i, j] = (i + j) % 2 == 1
        mask = mask.expand(num_channels, -1, -1)

        if reverse_mask:
            mask = 1 - mask

        return mask


class ChannelwiseMaskedAffineCouplingBijection(AffineCouplingBijection):
    def __init__(
            self,
            x_shape,
            coupler,
            mask
    ):
        super().__init__(x_shape=x_shape, coupler=coupler)

        assert torch.any(mask), "Not a bijection without passthrough"

        self.mask = mask

    def _x_to_z(self, x, **kwargs):
        passthrough_x = x[:, self.mask]
        modified_x = x[:, ~self.mask]
        shift, scale = self._couple(passthrough_x, **kwargs)

        z = torch.empty_like(x)
        z[:, self.mask] = passthrough_x
        z[:, ~self.mask] = (modified_x + shift) * torch.exp(scale)

        return {"z": z, "log-jac": self._log_jac_x_to_z(scale)}

    def _z_to_x(self, z, **kwargs):
        passthrough_z = z[:, self.mask]
        modified_z = z[:, ~self.mask]
        shift, scale = self._couple(passthrough_z, **kwargs)

        x = torch.empty_like(z)
        x[:, self.mask] = passthrough_z
        x[:, ~self.mask] = modified_z * torch.exp(-scale) - shift

        return {"x": x, "log-jac": self._log_jac_z_to_x(scale)}
