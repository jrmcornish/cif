import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "gitmodules" / "nsf"))
try:
    from nde.transforms import LULinear
finally:
    sys.path.pop(0)

from .bijection import Bijection


class LULinearBijection(Bijection):
    def __init__(self, num_input_channels):
        shape = (num_input_channels,)
        super().__init__(x_shape=shape, z_shape=shape)

        self.linear = LULinear(
            features=num_input_channels,
            identity_init=True
        )

    def _x_to_z(self, x, **kwargs):
        z, log_jac = self.linear(x)
        return {
            "z": z,
            "log-jac": log_jac.view(x.shape[0], 1)
        }

    def _z_to_x(self, z, **kwargs):
        x, log_jac = self.linear(z)
        return {
            "x": x,
            "log-jac": log_jac.view(z.shape[0], 1)
        }
