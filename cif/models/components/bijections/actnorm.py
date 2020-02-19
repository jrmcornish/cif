import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[4] / "gitmodules"))
try:
    from residual_flows.lib.layers.act_norm import ActNormNd
finally:
    sys.path.pop(0)

# ActNormNd.shape is a property, but we want to override it
delattr(ActNormNd, "shape")

from .bijection import Bijection

class ActNormBijection(Bijection):
    def __init__(self, x_shape):
        super().__init__(x_shape=x_shape, z_shape=x_shape)

        self.actnorm = ActNormNd(
            num_features=x_shape[0]
        )

        self.actnorm.shape = (1, -1) + ((1,) * len(x_shape[1:]))

    def _x_to_z(self, x, **kwargs):
        z, neg_log_jac = self.actnorm(x=x, logpx=0.)
        return {"z": z, "log-jac": -neg_log_jac}

    def _z_to_x(self, z, **kwargs):
        x, neg_log_jac = self.actnorm.inverse(y=z, logpy=0.)
        return {"x": x, "log-jac": -neg_log_jac}
