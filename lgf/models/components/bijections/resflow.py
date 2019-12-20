import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[4] / "gitmodules"))
try:
    from residual_flows.lib.layers import iResBlock
    from residual_flows.lib.layers.base import Swish, get_linear
finally:
    sys.path.pop(0)

from .bijection import Bijection


# Constitutes a single block
class ResidualFlowBijection(Bijection):
    def __init__(
            self,
            num_input_channels,
            hidden_channels
    ):
        super().__init__(
            x_shape=(num_input_channels,),
            z_shape=(num_input_channels,)
        )

        dims = [num_input_channels] + hidden_channels + [num_input_channels]
        self.layer = iResBlock(
            self._build_nnet(dims, Swish),
            n_dist="geometric",
            n_power_series=None,
            exact_trace=False,
            brute_force=False,
            n_samples=1,
            neumann_grad=False,
            grad_in_forward=False,
        )

    def _x_to_z(self, x, **kwargs):
        z, neg_log_jac = self.layer(x, x.new_zeros((x.shape[0], 1)))
        return {"z": z, "log-jac": -neg_log_jac}

    def _z_to_x(self, z, **kwargs):
        x, neg_log_jac = self.layer.inverse(z, z.new_zeros((z.shape[0], 1)))
        return {"x": x, "log-jac": -neg_log_jac}

    def _build_nnet(self, dims, activation_fn):
        nnet = []
        domains = codomains = [2.] * 5
        for i, (in_dim, out_dim, domain, codomain) in enumerate(zip(dims[:-1], dims[1:], domains, codomains)):
            nnet.append(activation_fn())
            nnet.append(
                get_linear(
                    in_dim,
                    out_dim,
                    coeff=0.9,
                    n_iterations=10, # NOTE: the default was 5, but this is more stable
                    atol=None,
                    rtol=None,
                    domain=domain,
                    codomain=codomain,
                    zero_init=(out_dim == 2),
                )
            )
        return torch.nn.Sequential(*nnet)
