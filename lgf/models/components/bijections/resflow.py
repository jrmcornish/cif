import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[4] / "gitmodules"))
try:
    from residual_flows.lib.layers import iResBlock
finally:
    sys.path.pop(0)

from .bijection import Bijection


# Constitutes a single block
class ResidualFlowBijection(Bijection):
    def __init__(
            self,
            x_shape,
            lipschitz_net,
            reduce_memory
    ):
        super().__init__(x_shape=x_shape, z_shape=x_shape)

        self.block = self._get_iresblock(net=lipschitz_net, reduce_memory=reduce_memory)

    def _x_to_z(self, x, **kwargs):
        z, neg_log_jac = self.block(x=x, logpx=0.)
        return {"z": z, "log-jac": -neg_log_jac}

    def _z_to_x(self, z, **kwargs):
        x, neg_log_jac = self.block.inverse(y=z, logpy=0.)
        return {"x": x, "log-jac": -neg_log_jac}

    def _get_iresblock(self, net, reduce_memory):
        return iResBlock(
            nnet=net,

            # NOTE: All settings below are the defaults from train_toy.py

            # Compute the log Jacobian determinant directly via brute force.
            # Only implemented for 2D inputs.
            brute_force=False,

            # If not None, uses a truncated approximation (as in the original iResNet
            # paper) to the log Jacobian determinant with `n_power_series` terms
            n_power_series=None,

            # Use the Neumann series estimator for the gradient of the log Jacobian (8)
            # instead of the estimator (6). Reduces memory, but may give different
            # behaviour because not equivalent to the standard estimator.
            neumann_grad=reduce_memory,

            # Distribution of N to use in Russian Roulette estimator (6) or (8). Can be
            # either "geometric" or "poisson"
            n_dist="geometric",

            # Parameter of N if n_dist == "geometric"
            geom_p=0.5,

            # Parameter of N if n_dist == "poisson". Ignored since
            # `n_dist == "geometric"`
            lamb=-1.,

            # Shifts the distribution of N in (6) or (8) to the right by the amount
            # specified. This means more terms in the expectations are computed exactly,
            # which reduces variance, but with more computational cost.
            n_exact_terms=2,

            # Number of Monte Carlo samples of (n, v) to use when estimating (6) or (8)
            n_samples=1,

            # Estimate the log Jacobian determinant using the exact trace, rather than
            # the Hutchinson's estimator as in (6) or (8)
            exact_trace=False,

            # Use the decomposition of the gradient of the loss (9) to allow computing
            # gradients during the forward pass in order to save memory. Should not
            # change behaviour of algorithm.
            grad_in_forward=reduce_memory
        )
