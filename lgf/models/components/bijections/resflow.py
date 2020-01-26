import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[4] / "gitmodules"))
try:
    from residual_flows.lib.layers import iResBlock
    from residual_flows.lib.layers.base import (
        SpectralNormConv2d,
        SpectralNormLinear,
        InducedNormConv2d,
        InducedNormLinear
    )
finally:
    sys.path.pop(0)

from .bijection import Bijection


# Constitutes a single block
class ResidualFlowBijection(Bijection):
    def __init__(self, num_input_channels, net):
        super().__init__(
            x_shape=(num_input_channels,),
            z_shape=(num_input_channels,)
        )

        self.layer = self._get_iresblock(net=net)

        self.register_forward_pre_hook(self._update_lipschitz_forward_hook)
        self.register_backward_hook(self._queue_lipschitz_update_backward_hook)

        self._requires_train_lipschitz_update = True
        self._requires_eval_lipschitz_update = True

    def _x_to_z(self, x, **kwargs):
        z, neg_log_jac = self.layer(x, x.new_zeros((x.shape[0], 1)))
        return {"z": z, "log-jac": -neg_log_jac}

    def _z_to_x(self, z, **kwargs):
        x, neg_log_jac = self.layer.inverse(z, z.new_zeros((z.shape[0], 1)))
        return {"x": x, "log-jac": -neg_log_jac}

    def _queue_lipschitz_update_backward_hook(self, *args, **kwargs):
        self._requires_train_lipschitz_update = True
        self._requires_eval_lipschitz_update = True

    def _update_lipschitz_forward_hook(self, *args, **kwargs):
        # NOTE: Numbers of iterations from defaults in train_toy.py

        if self.training:
            if self._requires_train_lipschitz_update:
                self._update_lipschitz(num_iterations=5)
                self._requires_train_lipschitz_update = False

        else:
            if self._requires_eval_lipschitz_update:
                self._update_lipschitz(num_iterations=200)
                self._requires_eval_lipschitz_update = False
                self._requires_train_lipschitz_update = False

    def _update_lipschitz(self, num_iterations):
        modules_to_update = (InducedNormConv2d, InducedNormLinear)

        for m in self.layer.modules():
            # TODO: Would be better to have the net inside self.layer do this itself
            if isinstance(m, modules_to_update):
                m.compute_weight(
                    # Updates the estimate of the operator norm, i.e.
                    # \tilde{\sigma}_i in (2) in the original iResNet paper
                    update=True,

                    # Maximum number of power iterations to use. Must specify
                    # this or `(atol, rtol)` below.
                    n_iterations=num_iterations,

                    # Tolerances to use for adaptive number of power iterations
                    # as described in Appendix E of ResFlow paper.
                    atol=None,
                    rtol=None

                    # NOTE: If both `n_iterations` and `(atol, rtol)` are specified,
                    # then power iterations are stopped when the first condition is met.
                )

    def _get_iresblock(self, net):
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
            neumann_grad=False,

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
            grad_in_forward=False,
        )
