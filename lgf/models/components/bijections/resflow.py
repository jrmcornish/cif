import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[4] / "gitmodules"))
try:
    from residual_flows.lib.layers import iResBlock
    from residual_flows.lib.layers.base import (
        Swish,
        get_linear,
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
            self._build_nnet(dims),
            n_dist="geometric",

            # NOTE: this is the default from train_toy.py. Despite what the iResBlock  # docstring says, this setting uses unbiased estimation of the log Jacobian
            n_power_series=None,

            # NOTE: All settings are the defaults from train_toy.py
            exact_trace=False,
            brute_force=False,
            n_samples=1,
            neumann_grad=False,
            grad_in_forward=False,
        )

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
        modules_to_update = (
            SpectralNormConv2d,
            SpectralNormLinear,
            InducedNormConv2d,
            InducedNormLinear
        )
        for m in self.layer.modules():
            if isinstance(m, modules_to_update):
                m.compute_weight(update=True, n_iterations=num_iterations)

    def _build_nnet(self, dims):
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers += [
                # TODO: Copied from train_toy.py, but should not apply for first layer?
                Swish(),

                get_linear(
                    in_dim,
                    out_dim,

                    # NOTE: settings all taken from defaults in train_toy.py
                    coeff=0.9,
                    domain=2,
                    codomain=2,

                    # NOTE: All set to None because we specify these manually at each call
                    # to compute_weight()
                    n_iterations=None,
                    atol=None,
                    rtol=None,

                    # TODO: Copied from train_toy.py, but should only apply to last layer?
                    zero_init=(out_dim == 2),
                )
            ]

        return torch.nn.Sequential(*layers)
