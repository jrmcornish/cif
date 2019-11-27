import sys
from pathlib import Path
import warnings

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parents[4] / "gitmodules" / "ffjord"))
try:
    from lib.layers.odefunc import ODEnet, ODEfunc
    from lib.layers.cnf import CNF 
finally:
    sys.path.pop(0)

from .bijection import Bijection


class ODEVelocityFunction(ODEnet):
    def __init__(
            self, 
            hidden_dims,
            x_input_shape,
            nonlinearity,
            num_u_channels=0,
            strides=None,
            conv=False,
            layer_type="concatsquash"
    ):
        super().__init__(
            hidden_dims=hidden_dims,
            input_shape=x_input_shape,
            strides=strides,
            conv=conv,
            layer_type=layer_type,
            nonlinearity=nonlinearity
        )

        if num_u_channels > 0:
            # TODO: Make this work for convolutions
            layer_0_class = self.layers[0].__class__
            self.layers[0] = layer_0_class(
                x_input_shape[0] + num_u_channels,
                hidden_dims[0]
            )
        
        self.num_u_channels = num_u_channels

    def set_u(self, u):
        assert self.num_u_channels > 0
        self._u = u

    def forward(self, t, y):
        if self.num_u_channels > 0:
            y = torch.cat((y, self._u), 1)        
        return super().forward(t, y)


class FFJORDBijection(Bijection):
    _VELOCITY_NONLINEARITY = "tanh"
    _DIVERGENCE_METHOD = "brute_force" # TODO: Incorporate approximate
    _SOLVER = "dopri5"
    _INTEGRATION_TIME = 0.5 # TODO: Learn this?

    def __init__(
            self, 
            x_shape,
            velocity_hidden_channels,
            num_u_channels,
            relative_tolerance,
            absolute_tolerance,
    ):
        super().__init__(x_shape=x_shape, z_shape=x_shape)

        self.diffeq = ODEVelocityFunction(
            hidden_dims=tuple(velocity_hidden_channels),
            x_input_shape=x_shape,
            nonlinearity=self._VELOCITY_NONLINEARITY,
            num_u_channels=num_u_channels
        )

        odefunc = ODEfunc(
            diffeq=self.diffeq,
            divergence_fn=self._DIVERGENCE_METHOD,
            residual=False,
            rademacher=False
        )

        self.cnf = CNF(
            odefunc=odefunc,
            T=self._INTEGRATION_TIME,
            train_T=False,
            regularization_fns=None,
            solver=self._SOLVER,
            atol=absolute_tolerance,
            rtol=relative_tolerance
        )

    def _get_nfes(self):
        return torch.tensor(self.cnf.odefunc.num_evals())

    def _evolve_ODE(self, input_state, reverse, **kwargs):
        if "u" in kwargs:
            self.diffeq.set_u(kwargs["u"])

        init_log_jac = input_state.new_zeros(input_state.shape[0], 1)        
        output_state, neg_log_jac = self.cnf(
            input_state,
            init_log_jac,
            reverse=reverse
        )

        return output_state, -neg_log_jac

    def _x_to_z(self, x, **kwargs):
        z, log_jac = self._evolve_ODE(
            input_state=x,
            reverse=False,
            **kwargs
        )
        return {"z": z, "log-jac": log_jac, "nfes": self._get_nfes()}

    def _z_to_x(self, z, **kwargs):
        x, log_jac = self._evolve_ODE(
            input_state=z,
            reverse=True,
            **kwargs
        )
        return {"x": x, "log-jac": log_jac}
