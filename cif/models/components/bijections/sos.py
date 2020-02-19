import numpy as np

import torch
import torch.nn as nn

from pyro.distributions.transforms.polynomial import Polynomial
from pyro.nn import AutoRegressiveNN

from .bijection import Bijection


class SumOfSquaresPolynomialBijection(Bijection):
    def __init__(
            self,
            num_input_channels,
            hidden_channels,
            activation,
            num_polynomials,
            polynomial_degree,
    ):
        super().__init__(x_shape=(num_input_channels,), z_shape=(num_input_channels,))

        arn = AutoRegressiveNN(
            input_dim=int(num_input_channels),
            hidden_dims=hidden_channels,
            param_dims=[(polynomial_degree + 1)*num_polynomials],
            nonlinearity=activation()
        )

        self.flow = Polynomial(
            autoregressive_nn=arn,
            input_dim=int(num_input_channels),
            count_degree=polynomial_degree,
            count_sum=num_polynomials
        )

    def _x_to_z(self, x):
        z = self.flow._call(x)
        log_jac = self.flow.log_abs_det_jacobian(None, None).view(x.shape[0], 1)
        return {
            "z": z,
            "log-jac": log_jac
        }
