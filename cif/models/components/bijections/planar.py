import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bijection import Bijection
from ..networks import get_mlp


def _batch_dot(t1, t2):
    return torch.sum(t1 * t2, dim=-1, keepdim=True)

# NOTE: The notation here follows (Rezende & Mohamed, 2015).
# In particular, u does not refer to the conditioning variable
# used by CIFs.
def planar_map(z, u, w, b):
    assert len(z.shape) == 2

    # XXX
    assert z.shape == u.shape == w.shape

    assert b.shape == (z.shape[0], 1)

    # See A.1 of (Rezende & Mohamed, 2015):

    wT_u = _batch_dot(u, w)
    m = -1 + F.softplus(wT_u)
    u_hat = u + (m - wT_u) / torch.sum(w**2, dim=1, keepdim=True) * w

    # See section 4.1 of (Rezende & Mohamed, 2015) for explanation:

    inner = _batch_dot(z, w) + b

    h = torch.tanh(inner)
    f = z + u_hat * h

    h_prime = 1 - torch.tanh(inner)**2
    psi = h_prime * w
    jac = torch.abs(1 + _batch_dot(psi, u_hat))
    log_jac = torch.log(jac)

    return f, log_jac


class PlanarBijection(Bijection):
    def __init__(self, num_input_channels):
        shape = (num_input_channels,)
        super().__init__(x_shape=shape, z_shape=shape)

        self.u = nn.Parameter(self._xavier_init_vector(num_input_channels))
        self.w = nn.Parameter(self._xavier_init_vector(num_input_channels))
        self.b = nn.Parameter(torch.zeros(1))

    def _xavier_init_vector(self, length):
        a = np.sqrt(6 / (length + 1))
        return torch.empty(length).uniform_(-a, a)

    def _x_to_z(self, x, **kwargs):
        u = self.u.expand(x.shape[0], -1)
        w = self.w.expand(x.shape[0], -1)
        b = self.b.expand(x.shape[0], -1)

        z, log_jac = planar_map(z=x, u=u, w=w, b=b)

        return {"z": z, "log-jac": log_jac}


class ConditionalPlanarBijection(Bijection):
    def __init__(
            self,
            num_input_channels,
            num_u_channels,
            cond_hidden_channels,
            cond_activation
    ):
        shape = (num_input_channels,)
        super().__init__(x_shape=shape, z_shape=shape)

        self.num_input_channels = num_input_channels

        self.net = get_mlp(
            num_input_channels=num_u_channels,
            hidden_channels=cond_hidden_channels,
            num_output_channels=2*num_input_channels + 1,
            activation=cond_activation
        )

    def _x_to_z(self, x, **kwargs):
        planar_u, w, b = self._get_params(kwargs["u"])
        z, log_jac = planar_map(x, planar_u, w, b)
        return {"z": z, "log-jac": log_jac}

    def _get_params(self, u):
        params = self.net(u)

        planar_u = params[:, :self.num_input_channels]
        w = params[:, self.num_input_channels:2*self.num_input_channels]
        b = params[:, 2*self.num_input_channels].view(u.shape[0], 1)

        return planar_u, w, b
