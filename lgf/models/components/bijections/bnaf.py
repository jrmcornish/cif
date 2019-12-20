from collections import namedtuple
import sys
from pathlib import Path
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[4] / "gitmodules"))
try:
    from BNAF.bnaf import BNAF, MaskedWeight, Tanh
finally:
    sys.path.pop(0)

# Ideally we wouldn't do this globally, but we drown in warning messages otherwise 
# The culprit is line 199 in BNAF/bnaf.py. This affects both the forward and
# backwards passes, so we can't just disable with a context manager.
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead.")

from .bijection import Bijection


class BlockNeuralAutoregressiveBijection(Bijection):
    def __init__(
            self,
            num_input_channels,
            num_hidden_layers,
            hidden_channels_factor,
            activation,
            residual
    ):
        shape = (num_input_channels,)
        super().__init__(x_shape=shape, z_shape=shape)

        if activation == "tanh":
            warnings.warn("BNAF with tanh nonlinearities is not surjective")
            act_class = Tanh
        elif activation == "leaky-relu":
            act_class = LeakyReLU
        elif activation == "soft-leaky-relu":
            act_class = SoftLeakyReLU
        else:
            assert False, F"Invalid activation {activation}"

        layers = [
            MaskedWeight(
                in_features=num_input_channels,
                out_features=num_input_channels * hidden_channels_factor,
                dim=num_input_channels
            ),
            act_class()
        ]

        for _ in range(num_hidden_layers):
            layers += [
                MaskedWeight(
                    in_features=num_input_channels * hidden_channels_factor,
                    out_features=num_input_channels * hidden_channels_factor,
                    dim=num_input_channels
                ),
                act_class()
            ]

        layers += [
            MaskedWeight(
                in_features=num_input_channels * hidden_channels_factor,
                out_features=num_input_channels,
                dim=num_input_channels
            )
        ]

        self.bnaf = BNAF(*layers, res=residual)

    def _x_to_z(self, x, **kwargs):
        z, log_jac = self.bnaf(x)
        return {
            "z": z,
            "log-jac": log_jac.view(x.shape[0], 1)
        }


# Drop-in surjective replacement for Tanh in bnaf module
class Nonlinearity(nn.Module):
    def forward(self, inputs, grad=None):
        outputs, log_jac = self._do_forward(inputs)

        if grad is None:
            grad = log_jac
        else:
            grad = log_jac.view(grad.shape) + grad

        return outputs, grad


class LeakyReLU(Nonlinearity):
    def _do_forward(self, inputs):
        outputs = F.leaky_relu(inputs, negative_slope=self.negative_slope)

        log_jac = torch.zeros_like(inputs)
        log_jac[inputs < 0] = np.log(self.negative_slope)

        return outputs, log_jac


# LeakyReLU doesn't give any gradient signal for the Jacobian term.
# This soft approximation does.
class SoftLeakyReLU(Nonlinearity):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def _do_forward(self, inputs):
        eps = self.negative_slope
        outputs = eps * inputs + (1 - eps) * F.softplus(inputs)
        log_jac = torch.log(eps + (1 - eps) * torch.sigmoid(inputs))
        return outputs, log_jac
