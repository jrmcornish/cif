import torch
import torch.nn as nn
import torch.nn.functional as F

from .nsf_bayesiains.transforms import coupling as nsf_couplers
from .nsf_bayesiains.transforms import autoregressive as nsf_autoreg
from .nsf_bayesiains.resnet import ResidualNet as nsf_resnet
from .nsf_bayesiains import utils as nsf_utils

from .bijection import Bijection

class CoupledRationalQuadraticSplineBijection(Bijection):
    def __init__(
            self,
            num_input_channels,
            hidden_channels,
            num_resnet_blocks,
            dropout_probability,
            num_bins,
            use_batchnorm_in_nets,
            apply_unconditional_transform,
            tail_bound,
            evens_masked,
    ):
        super().__init__(x_shape=(num_input_channels,), z_shape=(num_input_channels,))

        def get_nn_fn(in_features, out_features):
            net = nsf_resnet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_channels,
                context_features=None,
                num_blocks=num_resnet_blocks,
                activation=F.relu,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batchnorm_in_nets,
            )
            return net

        self.flow = nsf_couplers.PiecewiseRationalQuadraticCouplingTransform(
            mask=nsf_utils.create_alternating_binary_mask(num_input_channels, even=evens_masked),
            transform_net_create_fn=get_nn_fn,
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform,
        )

    def _x_to_z(self, x):
        (z, log_jac) = self.flow(x)
        return {
            "z": z,
            "log-jac": torch.unsqueeze(log_jac, 1)
        }


class AutoregressiveRationalQuadraticSplineBijection(Bijection):
    def __init__(
            self,
            num_input_channels,
            hidden_channels,
            num_resnet_blocks,
            dropout_probability,
            use_batchnorm_in_nets,
            tail_bound,
            num_bins,
    ):
        super().__init__(x_shape=(num_input_channels,), z_shape=(num_input_channels,))

        self.flow = nsf_autoreg.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=num_input_channels,
            hidden_features=hidden_channels,
            context_features=None,
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            num_blocks=num_resnet_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batchnorm_in_nets,
        )

    def _x_to_z(self, x):
        (z, log_jac) = self.flow(x)
        return {
            "z": z,
            "log-jac": torch.unsqueeze(log_jac, 1)
        }
