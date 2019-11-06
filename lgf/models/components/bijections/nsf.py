import torch
import torch.nn as nn
import torch.nn.functional as F

import nsf.nde.transforms.coupling as nsf_couplers
import nsf.nn as nsf_nn
import nsf.utils as nsf_utils

from .bijection import Bijection

_TAIL_BOUND = 3

class CoupledRationalQuadraticSplineBijection(Bijection):
    def __init__(
            self,
            num_input_channels,
            hidden_channels,
            num_resnet_blocks,
            dropout_probability,
            num_bins,
            evens_masked,
    ):
        super().__init__(x_shape=(num_input_channels,), z_shape=(num_input_channels,))

        def get_nn_fn(in_features, out_features):
            net = nsf_nn.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_channels,
                context_features=None,
                num_blocks=num_resnet_blocks,
                activation=F.relu,
                dropout_probability=dropout_probability,
                use_batch_norm=0,
            )
            return net

        self.flow = nsf_couplers.PiecewiseRationalQuadraticCouplingTransform(
            mask=nsf_utils.create_alternating_binary_mask(num_input_channels, even=evens_masked),
            transform_net_create_fn=get_nn_fn,
            num_bins=num_bins,
            tails='linear',
            tail_bound=_TAIL_BOUND,
            apply_unconditional_transform=1,
        )

    def _x_to_z(self, x):
        (z, log_jac) = self.flow(x)
        return {
            "z": z,
            "log-jac": log_jac
        }


class AutoregressiveRationalQuadraticSplineBijection(Bijection):
    def __init__(
            self,
            num_input_channels,
            hidden_channels,
            num_resnet_blocks,
            dropout_probability,
            num_bins,
    ):
        super().__init__(x_shape=(num_input_channels,), z_shape=(num_input_channels,))

        self.flow = nsf_couplers.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=num_input_channels,
            hidden_features=hidden_channels,
            context_features=None,
            num_bins=num_bins,
            tails='linear',
            tail_bound=_TAIL_BOUND,
            num_blocks=num_resnet_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=1,
        )

    def _x_to_z(self, x):
        (z, log_jac) = self.flow(x)
        return {
            "z": z,
            "log-jac": log_jac
        }