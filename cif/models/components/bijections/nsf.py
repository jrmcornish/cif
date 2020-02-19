import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[4] / "gitmodules" / "nsf"))
try:
    from nde.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
    from nde.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
    from nn import ResidualNet
    from utils import create_alternating_binary_mask
finally:
    sys.path.pop(0)

from .bijection import Bijection


class RationalQuadraticSplineBijection(Bijection):
    def __init__(
            self,
            num_input_channels,
            flow
    ):
        shape = (num_input_channels,)
        super().__init__(x_shape=shape, z_shape=shape)

        self.flow = flow

    def _x_to_z(self, x):
        z, log_jac = self.flow(x)
        return {
            "z": z,
            "log-jac": log_jac.view(x.shape[0], 1)
        }

    def _z_to_x(self, z):
        x, log_jac = self.flow.inverse(z)
        return {
            "x": x,
            "log-jac": log_jac.view(z.shape[0], 1)
        }


class CoupledRationalQuadraticSplineBijection(RationalQuadraticSplineBijection):
    def __init__(
            self,
            num_input_channels,
            num_hidden_layers,
            num_hidden_channels,
            num_bins,
            tail_bound,
            activation,
            dropout_probability,
            reverse_mask
    ):
        def transform_net_create_fn(in_features, out_features):
            return ResidualNet(
                in_features=in_features,
                out_features=out_features,
                context_features=None,
                hidden_features=num_hidden_channels,
                num_blocks=num_hidden_layers,
                activation=activation(),
                dropout_probability=dropout_probability,
                use_batch_norm=False
            )

        super().__init__(
            num_input_channels=num_input_channels,
            flow=PiecewiseRationalQuadraticCouplingTransform(
                mask=create_alternating_binary_mask(
                    num_input_channels,
                    even=reverse_mask
                ),
                transform_net_create_fn=transform_net_create_fn,
                num_bins=num_bins,
                tails='linear',
                tail_bound=tail_bound,

                # Setting True corresponds to equations (4), (5), (6) in the NSF paper:
                apply_unconditional_transform=True
            )
        )


class AutoregressiveRationalQuadraticSplineBijection(RationalQuadraticSplineBijection):
    def __init__(
            self,
            num_input_channels,
            num_hidden_layers,
            num_hidden_channels,
            num_bins,
            tail_bound,
            activation,
            dropout_probability
    ):
        super().__init__(
            num_input_channels=num_input_channels,
            flow=MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=num_input_channels,
                hidden_features=num_hidden_channels,
                context_features=None,
                num_bins=num_bins,
                tails='linear',
                tail_bound=tail_bound,
                num_blocks=num_hidden_layers,
                use_residual_blocks=True,
                random_mask=False,
                activation=activation(),
                dropout_probability=dropout_probability,
                use_batch_norm=False,
            )
        )
