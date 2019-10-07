import numpy as np

import torch
import torch.nn as nn

from .components.bijections import (
    FlipBijection,
    CheckerboardMasked2dAffineCouplingBijection,
    ChannelwiseMaskedAffineCouplingBijection,
    MADEBijection,
    BatchNormBijection,
    Squeeze2dBijection,
    LogitTransformBijection,
    ViewBijection,
    ConditionalAffineBijection
)
from .components.densities import (
    DiagonalGaussianDensity,
    DiagonalGaussianConditionalDensity,
    ELBODensity,
    BijectionDensity,
    SplitDensity
)
from .components.helpers import (
    SplittingModule,
    JoiningModule,
    ConstantMap
)
from .components.networks import get_mlp, get_resnet


def get_density(
        schema,
        x_shape
):
    if not schema:
        return get_standard_gaussian_density(x_shape=x_shape)

    layer_config = schema[0]

    if layer_config["type"] == "split":
        split_x_shape = (x_shape[0] // 2, *x_shape[1:])
        return SplitDensity(
            density_1=get_density(
                schema=schema[1:],
                x_shape=split_x_shape
            ),
            density_2=get_standard_gaussian_density(x_shape=split_x_shape),
            dim=1
        )

    bijection = get_bijection(layer_config=layer_config, x_shape=x_shape)

    prior = get_density(
        schema=schema[1:],
        x_shape=bijection.z_shape
    )

    if layer_config.get("num_u_channels", 0) == 0:
        return BijectionDensity(bijection=bijection, prior=prior)

    else:
        return ELBODensity(
            bijection=bijection,
            prior=prior,
            p_u_density=get_conditional_density(
                num_u_channels=layer_config["num_u_channels"],
                net_config=layer_config["p_net"],
                x_shape=x_shape
            ),
            q_u_density=get_conditional_density(
                num_u_channels=layer_config["num_u_channels"],
                net_config=layer_config["q_net"],
                x_shape=x_shape
            )
        )


def get_bijection(
        layer_config,
        x_shape
):
    if layer_config["type"] == "acl":
        return get_acl_bijection(
            mask_type=layer_config["mask_type"],
            reverse_mask=layer_config["reverse_mask"],
            num_u_channels=layer_config["num_u_channels"],
            separate_coupler_nets=layer_config["separate_coupler_nets"],
            coupler_net_config=layer_config["coupler_net"],
            x_shape=x_shape
        )

    elif layer_config["type"] == "squeeze":
        return Squeeze2dBijection(x_shape=x_shape, factor=layer_config["factor"])

    elif layer_config["type"] == "logit":
        return LogitTransformBijection(x_shape=x_shape, lam=layer_config["lambda"], scale=layer_config["scale"])

    elif layer_config["type"] == "flatten":
        return ViewBijection(x_shape=x_shape, z_shape=(np.prod(x_shape),))

    elif layer_config["type"] == "cond-affine":
        return ConditionalAffineBijection(
            x_shape=x_shape,
            coupler=get_coupler(
                num_input_channels=layer_config["num_u_channels"],
                output_shape=x_shape,
                separate_nets=layer_config["separate_coupler_nets"],
                net_config=layer_config["st_net"]
            )
        )

    elif layer_config["type"] == "made":
        assert len(x_shape) == 1
        return MADEBijection(
            num_inputs=x_shape[0],
            hidden_units=layer_config["ar_map_hidden_units"],
            activation=nn.Tanh
        )

    elif layer_config["type"] == "batch-norm":
        return BatchNormBijection(x_shape=x_shape)

    elif layer_config["type"] == "flip":
        return FlipBijection(x_shape=x_shape, dim=1)

    else:
        assert False, f"Invalid layer type {layer_config['type']}"


def get_acl_bijection(
        mask_type,
        reverse_mask,
        num_u_channels,
        separate_coupler_nets,
        coupler_net_config,
        x_shape
):
    num_x_channels = x_shape[0]

    if mask_type == "checkerboard":
        return CheckerboardMasked2dAffineCouplingBijection(
            x_shape=x_shape,
            coupler=get_coupler(
                num_input_channels=num_x_channels+num_u_channels,
                output_shape=x_shape,
                separate_nets=separate_coupler_nets,
                net_config=coupler_net_config
            ),
            reverse_mask=reverse_mask
        )

    else:
        if mask_type == "split_channel":
            mask = torch.arange(x_shape[0]) < x_shape[0] // 2
        elif mask_type == "alternating_channel":
            mask = torch.arange(x_shape[0]) % 2 == 0
        else:
            assert False, f"Invalid mask type {mask_type}"

        if reverse_mask:
            mask = ~mask

        num_coupler_in_channels = torch.sum(mask).item()
        num_coupler_out_channels = num_x_channels - num_coupler_in_channels

        return ChannelwiseMaskedAffineCouplingBijection(
            x_shape=x_shape,
            mask=mask,
            coupler=get_coupler(
                num_input_channels=num_coupler_in_channels+num_u_channels,
                output_shape=(num_coupler_out_channels, *x_shape[1:]),
                separate_nets=separate_coupler_nets,
                net_config=coupler_net_config
            )
        )


def get_coupler(
        num_input_channels,
        output_shape,
        separate_nets,
        net_config
):
    if separate_nets:
        return get_coupler_with_separate_nets(
            num_input_channels=num_input_channels,
            output_shape=output_shape,
            net_config=net_config
        )

    else:
        return get_coupler_with_shared_net(
            num_input_channels=num_input_channels,
            output_shape=output_shape,
            net_config=net_config
        )


def get_coupler_with_shared_net(
        num_input_channels,
        output_shape,
        net_config
):
    if net_config["type"] == "mlp":
        assert len(output_shape) == 1

        coupler_net = get_mlp(
            num_inputs=num_input_channels,
            hidden_units=net_config["hidden_units"],
            output_dim=2*output_shape[0],
            activation=nn.Tanh
        )

    elif net_config["type"] == "resnet":
        assert len(output_shape) == 3
        assert net_config["num_blocks"] > 0

        coupler_net = get_resnet(
            num_input_channels=num_input_channels,
            num_blocks=net_config["num_blocks"],
            num_hidden_channels_per_block=net_config["num_hidden_channels_per_block"],
            num_output_channels=2*output_shape[0]
        )

    else:
        assert False, f"Invalid net type {net_config['type']}"

    return SplittingModule(
        module=coupler_net,
        output_names=["scale", "shift"],
        dim=1
    )


def get_coupler_with_separate_nets(
        num_input_channels,
        output_shape,
        net_config
):
    assert net_config["type"] == "mlp", "Should share convolutional coupler weights"
    assert len(output_shape) == 1

    def get_mlp_coupler_net(activation):
        return get_mlp(
            num_inputs=num_input_channels,
            hidden_units=net_config["hidden_units"],
            output_dim=output_shape[0],
            activation=activation
        )

    return JoiningModule(
        modules=[get_mlp_coupler_net(nn.Tanh), get_mlp_coupler_net(nn.ReLU)],
        output_names=["scale", "shift"]
    )


def get_uniform_density(x_shape):
    return BijectionDensity(
        bijection=LogitTransformBijection(
            x_shape=x_shape,
            lam=0,
            scale=1
        ).inverse(),
        prior=UniformDensity(x_shape)
    )


def get_standard_gaussian_density(x_shape):
    return DiagonalGaussianDensity(
        mean=torch.zeros(x_shape),
        stddev=torch.ones(x_shape),
        num_fixed_samples=64
    )


def get_conditional_density(
        num_u_channels,
        net_config,
        x_shape
):
    if net_config["type"] == "constant":
        return get_constant_gaussian_conditional_density(
            num_u_channels=num_u_channels
        )

    else:
        return get_mean_field_gaussian_conditional_density(
            num_u_channels=num_u_channels,
            net_config=net_config,
            x_shape=x_shape
        )


def get_constant_gaussian_conditional_density(num_u_channels):
    return DiagonalGaussianConditionalDensity(
        mean_log_std_map=JoiningModule(
            modules=[
                ConstantMap(value=torch.zeros(num_u_channels)),
                ConstantMap(value=torch.ones(num_u_channels))
            ],
            output_names=["mean", "log-stddev"]
        )
    )


def get_mean_field_gaussian_conditional_density(
        num_u_channels,
        net_config,
        x_shape
):
    num_x_channels = x_shape[0]
    num_output_channels = 2 * num_u_channels

    if net_config["type"] == "resnet":
        assert len(x_shape) == 3
        assert net_config['num_blocks'] > 0

        network = get_resnet(
            num_input_channels=num_x_channels,
            num_blocks=net_config['num_blocks'],
            num_hidden_channels_per_block=net_config['num_hidden_channels_per_block'],
            num_output_channels=num_output_channels
        )

    elif net_config["type"] == "mlp":
        network = get_mlp(
            num_inputs=num_x_channels,
            hidden_units=net_config["hidden_units"],
            output_dim=num_output_channels,
            activation=nn.Tanh
        )

    else:
        assert False, f"Invalid net type {net_config['type']}"

    return DiagonalGaussianConditionalDensity(
        mean_log_std_map=SplittingModule(
            module=network,
            output_names=["mean", "log-stddev"],
            dim=1
        )
    )
