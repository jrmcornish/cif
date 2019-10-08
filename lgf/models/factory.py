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
            coupler_config=layer_config["coupler"],
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
                num_output_channels=x_shape[0],
                config=layer_config["coupler"]
            )
        )

    elif layer_config["type"] == "made":
        assert len(x_shape) == 1
        return MADEBijection(
            num_inputs=x_shape[0],
            hidden_units=layer_config["ar_map_hidden_units"],
            activation=get_activation(layer_config["ar_map_activation"])
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
        coupler_config,
        x_shape
):
    num_x_channels = x_shape[0]

    if mask_type == "checkerboard":
        return CheckerboardMasked2dAffineCouplingBijection(
            x_shape=x_shape,
            coupler=get_coupler(
                num_input_channels=num_x_channels+num_u_channels,
                num_output_channels=num_x_channels,
                config=coupler_config
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

        num_passthrough_channels = torch.sum(mask).item()

        return ChannelwiseMaskedAffineCouplingBijection(
            x_shape=x_shape,
            mask=mask,
            coupler=get_coupler(
                num_input_channels=num_passthrough_channels+num_u_channels,
                num_output_channels=num_x_channels-num_passthrough_channels,
                config=coupler_config
            )
        )


def get_coupler(
        num_input_channels,
        num_output_channels,
        config
):
    if "shift_net" in config and "scale_net" in config:
        return get_coupler_with_separate_nets(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            shift_net_config=config["shift_net"],
            scale_net_config=config["scale_net"]
        )

    elif "shift_scale_net" in config:
        return get_coupler_with_shared_net(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            net_config=config["shift_scale_net"]
        )

    else:
        assert False, "Unspecified coupler net config"


def get_coupler_with_shared_net(
        num_input_channels,
        num_output_channels,
        net_config
):
    if net_config["type"] == "mlp":
        coupler_net = get_mlp(
            num_inputs=num_input_channels,
            hidden_units=net_config["hidden_units"],
            num_outputs=2*num_output_channels,
            activation=get_activation(net_config["activation"])
        )

    elif net_config["type"] == "resnet":
        coupler_net = get_resnet(
            num_input_channels=num_input_channels,
            hidden_channels=net_config["hidden_channels"],
            num_output_channels=2*num_output_channels
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
        num_output_channels,
        shift_net_config,
        scale_net_config
):
    def get_coupler_net(net_config):
        assert net_config["type"] == "mlp", "Should share convolutional coupler weights"

        return get_mlp(
            num_inputs=num_input_channels,
            hidden_units=net_config["hidden_units"],
            num_outputs=num_output_channels,
            activation=get_activation(net_config["activation"])
        )

    return JoiningModule(
        modules=[
            get_coupler_net(shift_net_config),
            get_coupler_net(scale_net_config)
        ],
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

        network = get_resnet(
            num_input_channels=num_x_channels,
            hidden_channels=net_config["hidden_channels"],
            num_output_channels=num_output_channels
        )

    elif net_config["type"] == "mlp":
        network = get_mlp(
            num_inputs=num_x_channels,
            hidden_units=net_config["hidden_units"],
            num_outputs=num_output_channels,
            activation=get_activation(net_config["activation"])
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


def get_activation(name):
    if name == "tanh":
        return nn.Tanh
    elif name == "relu":
        return nn.ReLU
    else:
        assert False, f"Invalid activation {name}"
