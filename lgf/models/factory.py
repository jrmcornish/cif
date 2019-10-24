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
    LogitBijection,
    PositiveScalarMultiplicationBijection,
    ScalarAdditionBijection,
    ViewBijection,
    ConditionalAffineBijection,
    BruteForceInvertible1x1ConvBijection,
    LUInvertible1x1ConvBijection
)
from .components.densities import (
    DiagonalGaussianDensity,
    DiagonalGaussianConditionalDensity,
    ELBODensity,
    BijectionDensity,
    SplitDensity
)
from .components.couplers import IndependentCoupler, SharedCoupler
from .components.networks import (
    ConstantNetwork,
    get_mlp,
    get_resnet,
    get_glow_cnn
)


def get_density(
        schema,
        x_shape
):
    if not schema:
        return get_standard_gaussian_density(x_shape=x_shape)

    layer_config = schema[0]
    schema_tail = schema[1:]

    if layer_config["type"] == "split":
        split_x_shape = (x_shape[0] // 2, *x_shape[1:])
        return SplitDensity(
            density_1=get_density(
                schema=schema_tail,
                x_shape=split_x_shape
            ),
            density_2=get_standard_gaussian_density(x_shape=split_x_shape),
            dim=1
        )

    bijection = get_bijection(layer_config=layer_config, x_shape=x_shape)

    prior = get_density(
        schema=schema_tail,
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
                coupler_config=layer_config["p_coupler"],
                x_shape=x_shape
            ),
            q_u_density=get_conditional_density(
                num_u_channels=layer_config["num_u_channels"],
                coupler_config=layer_config["q_coupler"],
                x_shape=x_shape
            )
        )


def get_uniform_density(x_shape):
    return BijectionDensity(
        bijection=LogitBijection(x_shape=x_shape).inverse(),
        prior=UniformDensity(x_shape)
    )


def get_standard_gaussian_density(x_shape):
    return DiagonalGaussianDensity(
        mean=torch.zeros(x_shape),
        stddev=torch.ones(x_shape),
        num_fixed_samples=64
    )


def get_bijection(
        layer_config,
        x_shape
):
    if layer_config["type"] == "acl":
        return get_acl_bijection(config=layer_config, x_shape=x_shape)

    elif layer_config["type"] == "squeeze":
        return Squeeze2dBijection(x_shape=x_shape, factor=layer_config["factor"])

    elif layer_config["type"] == "logit":
        return LogitBijection(x_shape=x_shape)

    elif layer_config["type"] == "scalar-mult":
        return PositiveScalarMultiplicationBijection(
            x_shape=x_shape,
            value=layer_config["value"]
        )

    elif layer_config["type"] == "scalar-add":
        return ScalarAdditionBijection(x_shape=x_shape, value=layer_config["value"])

    elif layer_config["type"] == "flatten":
        return ViewBijection(x_shape=x_shape, z_shape=(np.prod(x_shape),))

    elif layer_config["type"] == "cond-affine":
        return ConditionalAffineBijection(
            x_shape=x_shape,
            coupler=get_coupler(
                input_shape=(layer_config["num_u_channels"], *x_shape[1:]),
                num_channels_per_output=x_shape[0],
                config=layer_config["st_coupler"]
            )
        )

    elif layer_config["type"] == "made":
        assert len(x_shape) == 1
        return MADEBijection(
            num_input_channels=x_shape[0],
            hidden_channels=layer_config["hidden_channels"],
            activation=get_activation(layer_config["activation"])
        )

    elif layer_config["type"] == "batch-norm":
        return BatchNormBijection(
            x_shape=x_shape,
            per_channel=layer_config["per_channel"]
        )

    elif layer_config["type"] == "flip":
        return FlipBijection(x_shape=x_shape, dim=1)

    elif layer_config["type"] == "invconv":
        if layer_config["lu"]:
            return LUInvertible1x1ConvBijection(x_shape=x_shape)
        else:
            return BruteForceInvertible1x1ConvBijection(x_shape=x_shape)

    else:
        assert False, f"Invalid layer type {layer_config['type']}"


def get_acl_bijection(config, x_shape):
    num_x_channels = x_shape[0]
    num_u_channels = config["num_u_channels"]

    if config["mask_type"] == "checkerboard":
        return CheckerboardMasked2dAffineCouplingBijection(
            x_shape=x_shape,
            coupler=get_coupler(
                input_shape=(num_x_channels+num_u_channels, *x_shape[1:]),
                num_channels_per_output=num_x_channels,
                config=config["coupler"]
            ),
            reverse_mask=config["reverse_mask"]
        )

    else:
        if config["mask_type"] == "split-channel":
            mask = torch.arange(x_shape[0]) < x_shape[0] // 2
        elif config["mask_type"] == "alternating-channel":
            mask = torch.arange(x_shape[0]) % 2 == 0
        else:
            assert False, f"Invalid mask type {config['mask_type']}"

        if config["reverse_mask"]:
            mask = ~mask

        num_passthrough_channels = torch.sum(mask).item()

        return ChannelwiseMaskedAffineCouplingBijection(
            x_shape=x_shape,
            mask=mask,
            coupler=get_coupler(
                input_shape=(num_passthrough_channels+num_u_channels, *x_shape[1:]),
                num_channels_per_output=num_x_channels-num_passthrough_channels,
                config=config["coupler"]
            )
        )


def get_conditional_density(
        num_u_channels,
        coupler_config,
        x_shape
):
    return DiagonalGaussianConditionalDensity(
        coupler=get_coupler(
            input_shape=x_shape,
            num_channels_per_output=num_u_channels,
            config=coupler_config
        )
    )


def get_coupler(
        input_shape,
        num_channels_per_output,
        config
):
    if config["independent_nets"]:
        return get_coupler_with_independent_nets(
            input_shape=input_shape,
            num_channels_per_output=num_channels_per_output,
            shift_net_config=config["shift_net"],
            log_scale_net_config=config["log_scale_net"]
        )

    else:
        return get_coupler_with_shared_net(
            input_shape=input_shape,
            num_channels_per_output=num_channels_per_output,
            net_config=config["shift_log_scale_net"]
        )


def get_coupler_with_shared_net(
        input_shape,
        num_channels_per_output,
        net_config
):
    return SharedCoupler(
        shift_log_scale_net=get_coupler_net(
            input_shape=input_shape,
            num_output_channels=2*num_channels_per_output,
            net_config=net_config
        )
    )


def get_coupler_with_independent_nets(
        input_shape,
        num_channels_per_output,
        shift_net_config,
        log_scale_net_config
):
    return IndependentCoupler(
        shift_net=get_coupler_net(
            input_shape=input_shape,
            num_output_channels=num_channels_per_output,
            net_config=shift_net_config
        ),
        log_scale_net=get_coupler_net(
            input_shape=input_shape,
            num_output_channels=num_channels_per_output,
            net_config=log_scale_net_config
        )
    )


def get_coupler_net(input_shape, num_output_channels, net_config):
    num_input_channels = input_shape[0]

    if net_config["type"] == "mlp":
        assert len(input_shape) == 1
        return get_mlp(
            num_input_channels=num_input_channels,
            hidden_channels=net_config["hidden_channels"],
            num_output_channels=num_output_channels,
            activation=get_activation(net_config["activation"])
        )

    elif net_config["type"] == "resnet":
        assert len(input_shape) == 3
        return get_resnet(
            num_input_channels=num_input_channels,
            hidden_channels=net_config["hidden_channels"],
            num_output_channels=num_output_channels
        )

    elif net_config["type"] == "glow-cnn":
        assert len(input_shape) == 3
        return get_glow_cnn(
            num_input_channels=num_input_channels,
            num_hidden_channels=net_config["num_hidden_channels"],
            num_output_channels=num_output_channels
        )

    elif net_config["type"] == "constant":
        value = torch.full((num_output_channels, *input_shape[1:]), net_config["value"])
        return ConstantNetwork(value=value, fixed=net_config["fixed"])

    elif net_config["type"] == "identity":
        assert num_output_channels == num_input_channels
        return lambda x: x

    else:
        assert False, f"Invalid net type {net_config['type']}"


def get_activation(name):
    if name == "tanh":
        return nn.Tanh
    elif name == "relu":
        return nn.ReLU
    else:
        assert False, f"Invalid activation {name}"
