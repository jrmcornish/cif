import numpy as np

import torch
import torch.nn as nn

from .components.bijections import (
    FlipBijection,
    Checkerboard2dAffineCouplingBijection,
    MaskedChannelwiseAffineCouplingBijection,
    SplitChannelwiseAffineCouplingBijection,
    AlternatingChannelwiseAffineCouplingBijection,
    MADEBijection,
    BatchNormBijection,
    AffineBijection,
    Squeeze2dBijection,
    LogitBijection,
    TanhBijection,
    ScalarMultiplicationBijection,
    ScalarAdditionBijection,
    ViewBijection,
    ConditionalAffineBijection,
    BruteForceInvertible1x1ConvBijection,
    LUInvertible1x1ConvBijection,
    SumOfSquaresPolynomialBijection,
    CoupledRationalQuadraticSplineBijection,
    AutoregressiveRationalQuadraticSplineBijection,
    BlockNeuralAutoregressiveBijection,
    LULinearBijection,
    RandomChannelwisePermutationBijection,
    FFJORDBijection,
    PlanarBijection,
    ConditionalPlanarBijection,
    ResidualFlowBijection,
    ActNormBijection
)
from .components.densities import (
    DiagonalGaussianDensity,
    DiagonalGaussianConditionalDensity,
    ELBODensity,
    BijectionDensity,
    SplitDensity,
    DequantizationDensity,
    PassthroughBeforeEvalDensity,
    UpdateLipschitzBeforeForwardDensity,
    DataParallelDensity
)
from .components.couplers import IndependentCoupler, ChunkedSharedCoupler
from .components.networks import (
    ConstantNetwork,
    get_mlp,
    get_resnet,
    get_glow_cnn,
    get_lipschitz_mlp,
    get_lipschitz_cnn
)

def get_density(schema, x_train):
    x_shape = x_train.shape[1:]

    # TODO: Ugly to have the first schema item be special like this.
    # Would be better to have a schema be a dict of form:
    #   {
    #       "wrappers": [{"type": "wrapper-1-type", ...}, ...],
    #       "x-to-z": [{"type": ...}, ...]
    #   }
    if schema[0]["type"] == "passthrough-before-eval":
        assert not data_parallel, "Not yet supported due to possibly unexpected behaviour"

        num_points = schema[0]["num_passthrough_data_points"]
        x_idxs = torch.randperm(x_train.shape[0])[:num_points]
        return PassthroughBeforeEvalDensity(
            density=get_density_recursive(schema[1:], x_shape),
            x=x_train[x_idxs]
        )

    density = get_density_recursive(schema, x_shape)

    # We always add this for generality. If data parallelism is not desired, then
    # this can be controlled by manipulating CUDA_VISIBLE_DEVICES. But if we don't
    # include this component, then we won't be able to save/load state dicts across
    # different runs easily unless the runs always use the same number of GPUs
    density = DataParallelDensity(density)

    # We have to do this _after_ DataParallel because we need Lipschitz updates to
    # happen globally, i.e. not be split on separate GPUs, or we will get autograd
    # errors.
    for layer in schema:
        if layer["type"] == "resblock":
            density = UpdateLipschitzBeforeForwardDensity(density)
            break

    return density


def get_density_recursive(
        schema,
        x_shape
):
    # TODO: We could specify this explicitly to allow different prior distributions
    if not schema:
        return get_standard_gaussian_density(x_shape=x_shape)

    layer_config = schema[0]
    schema_tail = schema[1:]

    if layer_config["type"] == "dequantization":
        return DequantizationDensity(
            density=get_density_recursive(
                schema=schema_tail,
                x_shape=x_shape
            )
        )

    elif layer_config["type"] == "split":
        split_x_shape = (x_shape[0] // 2, *x_shape[1:])
        return SplitDensity(
            density_1=get_density_recursive(
                schema=schema_tail,
                x_shape=split_x_shape
            ),
            density_2=get_standard_gaussian_density(x_shape=split_x_shape),
            dim=1
        )

    elif layer_config["type"] == "passthrough-before-eval":
        assert False, "`passthrough-before-eval` must occur as the first item in a schema"

    else:
        return get_bijection_density(
            layer_config=layer_config,
            schema_tail=schema_tail,
            x_shape=x_shape
        )


def get_bijection_density(layer_config, schema_tail, x_shape):
    bijection = get_bijection(layer_config=layer_config, x_shape=x_shape)

    prior = get_density_recursive(
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

    elif layer_config["type"] == "sigmoid":
        return LogitBijection(x_shape=x_shape).inverse()

    elif layer_config["type"] == "tanh":
        return TanhBijection(x_shape=x_shape)

    elif layer_config["type"] == "scalar-mult":
        return ScalarMultiplicationBijection(
            x_shape=x_shape,
            value=layer_config["value"]
        )

    elif layer_config["type"] == "scalar-add":
        return ScalarAdditionBijection(x_shape=x_shape, value=layer_config["value"])

    elif layer_config["type"] == "flatten":
        return ViewBijection(x_shape=x_shape, z_shape=(int(np.prod(x_shape)),))

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
            per_channel=layer_config["per_channel"],
            apply_affine=layer_config["apply_affine"],
            momentum=layer_config["momentum"]
        )

    elif layer_config["type"] == "act-norm":
        return ActNormBijection(x_shape=x_shape)

    elif layer_config["type"] == "affine":
        return AffineBijection(
            x_shape=x_shape,
            per_channel=layer_config["per_channel"]
        )

    elif layer_config["type"] == "cond-affine":
        return ConditionalAffineBijection(
            x_shape=x_shape,
            coupler=get_coupler(
                input_shape=(layer_config["num_u_channels"], *x_shape[1:]),
                num_channels_per_output=x_shape[0],
                config=layer_config["st_coupler"]
            )
        )

    elif layer_config["type"] == "flip":
        return FlipBijection(x_shape=x_shape, dim=1)

    elif layer_config["type"] == "invconv":
        if layer_config["lu"]:
            return LUInvertible1x1ConvBijection(x_shape=x_shape)
        else:
            return BruteForceInvertible1x1ConvBijection(x_shape=x_shape)

    elif layer_config["type"] == "linear":
        assert len(x_shape) == 1
        return LULinearBijection(num_input_channels=x_shape[0])

    elif layer_config["type"] == "rand-channel-perm":
        return RandomChannelwisePermutationBijection(x_shape=x_shape)

    elif layer_config["type"] == "sos":
        assert len(x_shape) == 1
        return SumOfSquaresPolynomialBijection(
            num_input_channels=x_shape[0],
            hidden_channels=layer_config["hidden_channels"],
            activation=get_activation(layer_config["activation"]),
            num_polynomials=layer_config["num_polynomials"],
            polynomial_degree=layer_config["polynomial_degree"],
        )

    elif layer_config["type"] == "nsf-ar":
        assert len(x_shape) == 1
        return AutoregressiveRationalQuadraticSplineBijection(
            num_input_channels=x_shape[0],
            num_hidden_layers=layer_config["num_hidden_layers"],
            num_hidden_channels=layer_config["num_hidden_channels"],
            num_bins=layer_config["num_bins"],
            tail_bound=layer_config["tail_bound"],
            activation=get_activation(layer_config["activation"]),
            dropout_probability=layer_config["dropout_probability"]
        )

    elif layer_config["type"] == "nsf-c":
        assert len(x_shape) == 1
        return CoupledRationalQuadraticSplineBijection(
            num_input_channels=x_shape[0],
            num_hidden_layers=layer_config["num_hidden_layers"],
            num_hidden_channels=layer_config["num_hidden_channels"],
            num_bins=layer_config["num_bins"],
            tail_bound=layer_config["tail_bound"],
            activation=get_activation(layer_config["activation"]),
            dropout_probability=layer_config["dropout_probability"],
            reverse_mask=layer_config["reverse_mask"]
        )

    elif layer_config["type"] == "bnaf":
        assert len(x_shape) == 1
        return BlockNeuralAutoregressiveBijection(
            num_input_channels=x_shape[0],
            num_hidden_layers=layer_config["num_hidden_layers"],
            hidden_channels_factor=layer_config["hidden_channels_factor"],
            activation=layer_config["activation"],
            residual=layer_config["residual"]
        )

    elif layer_config["type"] == "ode":
        assert len(x_shape) == 1 # TODO: Make possible for images
        return FFJORDBijection(
            x_shape=x_shape,
            velocity_hidden_channels=layer_config["hidden_channels"],
            relative_tolerance=layer_config["numerical_tolerance"],
            absolute_tolerance=layer_config["numerical_tolerance"],
            num_u_channels=layer_config["num_u_channels"]
        )

    elif layer_config["type"] == "planar":
        assert len(x_shape) == 1
        return PlanarBijection(
            num_input_channels=x_shape[0],
        )

    elif layer_config["type"] == "cond-planar":
        assert len(x_shape) == 1
        return ConditionalPlanarBijection(
            num_input_channels=x_shape[0],
            num_u_channels=layer_config["num_u_channels"],
            cond_hidden_channels=layer_config["cond_hidden_channels"],
            cond_activation=get_activation(layer_config["cond_activation"])
        )

    elif layer_config["type"] == "resblock":
        # TODO: Rename Bijection
        return ResidualFlowBijection(
            x_shape=x_shape,
            lipschitz_net=get_lipschitz_net(
                input_shape=x_shape,
                num_output_channels=x_shape[0],
                config=layer_config["net"]
            ),
            reduce_memory=layer_config["reduce_memory"]
        )

    else:
        assert False, f"Invalid layer type {layer_config['type']}"


def get_acl_bijection(config, x_shape):
    num_x_channels = x_shape[0]
    num_u_channels = config["num_u_channels"]

    if config["mask_type"] == "checkerboard":
        return Checkerboard2dAffineCouplingBijection(
            x_shape=x_shape,
            coupler=get_coupler(
                input_shape=(num_x_channels+num_u_channels, *x_shape[1:]),
                num_channels_per_output=num_x_channels,
                config=config["coupler"]
            ),
            reverse_mask=config["reverse_mask"]
        )

    else:
        def coupler_factory(num_passthrough_channels):
            return get_coupler(
                input_shape=(num_passthrough_channels+num_u_channels, *x_shape[1:]),
                num_channels_per_output=num_x_channels-num_passthrough_channels,
                config=config["coupler"]
            )

        if config["mask_type"] == "alternating-channel":
            return AlternatingChannelwiseAffineCouplingBijection(
                x_shape=x_shape,
                coupler_factory=coupler_factory,
                reverse_mask=config["reverse_mask"]
            )

        elif config["mask_type"] == "split-channel":
            return SplitChannelwiseAffineCouplingBijection(
                x_shape=x_shape,
                coupler_factory=coupler_factory,
                reverse_mask=config["reverse_mask"]
            )

        else:
            assert False, f"Invalid mask type {config['mask_type']}"


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
    return ChunkedSharedCoupler(
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
            num_output_channels=num_output_channels,
            zero_init_output=net_config["zero_init_output"]
        )

    elif net_config["type"] == "constant":
        value = torch.full((num_output_channels, *input_shape[1:]), net_config["value"], dtype=torch.get_default_dtype())
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


def get_lipschitz_net(input_shape, num_output_channels, config):
    if config["type"] == "cnn":
        return get_lipschitz_cnn(
            input_shape=input_shape,
            num_hidden_channels=config["num_hidden_channels"],
            num_output_channels=num_output_channels,
            lipschitz_constant=config["lipschitz_constant"],
            max_train_lipschitz_iters=config["max_train_lipschitz_iters"],
            max_eval_lipschitz_iters=config["max_test_lipschitz_iters"],
            lipschitz_tolerance=config["lipschitz_tolerance"]
        )

    elif config["type"] == "mlp":
        assert len(input_shape) == 1
        return get_lipschitz_mlp(
            num_input_channels=input_shape[0],
            hidden_channels=config["hidden_channels"],
            num_output_channels=num_output_channels,
            lipschitz_constant=config["lipschitz_constant"],
            max_train_lipschitz_iters=config["max_train_lipschitz_iters"],
            max_eval_lipschitz_iters=config["max_test_lipschitz_iters"],
            lipschitz_tolerance=config["lipschitz_tolerance"]
        )

    else:
        assert False, f"Invalid Lipschitz net type {config['net']}"
