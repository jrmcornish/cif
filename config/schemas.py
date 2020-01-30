def get_schema(config):
    schema = get_base_schema(config=config)

    if config["pure_cond_affine"]:
        assert config["use_cond_affine"]
        schema = remove_non_normalise_layers(schema=schema)

    if config["use_cond_affine"]:
        assert config["num_u_channels"] > 0
        schema = add_cond_affine_before_each_normalise(schema=schema, config=config)

    schema = apply_pq_coupler_config_settings(schema=schema, config=config)

    schema = get_preproc_schema(config=config) + schema

    assert not (config["batch_norm"] and config["act_norm"])

    if config["batch_norm"]:
        schema = replace_normalise_with_batch_norm(schema=schema, config=config)
    elif config["act_norm"]:
        schema = replace_normalise_with_act_norm(schema=schema)
    else:
        schema = remove_normalise_layers(schema=schema)

    return schema


def get_preproc_schema(config):
    if config["dequantize"]:
        schema = [{"type": "dequantization"}]
    else:
        schema = []

    if config.get("logit_tf_lambda") is not None and config.get("logit_tf_scale") is not None:
        assert config.get("rescale_tf_scale") is None
        schema += get_logit_tf_schema(
            lam=config["logit_tf_lambda"],
            scale=config["logit_tf_scale"]
        )

    elif config.get("centering_tf_scale") is not None:
        assert config.get("logit_tf_lambda") is None
        assert config.get("logit_tf_scale") is None
        schema += get_centering_tf_schema(
            scale=config["centering_tf_scale"]
        )

    return schema


# TODO: Could just pass the whole config to each constructor
def get_base_schema(config):
    ty = config["schema_type"]

    if ty == "multiscale-realnvp":
        return get_multiscale_realnvp_schema(
            coupler_hidden_channels=config["g_hidden_channels"]
        )

    elif ty == "flat-realnvp":
        return get_flat_realnvp_schema(config=config)

    elif ty == "maf":
        return get_maf_schema(
            num_density_layers=config["num_density_layers"],
            hidden_channels=config["ar_map_hidden_channels"]
        )

    elif ty == "sos":
        return get_sos_schema(
            num_density_layers=config["num_density_layers"],
            hidden_channels=config["g_hidden_channels"],
            num_polynomials_per_layer=config["num_polynomials_per_layer"],
            polynomial_degree=config["polynomial_degree"],
        )

    elif ty == "nsf":
        return get_nsf_schema(
            num_density_layers=config["num_density_layers"],
            num_hidden_layers=config["num_hidden_layers"],
            num_hidden_channels=config["num_hidden_channels"],
            num_bins=config["num_bins"],
            tail_bound=config["tail_bound"],
            autoregressive=config["autoregressive"],
            dropout_probability=config["dropout_probability"]
        )

    elif ty == "bnaf":
        return get_bnaf_schema(
            num_density_layers=config["num_density_layers"],
            num_hidden_layers=config["num_hidden_layers"],
            activation=config["activation"],
            hidden_channels_factor=config["hidden_channels_factor"]
        )

    elif ty == "glow":
        return get_glow_schema(
            num_scales=config["num_scales"],
            num_steps_per_scale=config["num_steps_per_scale"],
            coupler_num_hidden_channels=config["g_num_hidden_channels"],
            lu_decomposition=True
        )

    elif ty == "ffjord":
        return get_ffjord_schema(
            num_density_layers=config["num_density_layers"],
            velocity_hidden_channels=config["hidden_channels"],
            numerical_tolerance=config["numerical_tolerance"],
            num_u_channels=config["num_u_channels"]
        )

    elif ty == "planar":
        return get_planar_schema(config=config)

    elif ty == "cond-affine":
        return get_cond_affine_schema(config=config)

    elif ty == "affine":
        return get_affine_schema(config=config)

    elif ty == "flat-resflow":
        return get_flat_resflow_schema(config=config)

    elif ty == "multiscale-resflow":
        return get_multiscale_resflow_schema(config=config)

    else:
        assert False, f"Invalid schema type `{ty}'"


def remove_non_normalise_layers(schema):
    return [layer for layer in schema if layer["type"] == "normalise"]


def remove_normalise_layers(schema):
    return [layer for layer in schema if layer["type"] != "normalise"]


def replace_normalise_with_batch_norm(schema, config):
    if config["batch_norm_use_running_averages"]:
        new_schema = []
        momentum = config["batch_norm_momentum"]

    else:
        new_schema = [
            {
                "type": "passthrough-before-eval",
                # XXX: This should be sufficient for most of the non-image
                # datasets we have but can be made a config value if necessary
                "num_passthrough_data_points": 100_000
            }
        ]
        momentum = 1.

    apply_affine = config["batch_norm_apply_affine"]

    for layer in schema:
        if layer["type"] == "normalise":
            new_schema.append({
                "type": "batch-norm",
                "per_channel": True, # Hard coded for now; seems always to do better
                "momentum": momentum,
                "apply_affine": config["batch_norm_apply_affine"]
            })

        else:
            new_schema.append(layer)

    return new_schema


def replace_normalise_with_act_norm(schema):
    new_schema = []

    for layer in schema:
        if layer["type"] == "normalise":
            new_schema.append({"type": "act-norm"})

        else:
            new_schema.append(layer)

    return new_schema


def add_cond_affine_before_each_norm(schema, config):
    new_schema = []
    for layer in schema:
        if layer["type"] == "normalise":
            new_schema.append(get_cond_affine_layer(config))

        new_schema.append(layer)

    return new_schema


def apply_pq_coupler_config_settings(schema, config):
    new_schema = []
    for layer in schema:
        if layer.get("num_u_channels", 0) > 0:
            layer = {
                **layer,
                "p_coupler": get_p_coupler_config(config),
                "q_coupler": get_q_coupler_config(config)
            }

        new_schema.append(layer)

    return new_schema


def get_logit_tf_schema(lam, scale):
    return [
        {"type": "scalar-mult", "value": (1 - 2*lam) / scale},
        {"type": "scalar-add", "value": lam},
        {"type": "logit"}
    ]


def get_centering_tf_schema(scale):
    return [
        {"type": "scalar-mult", "value": 1 / scale},
        {"type": "scalar-add", "value": -.5}
    ]


def get_cond_affine_layer(config):
    return {
        "type": "cond-affine",
        "num_u_channels": config["num_u_channels"],
        "st_coupler": get_st_coupler_config(config),
    }


def get_st_coupler_config(config):
    return get_coupler_config("t", "s", "st", config)


def get_p_coupler_config(config):
    return get_coupler_config("p_mu", "p_sigma", "p", config)


def get_q_coupler_config(config):
    return get_coupler_config("q_mu", "q_sigma", "q", config)


def get_coupler_config(shift_prefix, log_scale_prefix, shift_log_scale_prefix, config):
    schema_type = config["schema_type"]

    shift_key = f"{shift_prefix}_nets"
    log_scale_key = f"{log_scale_prefix}_nets"
    shift_log_scale_key = f"{shift_log_scale_prefix}_nets"

    if shift_key in config and log_scale_key in config:
        assert shift_log_scale_key not in config, "Over-specified coupler config"
        return {
            "independent_nets": True,
            "shift_net": get_coupler_net_config(config[shift_key], schema_type),
            "log_scale_net": get_coupler_net_config(config[log_scale_key], schema_type)
        }

    elif shift_log_scale_key in config:
        assert shift_key not in config and log_scale_key not in config, \
                "Over-specified coupler config"
        return {
            "independent_nets": False,
            "shift_log_scale_net": get_coupler_net_config(config[shift_log_scale_key], schema_type)
        }

    else:
        assert False, f"Must specify either `{shift_log_scale_key}', or both `{shift_key}' and `{log_scale_key}'"


def get_coupler_net_config(net_spec, schema_type):
    if net_spec in ["fixed-constant", "learned-constant"]:
        return {
            "type": "constant",
            "value": 0,
            "fixed": net_spec == "fixed-constant"
        }

    elif net_spec == "identity":
        return {
            "type": "identity"
        }

    elif isinstance(net_spec, list):
        if schema_type == "multiscale-realnvp":
            return {
                "type": "resnet",
                "hidden_channels": net_spec
            }

        elif schema_type in [
            "maf", "flat-realnvp", "sos", "nsf", "bnaf",
            "planar", "ffjord", "cond-affine", "resflow"
        ]:
            return {
                "type": "mlp",
                "activation": "tanh",
                "hidden_channels": net_spec
            }

        else:
            assert False, f"Invalid schema type {schema_type} for net specification {net_spec}"

    elif isinstance(net_spec, int):
        if schema_type == "glow":
            return {
                "type": "glow-cnn",
                "num_hidden_channels": net_spec
            }

        else:
            assert False, f"Invalid schema type {schema_type} for net specification {net_spec}"

    else:
        assert False, f"Invalid net specifier {net_spec}"


def get_multiscale_realnvp_schema(coupler_hidden_channels):
    base_schema = [
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": True},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
        {"type": "squeeze", "factor": 2},
        {"type": "acl", "mask_type": "split-channel", "reverse_mask": True},
        {"type": "acl", "mask_type": "split-channel", "reverse_mask": False},
        {"type": "acl", "mask_type": "split-channel", "reverse_mask": True},
        {"type": "split"},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": True},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": True}
    ]

    schema = []
    for layer in base_schema:
        if layer["type"] == "acl":
            schema += [
                {
                    **layer,
                    "num_u_channels": 0,
                    "coupler": {
                        "independent_nets": False,
                        "shift_log_scale_net": {
                            "type": "resnet",
                            "hidden_channels": coupler_hidden_channels
                        }
                    }
                },
                {
                    "type": "normalise"
                }
            ]

        else:
            schema.append(layer)

    return schema


def get_glow_schema(
        num_scales,
        num_steps_per_scale,
        coupler_num_hidden_channels,
        lu_decomposition
):
    schema = []
    for i in range(num_scales):
        if i > 0:
            schema.append({"type": "split"})

        schema.append({"type": "squeeze", "factor": 2})

        for _ in range(num_steps_per_scale):
            schema += [
                {
                    "type": "normalise"
                },
                {
                    "type": "invconv",
                    "lu": lu_decomposition
                },
                {
                    "type": "acl",
                    "mask_type": "split-channel",
                    "reverse_mask": False,
                    "coupler": {
                        "independent_nets": False,
                        "shift_log_scale_net": {
                            "type": "glow-cnn",
                            "num_hidden_channels": coupler_num_hidden_channels
                        }
                    },
                    "num_u_channels": 0
                }
            ]

    return schema


def get_flat_realnvp_schema(config):
    result = [{"type": "flatten"}]

    if config["coupler_shared_nets"]:
        coupler_config = {
            "independent_nets": False,
            "shift_log_scale_net": {
                "type": "mlp",
                "hidden_channels": config["coupler_hidden_channels"],
                "activation": "tanh"
            }
        }

    else:
        coupler_config = {
            "independent_nets": True,
            "shift_net": {
                "type": "mlp",
                "hidden_channels": config["coupler_hidden_channels"],
                "activation": "relu"
            },
            "log_scale_net": {
                "type": "mlp",
                "hidden_channels": config["coupler_hidden_channels"],
                "activation": "tanh"
            }
        }

    for i in range(config["num_density_layers"]):
        result += [
            {
                "type": "acl",
                "mask_type": "alternating-channel",
                "reverse_mask": i % 2 != 0,
                "coupler": coupler_config,
                "num_u_channels": 0
            },
            {
                "type": "normalise"
            }
        ]

    return result


def get_maf_schema(
        num_density_layers,
        hidden_channels
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        if i > 0:
            result.append({"type": "flip"})

        result += [
            {
                "type": "made",
                "hidden_channels": hidden_channels,
                "activation": "tanh"
            },
            {
                "type": "normalise"
            }
        ]

    return result


def get_sos_schema(
        num_density_layers,
        hidden_channels,
        num_polynomials_per_layer,
        polynomial_degree
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        if i > 0:
            # TODO: Try replacing with invconv
            result.append({"type": "flip"})

        result += [
            {
                "type": "sos",
                "hidden_channels": hidden_channels,
                "activation": "tanh",
                "num_polynomials": num_polynomials_per_layer,
                "polynomial_degree": polynomial_degree
            },
            {
                "type": "normalise"
            }
        ]

    return result


def get_nsf_schema(
        num_density_layers,
        num_hidden_layers, # TODO: Use a more descriptive name
        num_hidden_channels,
        num_bins,
        tail_bound,
        autoregressive,
        dropout_probability
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        result += [{"type": "rand-channel-perm"}, {"type": "linear"}]

        layer = {
            "type": "nsf-ar" if autoregressive else "nsf-c",
            "num_hidden_channels": num_hidden_channels,
            "num_hidden_layers": num_hidden_layers,
            "num_bins": num_bins,
            "tail_bound": tail_bound,
            "activation": "relu",
            "dropout_probability": dropout_probability
        }

        if not autoregressive:
            layer["reverse_mask"] = i % 2 == 0

        result.append(layer)

        result.append(
            {
                "type": "normalise"
            }
        )

    result += [{"type": "rand-channel-perm"}, {"type": "linear"}]

    return result


def get_bnaf_schema(
        num_density_layers,
        num_hidden_layers, # TODO: More descriptive name
        activation,
        hidden_channels_factor
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        if i > 0:
            result.append({"type": "flip"})

        result += [
            {
                "type": "bnaf",
                "num_hidden_layers": num_hidden_layers,
                "hidden_channels_factor": hidden_channels_factor,
                "activation": activation,
                "residual": i < num_density_layers - 1
            },
            {
                "type": "normalise"
            }
        ]

    return result


def get_ffjord_schema(
        num_density_layers,
        velocity_hidden_channels,
        numerical_tolerance,
        num_u_channels
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        result += [
            {
                "type": "ode",
                "hidden_channels": velocity_hidden_channels,
                "numerical_tolerance": numerical_tolerance,
                "num_u_channels": num_u_channels
            }
        ]
    
    return result


def get_planar_schema(config):
    if config["num_u_channels"] == 0:
        layer = {"type": "planar"}

    else:
        layer = {
            "type": "cond-planar",
            "num_u_channels": config["num_u_channels"],
            "cond_hidden_channels": config["cond_hidden_channels"],
            "cond_activation": "tanh"
        }

    result = [
        layer,
        {"type": "normalise"}
    ] * config["num_density_layers"]

    return [{"type": "flatten"}] + result


def get_cond_affine_schema(config):
    return (
        [{"type": "flatten"}] +
        [{"type": "normalise"}] * config["num_density_layers"]
    )


# TODO: Try using just cond-affines with constant u
def get_affine_schema(config):
    return (
        [{"type": "flatten"}] +
        [{"type": "affine", "per_channel": False}] * config["num_density_layers"]
    )


# TODO: Should have actnorm rather than batchnorm
def get_flat_resflow_schema(config):
    result = [{"type": "flatten"}]
    for _ in range(config["num_density_layers"]):
        result += [
            {
                "type": "resblock",
                "net": {
                    "type": "mlp",
                    "hidden_channels": config["hidden_channels"]
                }
            },
            {
                "type": "normalise"
            }
        ]

    add_lipschitz_config_to_resblocks(result, config)

    return result


# TODO: Should have actnorm rather than batchnorm
def get_multiscale_resflow_schema(config):
    result = []

    for i in range(config["num_scales"]):
        if i > 0:
            result.append({"type": "squeeze"})

        result.append({"type": "normalise"})

        for j in range(config["num_blocks_per_scale"]):
            result += [
                {
                    "type": "resblock",
                    "net": {
                        "type": "cnn",
                        "num_hidden_channels":  config["num_hidden_channels"]
                    }
                },
                {
                    "type": "normalise"
                }
            ]

    result.append({"type": "flatten"})

    for _ in range(config["num_output_fc_blocks"]):
        result += [
            {
                "type": "resblock",
                "net": {
                    "type": "mlp",
                    "hidden_channels": config["output_fc_hidden_channels"]
                }
            },
            {
                "type": "normalise"
            }
        ]

    add_lipschitz_config_to_resblocks(result, config)

    return result


def add_lipschitz_config_to_resblocks(schema, config):
    net_keys_to_copy = [
        "lipschitz_constant",
        "max_train_lipschitz_iters",
        "max_test_lipschitz_iters",
        "lipschitz_tolerance"
    ]

    for layer in schema:
        if layer["type"] == "resblock":
            for key in net_keys_to_copy:
                layer["net"][key] = config[key]

            layer["reduce_memory"] = config["reduce_memory"]
