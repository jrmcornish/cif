def get_schema(config):
    model = config["model"] 
    if model in ["glow", "multiscale-realnvp", "flat-realnvp", "maf", "sos"]:
        return get_schema_from_base(config)

    elif model == "pure-cond-affine-mlp":
        return get_pure_cond_affine_schema(config)

    else:
        assert False, f"Invalid model {model}"


def get_pure_cond_affine_schema(config):
    return [
        get_cond_affine_layer(config) for _ in range(config["num_density_layers"])
    ]


def get_schema_from_base(config):
    base_schema = get_preproc_schema(config) + get_base_schema(config)

    schema = []
    for layer in base_schema:
        if layer["type"] == "batch-norm" and config["num_u_channels"] > 0:
            assert layer["apply_affine"]

            schema += [
                {
                    **layer,
                    "apply_affine": False
                },
                get_cond_affine_layer(config)
            ]

        else:
            schema.append(layer)

    return schema


def get_preproc_schema(config):
    if config["dequantize"]:
        schema = [{"type": "dequantization"}]
    else:
        schema = []

    if "logit_tf_lambda" in config and "logit_tf_scale" in config:
        assert "rescale_tf_scale" not in config
        schema += get_logit_tf_schema(
            lam=config["logit_tf_lambda"],
            scale=config["logit_tf_scale"]
        )

    elif "centering_tf_scale" in config:
        assert "logit_tf_lambda" not in config
        assert "logit_tf_scale" not in config
        schema += get_centering_tf_schema(
            scale=config["centering_tf_scale"]
        )

    return schema


def get_logit_tf_schema(lam, scale):
    return [
        {"type": "scalar-mult", "value": (1 - 2*lam) / scale},
        {"type": "scalar-add", "value": lam},
        {"type": "logit", "lambda": 0., "scale": 1.}
    ]


def get_centering_tf_schema(scale):
    return [
        {"type": "scalar-mult", "value": 1 / scale},
        {"type": "scalar-add", "value": -.5}
    ]


def get_base_schema(config):
    model = config["model"]

    if model == "multiscale-realnvp":
        return get_multiscale_realnvp_schema(
            coupler_hidden_channels=config["g_hidden_channels"]
        )

    elif model == "flat-realnvp":
        return get_flat_realnvp_schema(
            num_density_layers=config["num_density_layers"],
            coupler_hidden_channels=config["g_hidden_channels"]
        )

    elif model == "maf":
        return get_maf_schema(
            num_density_layers=config["num_density_layers"],
            hidden_channels=config["g_hidden_channels"]
        )

    elif model == "sos":
        return get_sos_schema(
            num_density_layers=config["num_density_layers"],
            hidden_channels=config["g_hidden_channels"],
            num_polynomials_per_layer=config["num_polynomials_per_layer"],
            polynomial_degree=config["polynomial_degree"],
        )

    elif model == "glow":
        return get_glow_schema(
            num_scales=config["num_scales"],
            num_steps_per_scale=config["num_steps_per_scale"],
            coupler_num_hidden_channels=config["g_num_hidden_channels"],
            lu_decomposition=True
        )

    else:
        assert False, f"Invalid model `{model}'"


def get_cond_affine_layer(config):
    return {
        "type": "cond-affine",
        "num_u_channels": config["num_u_channels"],
        "st_coupler": get_st_coupler_config(config),
        "p_coupler": get_p_coupler_config(config),
        "q_coupler": get_q_coupler_config(config)
    }


def get_st_coupler_config(config):
    return get_coupler_config("t", "s", "st", config)


def get_p_coupler_config(config):
    return get_coupler_config("p_mu", "p_sigma", "p", config)


def get_q_coupler_config(config):
    return get_coupler_config("q_mu", "q_sigma", "q", config)


def get_coupler_config(shift_prefix, log_scale_prefix, shift_log_scale_prefix, config):
    model = config["model"]

    shift_key = f"{shift_prefix}_nets"
    log_scale_key = f"{log_scale_prefix}_nets"
    shift_log_scale_key = f"{shift_log_scale_prefix}_nets"

    if shift_key in config and log_scale_key in config:
        return {
            "independent_nets": True,
            "shift_net": get_coupler_net_config(config[shift_key], model),
            "log_scale_net": get_coupler_net_config(config[log_scale_key], model)
        }

    elif shift_log_scale_key in config:
        return {
            "independent_nets": False,
            "shift_log_scale_net": get_coupler_net_config(config[shift_log_scale_key], model)
        }

    else:
        assert False, f"Must specify either `{shift_log_scale_key}', or both `{shift_key}' and `{log_scale_key}'"


def get_coupler_net_config(net_spec, model):
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
        if model in ["multiscale-realnvp", "pure-cond-affine-resnet"]:
            return {
                "type": "resnet",
                "hidden_channels": net_spec
            }

        elif model in ["pure-cond-affine-mlp", "maf", "flat-realnvp", "sos"]:
            return {
                "type": "mlp",
                "activation": "tanh",
                "hidden_channels": net_spec
            }

        else:
            assert False, f"Invalid model {model} for net specification {net_spec}"

    elif isinstance(net_spec, int):
        if model == "glow":
            return {
                "type": "glow-cnn",
                "num_hidden_channels": net_spec
            }

        else:
            assert False, f"Invalid model {model} for net specification {net_spec}"

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
                    "type": "batch-norm",
                    "per_channel": True,
                    "apply_affine": True
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
                    "type": "batch-norm",
                    "per_channel": True,
                    "apply_affine": True
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


def get_flat_realnvp_schema(
        num_density_layers,
        coupler_hidden_channels
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        result += [
            {
                "type": "acl",
                "mask_type": "alternating-channel",
                "reverse_mask": i % 2 == 0,
                "coupler": {
                    "independent_nets": True,
                    "shift_net": {
                        "type": "mlp",
                        "hidden_channels": coupler_hidden_channels,
                        "activation": "relu"
                    },
                    "log_scale_net": {
                        "type": "mlp",
                        "hidden_channels": coupler_hidden_channels,
                        "activation": "tanh"
                    }
                },
                "num_u_channels": 0
            },
            {
                "type": "batch-norm",
                "per_channel": False,
                "apply_affine": True
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
                "type": "batch-norm",
                "per_channel": False,
                "apply_affine": True
            }
        ]

    return result


# TODO: Batch norm?
# TODO: Flip after each layer?
def get_sos_schema(
        num_density_layers,
        hidden_channels,
        num_polynomials_per_layer,
        polynomial_degree
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        if i > 0:
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
                "type": "batch-norm",
                "per_channel": False,
                "apply_affine": True
            }
       ]

    return result
