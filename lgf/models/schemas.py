def get_schema(config):
    if config["model"] == "pure-cond-affine-mlp":
        return get_pure_cond_affine_schema(config)
    else:
        return add_normalization_layers(get_base_schema(config), config)


def get_pure_cond_affine_schema(config):
    return [get_cond_affine_layer(config) for _ in range(config["num_density_layers"])]


def get_base_schema(config):
    schema = []

    if config["logit_tf_lambda"] is not None:
        schema.append({
            "type": "logit",
            "lambda": config["logit_tf_lambda"],
            "scale": config["logit_tf_scale"]
        })


    model = config["model"]

    if model == "multiscale-realnvp":
        schema += get_multiscale_realnvp_schema(
            coupler_hidden_channels=config["g_nets_hidden_channels"]
        )

    elif model == "flat-realnvp":
        schema += get_flat_realnvp_schema(
            num_density_layers=config["num_density_layers"],
            coupler_hidden_channels=config["g_nets_hidden_channels"]
        )

    elif model == "maf":
        schema += get_maf_schema(
            num_density_layers=config["num_density_layers"],
            ar_coupler_hidden_channels=config["g_nets_hidden_channels"]
        )

    else:
        assert False, f"Invalid model `{model}'"

    return schema


def add_normalization_layers(base_schema, config):
    if config["num_u_channels"] > 0:
        return add_normalization_layers_from_prototype(
            base_schema=base_schema,
            prototype=get_cond_affine_layer(config)
        )

    elif config["batch_norm"]:
        return add_normalization_layers_from_prototype(
            base_schema=base_schema,
            prototype={"type": "batch-norm"}
        )

    else:
        return base_schema


def get_cond_affine_layer(config):
    return {
        "type": "cond-affine",
        "num_u_channels": config["num_u_channels"],
        "st_coupler": get_st_coupler_config(config),
        "p_coupler": get_p_coupler_config(config),
        "q_coupler": get_q_coupler_config(config)
    }


def add_normalization_layers_from_prototype(base_schema, prototype):
    schema = []
    for layer in base_schema:
        schema.append(layer)

        if layer["type"] in ["acl", "made"]:
            schema.append(prototype)

    return schema


def get_st_coupler_config(config):
    return get_coupler_config("t", "s", "st", config)


def get_p_coupler_config(config):
    return get_coupler_config("p_mu", "p_sigma", "p", config)


def get_q_coupler_config(config):
    return get_coupler_config("q_mu", "q_sigma", "q", config)


def get_coupler_config(shift_prefix, log_scale_prefix, shift_log_scale_prefix, config):
    model = config["model"]

    shift_key = f"{shift_prefix}_nets_hidden_channels"
    log_scale_key = f"{log_scale_prefix}_nets_hidden_channels"
    shift_log_scale_key = f"{shift_log_scale_prefix}_nets_hidden_channels"

    if shift_key in config and log_scale_key in config:
        return {
            "independent_nets": True,
            "shift_net": get_net_config(config[shift_key], model),
            "log_scale_net": get_net_config(config[log_scale_key], model)
        }

    elif shift_log_scale_key in config:
        return {
            "independent_nets": False,
            "shift_log_scale_net": get_net_config(config[shift_log_scale_key], model)
        }

    else:
        assert False, f"Must specify either `{shift_log_scale_key}', or both `{shift_key}' and `{log_scale_key}'"


def get_net_config(net_size, model):
    if net_size is None:
        return {
            "type": "null"
        }

    elif model in ["multiscale-realnvp", "pure-cond-affine-resnet"]:
        return {
            "type": "resnet",
            "hidden_channels": net_size
        }

    elif model in ["pure-cond-affine-mlp", "maf", "flat-realnvp"]:
        return {
            "type": "mlp",
            "activation": "tanh",
            "hidden_channels": net_size
        }

    else:
        assert False, f"Invalid model type {model}"


def get_multiscale_realnvp_schema(coupler_hidden_channels):
    schema = [
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": True},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
        {"type": "squeeze", "factor": 2},
        {"type": "acl", "mask_type": "split_channel", "reverse_mask": True},
        {"type": "acl", "mask_type": "split_channel", "reverse_mask": False},
        {"type": "acl", "mask_type": "split_channel", "reverse_mask": True},
        {"type": "split"},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": True},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": False},
        {"type": "acl", "mask_type": "checkerboard", "reverse_mask": True}
    ]

    for layer in schema:
        if layer["type"] == "acl":
            layer["num_u_channels"] = 0
            layer["coupler"] = {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": coupler_hidden_channels
                }
            }

    return schema


def get_flat_realnvp_schema(
        num_density_layers,
        coupler_hidden_channels
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        result.append({
            "type": "acl",
            "mask_type": "alternating_channel",
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
        })

    return result


def get_maf_schema(
        num_density_layers,
        ar_coupler_hidden_channels
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        if i > 0:
            result.append({"type": "flip"})

        result.append({
            "type": "made",
            "ar_coupler_hidden_channels": ar_coupler_hidden_channels,
            "ar_coupler_activation": "tanh"
        })

    return result
