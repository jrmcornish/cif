def get_schema(config):
    return add_normalization_layers(get_base_schema(config), config)


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
            coupler_hidden_channels=config["g_nets"]["hidden_channels"]
        )

    elif model == "flat-realnvp":
        schema += get_flat_realnvp_schema(
            num_density_layers=config["num_density_layers"],
            coupler_hidden_units=config["g_nets"]["hidden_units"]
        )

    elif model == "maf":
        schema += get_maf_schema(
            num_density_layers=config["num_density_layers"],
            ar_map_hidden_units=config["g_nets"]["hidden_units"]
        )

    else:
        assert False, f"Invalid model {model}"

    return schema


def add_normalization_layers(base_schema, config):
    schema = []

    for layer in base_schema:
        if layer["type"] == "acl":
            layer["num_u_channels"] = 0

        schema.append(layer)

        if layer["type"] in ["acl", "made"]:
            if config["num_u_channels"] > 0:
                schema.append(get_cond_affine_layer_config(
                    model=config["model"],
                    num_u_channels=config["num_u_channels"],
                    st_nets_config=config["st_nets"],
                    p_nets_config=config["p_nets"],
                    q_nets_config=config["q_nets"]
                ))

            elif config["batch_norm"]:
                schema.append({"type": "batch-norm"})

    return schema


def get_cond_affine_layer_config(
        model,
        num_u_channels,
        st_nets_config,
        p_nets_config,
        q_nets_config
):
    def get_full_net_config(net_config):
        if model == "multiscale-realnvp":
            return {
                "type": "resnet",
                "hidden_channels": net_config["hidden_channels"]
            }

        elif model in ["flat-realnvp", "maf"]:
            return {
                "type": "mlp",
                "hidden_units": net_config["hidden_units"],
                "activation": "tanh"
            }

        else:
            assert False, f"Invalid model {model}"

    return {
        "type": "cond-affine",
        "num_u_channels": num_u_channels,
        "coupler": {"shift_scale_net": get_full_net_config(st_nets_config)},
        "p_net": get_full_net_config(p_nets_config),
        "q_net": get_full_net_config(q_nets_config)
    }


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
            layer["coupler"] = {
                "shift_scale_net": {
                    "type": "resnet",
                    "hidden_channels": coupler_hidden_channels
                }
            }

    return schema


def get_flat_realnvp_schema(
        num_density_layers,
        coupler_hidden_units
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        result.append({
            "type": "acl",
            "mask_type": "alternating_channel",
            "reverse_mask": i % 2 == 0,
            "coupler": {
                "shift_net": {
                    "type": "mlp",
                    "hidden_units": coupler_hidden_units,
                    "activation": "relu"
                },
                "scale_net": {
                    "type": "mlp",
                    "hidden_units": coupler_hidden_units,
                    "activation": "tanh"
                }
            }
        })

    return result


def get_maf_schema(
        num_density_layers,
        ar_map_hidden_units
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        if i > 0:
            result.append({"type": "flip"})

        result.append({
            "type": "made",
            "ar_map_hidden_units": ar_map_hidden_units,
            "ar_map_activation": "tanh"
        })

    return result
