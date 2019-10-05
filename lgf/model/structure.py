def get_model_structure(config):
    base_structure = get_base_density_structure(config)
    structure = add_normalization_layers(base_structure, config)
    return structure


def get_base_density_structure(config):
    structure = []

    if config["logit_tf_lambda"] is not None:
        structure.append({
            "type": "logit",
            "lambda": config["logit_tf_lambda"],
            "scale": config["logit_tf_scale"]
        })


    model = config["model"]

    if model == "multiscale-realnvp":
        structure += get_multiscale_realnvp_density_structure(
            coupler_num_blocks=config["g"]["num_blocks"],
            coupler_num_hidden_channels_per_block=config["g"]["num_hidden_channels_per_block"],
        )

    elif model == "flat-realnvp":
        structure += get_flat_realnvp_density_structure(
            num_density_layers=config["num_density_layers"],
            coupler_hidden_units=config["g"]["hidden_units"]
        )

    elif model == "maf":
        structure += get_maf_density_structure(
            num_density_layers=config["num_density_layers"],
            ar_map_hidden_units=config["g"]["hidden_units"]
        )

    else:
        assert False, f"Invalid model {model}"

    return structure


def add_normalization_layers(base_structure, config):
    structure = []

    for layer in base_structure:
        if layer["type"] == "acl":
            layer["num_u_channels"] = 0
            layer["p_hidden_channels"] = None
            layer["q_hidden_channels"] = None

        structure.append(layer)

        if layer["type"] in ["acl", "made"]:
            if config["num_u_channels"] > 0:
                cond_affine = {
                    "type": "cond-affine",
                    "separate_coupler_nets": False,
                    "num_u_channels": config["num_u_channels"]
                }

                if config["model"] in ["flat-realnvp", "maf"]:
                    cond_affine = {
                        **cond_affine,
                        "st_net": {"type": "mlp", **config["st"]},
                        "p_net": {"type": "mlp", **config["p"]},
                        "q_net": {"type": "mlp", **config["q"]}
                    }

                elif config["model"] == "multiscale-realnvp":
                    cond_affine = {
                        **cond_affine,
                        "st_net": {"type": "resnet", **config["st"]},
                        "p_net": {"type": "resnet", **config["p"]},
                        "q_net": {"type": "resnet", **config["q"]}
                    }

                else:
                    assert False, f"Invalid model {config['model']}"

                structure.append(cond_affine)

            elif config["batch_norm"]:
                structure.append({"type": "batch-norm"})

    return structure


def get_multiscale_realnvp_density_structure(
        coupler_num_blocks,
        coupler_num_hidden_channels_per_block
):
    structure = [
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

    for layer in structure:
        if layer["type"] == "acl":
            layer["separate_coupler_nets"] = False
            layer["coupler_net"] = {
                "type": "resnet",
                "num_blocks": coupler_num_blocks,
                "num_hidden_channels_per_block": coupler_num_hidden_channels_per_block
            }

    return structure


def get_flat_realnvp_density_structure(
        num_density_layers,
        coupler_hidden_units
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        result.append({
            "type": "acl",
            "mask_type": "alternating_channel",
            "reverse_mask": i % 2 == 0,
            "coupler_net": {
                "type": "mlp",
                "hidden_units": coupler_hidden_units
            },
            "separate_coupler_nets": True,
        })

    return result


def get_maf_density_structure(
        num_density_layers,
        ar_map_hidden_units
):
    result = [{"type": "flatten"}]

    for i in range(num_density_layers):
        if i > 0:
            result.append({"type": "flip"})

        result.append({
            "type": "made",
            "ar_map_hidden_units": ar_map_hidden_units
        })

    return result
