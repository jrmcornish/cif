import copy


def get_config(dataset, model, use_baseline):
    return {
        "dataset": dataset,
        "model": model,
        **get_config_base(dataset, model, use_baseline)
    }


def get_config_base(dataset, model, use_baseline):
    if dataset in ["2uniforms", "2lines", "8gaussians", "checkerboard", "2spirals", "rings"]:
        return get_2d_config(dataset, model, use_baseline)

    elif dataset in ["power", "gas", "hepmass", "miniboone"]:
        return get_uci_config(dataset, model, use_baseline)

    elif dataset in ["mnist", "fashion-mnist", "cifar10", "svhn"]:
        return get_images_config(dataset, model, use_baseline)

    else:
        assert False, f"Invalid dataset {dataset}"


def get_2d_config(dataset, model, use_baseline):
    if model in ["maf", "flat-realnvp"]:
        config = {
            "num_density_layers": 20 if use_baseline else 5,
            "g_hidden_channels": [50] * 4,

            "st_nets": [10] * 2,
            "p_nets": [50] * 4,
            "q_nets": [50] * 4,
        }

    elif model == "sos":
        config = {
            "num_density_layers": 3 if use_baseline else 2,
            "num_polynomials_per_layer": 2,
            "polynomial_degree": 4,

            "st_nets": [40] * 2,
            "p_nets": [40] * 4,
            "q_nets":  [40] * 4
        }

    elif model == "planar":
        config = {
            "num_density_layers": 20,

            "cond_hidden_channels": [10] * 2,
            "st_nets": [10] * 2,

            "p_nets": [10] * 2,
            "q_nets": [10] * 2
        }

    elif model == "nsf":
        config = {
            "num_density_layers": 2,
            "num_density_layers": 2,
            "num_bins": 64 if use_baseline else 24,
            "num_hidden_channels": 32,
            "num_hidden_layers": 2,
            "tail_bound": 5,
            "autoregressive": False,
            "dropout_probability": 0.,

            "st_nets": [24] * 2,
            "p_nets": [24] * 3,
            "q_nets": [24] * 3
        }

    elif model == "bnaf":
        config = {
            "num_density_layers": 1,
            "num_hidden_layers": 2,
            "hidden_channels_factor": 50 if use_baseline else 45,
            "activation": "soft-leaky-relu",

            "st_nets": [24] * 2,
            "p_nets": [24] * 3,
            "q_nets": [24] * 3,

            "max_epochs": 1000,
            "max_bad_valid_epochs": 1000,
            "test_batch_size": 1000
        }

    elif model == "ffjord":
        config = {
            "num_density_layers": 1,
            "hidden_channels": [64] * 3,
            "numerical_tolerance": 1e-5,

            "st_nets": [24] * 2,
            "p_nets": [24] * 3,
            "q_nets": [24] * 3
        }

    else:
        assert False, f"Invalid model `{model}' for dataset `{dataset}'"

    return {
        "num_u_channels": 0 if use_baseline else 1,
        "use_cond_affine": not use_baseline and model != "planar",

        "dequantize": False,

        "batch_norm": True,
        "batch_norm_apply_affine": True,
        "batch_norm_use_running_averages": False,

        "max_epochs": 1000,
        "max_grad_norm": None,
        "early_stopping": True,
        "max_bad_valid_epochs": 1000,
        "train_batch_size": 1000,
        "valid_batch_size": 1000,
        "test_batch_size": 10000,

        "opt": "adam",
        "lr": 1e-3,
        "lr_schedule": "none",
        "weight_decay": 0.,
        "epochs_per_test": 5,

        "num_train_elbo_samples": 10 if not use_baseline else 1,
        "num_valid_elbo_samples": 10 if not use_baseline else 1,
        "num_test_elbo_samples": 100 if not use_baseline else 1,

        **config,
    }


def get_uci_config(dataset, model, use_baseline):
    if model in ["maf", "flat-realnvp"]:
        if dataset in ["gas", "power"]:
            config = {
                "num_u_channels": 0 if use_baseline else 2,
                "num_density_layers": 10,
                "g_hidden_channels": [200] * 2 if use_baseline else [100] * 2,

                "st_nets": [100] * 2,
                "p_nets": [200] * 2,
                "q_nets": [200] * 2,
            }

        elif dataset in ["hepmass", "miniboone"]:
            if use_baseline:
                num_u_channels = 0
            elif dataset == "hepmass":
                num_u_channels = 5
            else:
                num_u_channels = 10

            config = {
                "num_u_channels": num_u_channels,
                "num_density_layers": 10,
                "g_hidden_channels": [512] * 2 if use_baseline else [128] * 2,

                "st_nets": [128] * 2,
                "p_nets": [512] * 2,
                "q_nets": [512] * 2
            }

    elif model == "sos":
        assert use_baseline
        config = {
            "num_u_channels": 0,

            "num_density_layers": 8,
            "g_hidden_channels": [200] * 2,
            "num_polynomials_per_layer": 5,
            "polynomial_degree": 4,

            "lr": 1e-3,
            "opt": "sgd"
        }

    elif model == "nsf":
        if dataset in ["power", "gas"]:
            config = {
                "num_u_channels": 0 if use_baseline else 2,
                "num_density_layers": 10 if use_baseline else 7,
                "num_hidden_layers": 2,
                "num_hidden_channels": 256,
                "num_bins": 8,
                "dropout_probability": 0. if dataset == "power" else 0.1,

                "st_nets": [120] * 2,
                "p_nets": [240] * 2,
                "q_nets": [240] * 2,

                "lr": 0.0005,
                "train_batch_size": 5120
            }

            # We convert the presecribed number of steps into epochs
            if dataset == "gas":
                config["max_epochs"] = (400_000 * 512) // 852_174
            elif dataset == "power":
                config["max_epochs"] = (400_000 * 512) // 1_615_917

            # We run for a bit longer to ensure convergence
            config["max_epochs"] += 100

        elif dataset == "hepmass":
            config = {
                "num_u_channels": 0 if use_baseline else 5,

                "num_density_layers": 20 if use_baseline else 10,
                "num_hidden_layers": 1,
                "num_hidden_channels": 128,
                "num_bins": 8,
                "dropout_probability": 0.2,

                "st_nets": [64] * 2,
                "p_nets": [192] * 2,
                "q_nets": [192] * 2,

                # We increase the lr and batch size by a factor of 10 from the prescribed values
                "lr": 0.0005 * 10,
                "train_batch_size": 256 * 10,

                # We convert the presecribed number of steps into epochs, and run for 400
                # epochs extra because we don't quite converge otherwise.
                "max_epochs": (400_000 * 256) // 315_123 + 400
            }

        elif dataset == "miniboone":
            config = {
                "num_u_channels": 0 if use_baseline else 10,

                "num_density_layers": 10 if use_baseline else 4,
                "num_hidden_layers": 1,
                "num_hidden_channels": 32,
                "num_bins": 4,
                "dropout_probability": 0.2,

                "st_nets": [32] * 2,
                "p_nets": [64] * 2,
                "q_nets": [64] * 2,

                # We increase the lr and batch size by a factor of 10 from the prescribed values
                "lr": 0.0003 * 10,
                "train_batch_size": 128 * 10,

                # We convert the presecribed number of steps into epochs
                "max_epochs": (200_000 * 128) // 29_556
            }

        config = {
            **config,
            "tail_bound": 3,
            "autoregressive": True,
            "batch_norm": False,
            "max_grad_norm": 5,

            "lr_schedule": "cosine"
        }

    else:
        assert False, f"Invalid model `{model}' for dataset `{dataset}''"

    config = {
        "dequantize": False,

        "batch_norm": True,
        "batch_norm_apply_affine": not use_baseline,
        "batch_norm_use_running_averages": False,

        "early_stopping": True,
        "train_batch_size": 1000,
        "valid_batch_size": 5000,
        "test_batch_size": 5000,

        "opt": "adam",
        "lr": 1e-3,
        "lr_schedule": "none",
        "weight_decay": 0.,
        "max_bad_valid_epochs": 5000,
        "max_epochs": 5000,
        "max_grad_norm": None,
        "epochs_per_test": 5,

        "num_train_elbo_samples": 1 if not use_baseline else 1,
        "num_valid_elbo_samples": 5 if not use_baseline else 1,
        "num_test_elbo_samples": 10 if not use_baseline else 1,

        **config
    }

    return config


def get_images_config(dataset, model, use_baseline):
    if model == "multiscale-realnvp":
        if use_baseline:
            config = {
                "g_hidden_channels": [64] * 8,
                "num_u_channels": 0
            }

        else:
            config = {
                "g_hidden_channels": [64] * 4,
                "num_u_channels": 1,
                "st_nets": [8] * 2,
                "p_nets": [64] * 2,
                "q_nets": [64] * 2
            }

        config["early_stopping"] = True
        config["train_batch_size"] = 100
        config["valid_batch_size"] = 500
        config["test_batch_size"] = 500
        config["opt"] = "adam"
        config["lr"] = 1e-4
        config["weight_decay"] = 0.

        if dataset in ["cifar10", "svhn"]:
            config["logit_tf_lambda"] = 0.05
            config["logit_tf_scale"] = 256

        elif dataset in ["mnist", "fashion-mnist"]:
            config["logit_tf_lambda"] = 1e-6
            config["logit_tf_scale"] = 256

    elif model == "glow":
        if use_baseline:
            config = {
                "num_scales": 3,
                "num_steps_per_scale": 32,
                "g_num_hidden_channels": 512,
                "num_u_channels": 0,
                "valid_batch_size": 500,
                "test_batch_size": 500
            }

        else:
            config = {
                "num_scales": 2,
                "num_steps_per_scale": 32,
                "g_num_hidden_channels": 256,
                "num_u_channels": 1,
                "st_nets": 64,
                "p_nets": 128,
                "q_nets": 128,
                "valid_batch_size": 100,
                "test_batch_size": 100
            }

        config["early_stopping"] = False
        config["train_batch_size"] = 64
        config["opt"] = "adamax"
        config["lr"] = 5e-4

        if dataset in ["cifar10"]:
            config["weight_decay"] = 0.1
        else:
            config["weight_decay"] = 0.

        config["centering_tf_scale"] = 256

    else:
        assert False, f"Invalid model {model} for dataset {dataset}"

    config = {
        **config,

        "dequantize": True,

        "batch_norm": True,
        "batch_norm_apply_affine": use_baseline,
        "batch_norm_use_running_averages": True,
        "batch_norm_momentum": 0.1,

        "lr_schedule": "none",
        "max_bad_valid_epochs": 50,
        "max_grad_norm": None,
        "max_epochs": 1000,
        "epochs_per_test": 1,

        "num_train_elbo_samples": 1 if not use_baseline else 1,
        "num_valid_elbo_samples": 5 if not use_baseline else 1,
        "num_test_elbo_samples": 10 if not use_baseline else 1
    }

    return config
