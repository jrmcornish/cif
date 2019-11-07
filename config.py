import warnings
import copy


def get_config(dataset, model, use_baseline):
    return {
        "dataset": dataset,
        "model": model,
        **get_config_base(dataset, model, use_baseline)
    }


def get_config_base(dataset, model, use_baseline):
    if dataset in ["2uniforms", "8gaussians", "checkerboard", "2spirals", "rings"]:
        return get_2d_config(dataset, model, use_baseline)

    elif dataset in ["power", "gas", "hepmass", "miniboone"]:
        return get_uci_config(dataset, model, use_baseline)

    elif dataset in ["mnist", "fashion-mnist", "cifar10", "svhn"]:
        return get_images_config(dataset, model, use_baseline)

    else:
        assert False, f"Invalid dataset {dataset}"


def get_2d_config(dataset, model, use_baseline):
    assert model in ["flat-realnvp", "maf", "sos"], f"Invalid model {model} for dataset {dataset}"

    if dataset == "2uniforms":
        if use_baseline:
            config = {
                "num_density_layers": 10,
                "g_hidden_channels": [50] * 4,
                "num_u_channels": 0
            }

        else:
            config = {
                "num_density_layers": 5,
                "g_hidden_channels": [10] * 2,
                "num_u_channels": 1,
                "st_nets": [10] * 2,
                "p_nets": [50] * 4,
                "q_nets": [50] * 4
            }

        config = {
            **config,
            "max_bad_valid_epochs": 300,
            "max_epochs": 300,
        }

    else:
        if use_baseline:
            config = {
                "num_density_layers": 20,
                "g_hidden_channels": [50] * 4,
                "num_u_channels": 0
            }

        else:
            config = {
                "num_density_layers": 5,
                "g_hidden_channels": [50] * 4,
                "num_u_channels": 1,
                "st_nets": [10] * 2,
                "p_nets": [50] * 4,
                "q_nets": [50] * 4,
            }

        config = {
            **config,
            "max_bad_valid_epochs": 500,
            "max_epochs": 500
        }

    config = {
        **config,

        "batch_norm": use_baseline,
        "dequantize": False,

        "train_batch_size": 1000,
        "valid_batch_size": 1000,
        "test_batch_size": 10000,

        "opt": "adam",
        "lr": 1e-3,
        "weight_decay": 0.,
        "epochs_per_test": 5,

        "num_train_elbo_samples": 10,
        "num_valid_elbo_samples": 10,
        "num_test_elbo_samples": 100
    }

    if model == "sos":
        warnings.warn("Overriding `num_density_layers`")
        config["num_density_layers"] = 3 if use_baseline else 2
        config["num_polynomials_per_layer"] = 2
        config["polynomial_degree"] = 4

        config["batch_norm"] = True

        config["st_nets"] = [40] * 2
        config["p_nets"] = [40] * 4
        config["q_nets"] =  [40] * 4

    return config


def get_uci_config(dataset, model, use_baseline):
    assert model in ["flat-realnvp", "maf"], f"Invalid model {model} for dataset {dataset}"

    if dataset in ["gas", "power"]:
        if use_baseline:
            config = {
                "num_density_layers": 10,
                "g_hidden_channels": [200] * 2,
                "num_u_channels": 0
            }

        else:
            config = {
                "num_density_layers": 10,
                "g_hidden_channels": [100] * 2,
                "num_u_channels": 2,
                "st_nets": [100] * 2,
                "p_nets": [200] * 2,
                "q_nets": [200] * 2,
            }

    elif dataset in ["hepmass", "miniboone"]:
        if use_baseline:
            config = {
                "num_density_layers": 10,
                "g_hidden_channels": [512] * 2,
                "num_u_channels": 0
            }

        else:
            config = {
                "num_density_layers": 10,
                "g_hidden_channels": [128] * 2,
                "num_u_channels": 5 if dataset == "hepmass" else 10,
                "st_nets": [128] * 2,
                "p_nets": [512] * 2,
                "q_nets": [512] * 2
            }

    config = {
        **config,

        "batch_norm": use_baseline,
        "dequantize": False,

        "train_batch_size": 1000,
        "valid_batch_size": 5000,
        "test_batch_size": 5000,

        "opt": "adam",
        "lr": 1e-3,
        "weight_decay": 0.,
        "max_bad_valid_epochs": 30,
        "max_epochs": 1000,
        "epochs_per_test": 5,

        "num_train_elbo_samples": 1,
        "num_valid_elbo_samples": 5,
        "num_test_elbo_samples": 10
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

        config["train_batch_size"] = 100
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
                "num_u_channels": 0
            }

        else:
            config = {
                "num_scales": 3,
                "num_steps_per_scale": 32,
                "g_num_hidden_channels": 256,
                "num_u_channels": 1,
                "st_nets": 64,
                "p_nets": 128,
                "q_nets": 128
            }

        config["train_batch_size"] = 64
        config["opt"] = "adamax"
        config["lr"] = 5e-4

        if dataset in ["cifar10"]:
            config["weight_decay"] = 0.15
        else:
            config["weight_decay"] = 0.

        config["centering_tf_scale"] = 256

    else:
        assert False, f"Invalid model {model} for dataset {dataset}"

    config = {
        **config,

        "batch_norm": use_baseline,
        "dequantize": True,

        "valid_batch_size": 500,
        "test_batch_size": 500,

        "max_bad_valid_epochs": 50,
        "max_epochs": 1000,
        "epochs_per_test": 1,

        "num_train_elbo_samples": 1,
        "num_valid_elbo_samples": 5,
        "num_test_elbo_samples": 10
    }

    return config
