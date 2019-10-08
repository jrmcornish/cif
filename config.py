def two_uniforms(baseline=False):
    config = {
        "dataset": "2uniforms",

        "model": "maf",

        "train_batch_size": 1000,
        "valid_batch_size": 1000,
        "test_batch_size": 10000,

        "lr": 1e-3,
        "max_bad_valid_epochs": 300,
        "max_epochs": 300,
        "epochs_per_test": 5,
        "epochs_per_checkpoint": 50
    }

    if baseline:
        config = {
            **config,
            "num_density_layers": 10,
            "g_nets": {
                "hidden_units": [50] * 4
            },
            "num_u_channels": 0
        }

    else:
        config = {
            **config,
            "num_density_layers": 5,
            "g_nets": {
                "hidden_units": [10] * 2
            },
            "num_u_channels": 1,
            "st_nets": {
                "hidden_units": [10] * 2,
            },
            "p_nets": {
                "hidden_units": [50] * 4,
            },
            "q_nets": {
                "hidden_units": [50] * 4,
            },
            "num_train_elbo_samples": 10,
            "num_test_elbo_samples": 100,
            "num_valid_elbo_samples": 100,
        }

    return config


def two_d(dataset, baseline=False):
    config = {
        "dataset": dataset,

        "model": "maf",

        "train_batch_size": 1000,
        "valid_batch_size": 1000,
        "test_batch_size": 10000,

        "lr": 1e-3,
        "max_bad_valid_epochs": 500,
        "max_epochs": 500,
        "epochs_per_test": 5,
        "epochs_per_checkpoint": 50
    }

    if baseline:
        config = {
            **config,
            "num_density_layers": 20,
            "g_nets": {
                "hidden_units": [50] * 4
            },
            "num_u_channels": 0
        }

    else:
        config = {
            **config,
            "num_density_layers": 5,
            "g_nets": {
                "hidden_units": [50] * 4
            },
            "num_u_channels": 1,
            "st_nets": {
                "hidden_units": [10] * 2,
            },
            "p_nets": {
                "hidden_units": [50] * 4,
            },
            "q_nets": {
                "hidden_units": [50] * 4,
            },
            "num_train_elbo_samples": 10,
            "num_test_elbo_samples": 100,
            "num_valid_elbo_samples": 100,
        }

    return config


def uci(dataset, baseline=False):
    config = {
        "dataset": dataset,

        "model": "maf",

        "train_batch_size": 1000,
        "valid_batch_size": 5000,
        "test_batch_size": 5000,

        "lr": 1e-3,
        "max_bad_valid_epochs": 30,
        "max_epochs": 1000,
        "epochs_per_test": 5,
        "epochs_per_checkpoint": 50
    }

    if dataset in ["gas", "power"]:
        if baseline:
            config = {
                **config,
                "num_density_layers": 10,
                "g_nets": {
                    "hidden_units": [200] * 2
                },
                "num_u_channels": 0
            }

        else:
            config = {
                **config,
                "num_density_layers": 10,
                "g_nets": {
                    "hidden_units": [100] * 2
                },
                "num_u_channels": 2,
                "st_nets": {
                    "hidden_units": [100] * 2,
                },
                "p_nets": {
                    "hidden_units": [200] * 2,
                },
                "q_nets": {
                    "hidden_units": [200] * 2,
                },
                "num_train_elbo_samples": 1,
                "num_valid_elbo_samples": 5,
                "num_test_elbo_samples": 10
            }

    elif dataset in ["hepmass", "miniboone"]:
        if baseline:
            config = {
                **config,
                "num_density_layers": 10,
                "g_nets": {
                    "hidden_units": [512] * 2
                },
                "num_u_channels": 0
            }

        else:
            config = {
                **config,
                "num_density_layers": 10,
                "g_nets": {
                    "hidden_units": [128] * 2
                },
                "num_u_channels": 5 if dataset == "hepmass" else 10,
                "st_nets": {
                    "hidden_units": [128] * 2,
                },
                "p_nets": {
                    "hidden_units": [512] * 2,
                },
                "q_nets": {
                    "hidden_units": [512] * 2,
                },
                "num_train_elbo_samples": 1,
                "num_valid_elbo_samples": 5,
                "num_test_elbo_samples": 10
            }

    return config


def images(dataset, baseline=False):
    config = {
        "dataset": dataset,
        "model": "multiscale-realnvp",

        "train_batch_size": 100,
        "valid_batch_size": 500,
        "test_batch_size": 500,

        "lr": 1e-4,
        "max_bad_valid_epochs": 50,
        "max_epochs": 1000,
        "epochs_per_test": 1,
        "epochs_per_checkpoint": 50
    }

    if baseline:
        config = {
            **config,
            "g_nets": {
                "hidden_channels": [64] * 8
            },
            "num_u_channels": 0
        }

    else:
        config = {
            **config,
            "g_nets": {
                "hidden_channels": [64] * 4
            },
            "num_u_channels": 1,
            "st_nets": {
                "hidden_channels": [8] * 2
            },
            "p_nets": {
                "hidden_channels": [64] * 2
            },
            "q_nets": {
                "hidden_channels": [64] * 2
            },
            "num_train_elbo_samples": 1,
            "num_valid_elbo_samples": 5,
            "num_test_elbo_samples": 10
        }

    return config
