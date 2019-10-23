def two_uniforms(baseline=False):
    config = {
        "dataset": "2uniforms",

        "model": "pure-cond-affine-mlp",

        "train_batch_size": 1000,
        "valid_batch_size": 1000,
        "test_batch_size": 10000,

        "lr": 1e-3,
        "max_bad_valid_epochs": 300,
        "max_epochs": 300,
        "epochs_per_test": 5
    }

    if baseline:
        config = {
            **config,
            "num_density_layers": 10,
            "g_nets": [50] * 4,
            "num_u_channels": 0
        }

    else:
        config = {
            **config,

            "num_density_layers": 5,
            "num_u_channels": 1,

            "g_nets": [50] * 4,
            "st_nets": [10] * 2,
            "p_nets": [50] * 4,
            "q_nets": [50] * 4,

            "num_train_elbo_samples": 1,
            "num_valid_elbo_samples": 5,
            "num_test_elbo_samples": 100
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
        "epochs_per_test": 5
    }

    if baseline:
        config = {
            **config,
            "num_density_layers": 20,
            "g_nets": [50] * 4,
            "num_u_channels": 0
        }

    else:
        config = {
            **config,
            "num_density_layers": 5,
            "g_nets": [50] * 4,
            "num_u_channels": 1,
            "st_nets": [10] * 2,
            "p_nets": [50] * 4,
            "q_nets": [50] * 4,
            "num_train_elbo_samples": 1,
            "num_valid_elbo_samples": 5,
            "num_test_elbo_samples": 100
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
        "epochs_per_test": 5
    }

    if dataset in ["gas", "power"]:
        if baseline:
            config = {
                **config,
                "num_density_layers": 10,
                "g_nets": [200] * 2,
                "num_u_channels": 0
            }

        else:
            config = {
                **config,
                "num_density_layers": 10,
                "g_nets": [100] * 2,
                "num_u_channels": 2,
                "st_nets": [100] * 2,
                "p_nets": [200] * 2,
                "q_nets": [200] * 2,
                "num_train_elbo_samples": 1,
                "num_valid_elbo_samples": 5,
                "num_test_elbo_samples": 10
            }

    elif dataset in ["hepmass", "miniboone"]:
        if baseline:
            config = {
                **config,
                "num_density_layers": 10,
                "g_nets": [512] * 2,
                "num_u_channels": 0
            }

        else:
            config = {
                **config,
                "num_density_layers": 10,
                "g_nets": [128] * 2,
                "num_u_channels": 5 if dataset == "hepmass" else 10,
                "st_nets": [128] * 2,
                "p_nets": [512] * 2,
                "q_nets": [512] * 2,
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
        "epochs_per_test": 1
    }

    if baseline:
        config = {
            **config,
            "g_nets": [64] * 8,
            "num_u_channels": 0
        }

    else:
        config = {
            **config,
            "g_nets": [64] * 4,
            "num_u_channels": 1,
            "st_nets": [8] * 2,
            "p_nets": [64] * 2,
            "q_nets": [64] * 2,
            "num_train_elbo_samples": 1,
            "num_valid_elbo_samples": 5,
            "num_test_elbo_samples": 10
        }

    return config
