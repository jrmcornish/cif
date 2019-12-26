from .dsl import group, base, provides


group(
    "uci",
    [
        "gas",
        "hepmass",
        "power",
        "miniboone"
    ]
)


@base
def config(dataset, use_baseline):
    if dataset in ["gas", "power"]:
        num_u_channels = 2
    elif dataset == "hepmass":
        num_u_channels = 5
    else:
        num_u_channels = 10

    return {
        "num_u_channels": num_u_channels,
        "use_cond_affine": True,
        "pure_cond_affine": False,

        "dequantize": False,

        "batch_norm": True,
        "batch_norm_apply_affine": use_baseline,
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

        "num_train_elbo_samples": 1,
        "num_valid_elbo_samples": 5,
        "num_test_elbo_samples": 10,
    }


@provides("resflow")
def resflow(dataset, model, use_baseline):
    config = {
        "schema_type": "resflow",
        "num_density_layers": 10,
        "hidden_channels": [128] * 4,

        "batch_norm": False,

        "st_nets": [10] * 2,
        "p_nets": [10] * 2,
        "q_nets": [10] * 2
    }

    if not use_baseline:
        config["valid_batch_size"] = 1000
        config["test_batch_size"] = 1000

    return config


@provides("maf")
def maf(dataset, model, use_baseline):
    if dataset in ["gas", "power"]:
        config = {
            "num_density_layers": 10,
            "ar_map_hidden_channels": [200] * 2 if use_baseline else [100] * 2,

            "st_nets": [100] * 2,
            "p_nets": [200] * 2,
            "q_nets": [200] * 2,
        }

    elif dataset in ["hepmass", "miniboone"]:
        config = {
            "num_density_layers": 10,
            "ar_map_hidden_channels": [512] * 2 if use_baseline else [128] * 2,

            "st_nets": [128] * 2,
            "p_nets": [512] * 2,
            "q_nets": [512] * 2
        }

    config["schema_type"] = "maf"

    return config


@provides("sos")
def sos(dataset, model, use_baseline):
    assert use_baseline

    return {
        "schema_type": "sos",

        "num_density_layers": 8,
        "g_hidden_channels": [200] * 2,
        "num_polynomials_per_layer": 5,
        "polynomial_degree": 4,

        "lr": 1e-3,
        "opt": "sgd"
    }


@provides("nsf-ar")
def nsf(dataset, model, use_baseline):
    common = {
        "schema_type": "nsf",

        "autoregressive": True,

        "batch_norm": False,
        "tail_bound": 3,

        "opt": "adam",
        "lr_schedule": "cosine",
        "weight_decay": 0.,
        "early_stopping": False,
        "max_grad_norm": 5,

        "valid_batch_size": 5000,
        "test_batch_size": 5000,

        "epochs_per_test": 5
    }

    if dataset in ["power", "gas", "hepmass"]:
        dropout = {"power": 0., "gas": 0.1, "hepmass": 0.2}[dataset]

        dset_size = {"power": 1_615_917, "gas": 852_174, "hepmass": 315_123}[dataset]
        batch_size = 512
        train_steps = 400_000

        config = {
            "lr": 0.0005,
            "num_density_layers": 10 if use_baseline else 5,
            "num_hidden_layers": 2,
            "num_hidden_channels": 256,
            "num_bins": 8,
            "dropout_probability": dropout,

            "num_u_channels": 2 if dataset in ["power", "gas"] else 5,
            "st_nets": 2 * [180],
            "p_nets": 2 * [360],
            "q_nets": 2 * [360]
        }

    elif dataset == "miniboone":
        dset_size = 29_556
        batch_size = 64
        train_steps = 250_000

        config = {
            "lr": 0.0003,
            "num_density_layers": 10 if use_baseline else 4,
            "num_hidden_layers": 1,
            "num_hidden_channels": 64,
            "num_bins": 4,
            "dropout_probability": 0.2,

            "num_u_channels": 10,
            "st_nets": 2 * [64],
            "p_nets": 2 * [128],
            "q_nets": 2 * [128]
        }

    else:
        assert False, f"Invalid dataset {dataset}"

    steps_per_epoch = dset_size // batch_size
    epochs = int(train_steps/steps_per_epoch + .5) # Round up

    return {
        **common,
        **config,
        "max_epochs": epochs,
        "train_batch_size": batch_size
    }


@provides("nsf-old")
def nsf_old(dataset, model, use_baseline):
    if dataset in ["power", "gas"]:
        config = {
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

    return {
        "schema_type": "nsf",

        **config,

        "tail_bound": 3,
        "autoregressive": True,
        "batch_norm": False,
        "max_grad_norm": 5,

        "lr_schedule": "cosine"
    }
