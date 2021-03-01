from .dsl import group, base, provides, GridParams


group(
    "tabular",
    [
        "gas",
        "hepmass",
        "power",
        "miniboone",
        "bsds300"
    ]
)


@base
def config(dataset, use_baseline):
    num_u_channels = {
        "gas": 2,
        "power": 2,
        "hepmass": 5,
        "miniboone": 10,
        "bsds300": 15
    }[dataset]

    return {
        "num_u_channels": num_u_channels,
        "use_cond_affine": True,
        "pure_cond_affine": False,

        "dequantize": False,

        "act_norm": False,
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
        "max_bad_valid_epochs": 50,
        "max_epochs": 2000,
        "max_grad_norm": None,
        "epochs_per_test": 5,

        "train_objective": "iwae",
        "num_train_importance_samples": 1,
        "num_valid_importance_samples": 5,
        "num_test_importance_samples": 10,
    }


@provides("resflow")
def resflow(dataset, model, use_baseline):
    config = {
        "schema_type": "flat-resflow",
        "num_density_layers": 10,
        "hidden_channels": [128] * 4,
        "lipschitz_constant": 0.9,
        "max_train_lipschitz_iters": 5,
        "max_test_lipschitz_iters": 200,
        "lipschitz_tolerance": None,
        "reduce_memory": False,

        "act_norm": False,
        "batch_norm": False,

        "st_nets": [10] * 2,
        "p_nets": [10] * 2,
        "q_nets": [10] * 2
    }

    if not use_baseline:
        config["valid_batch_size"] = 1000
        config["test_batch_size"] = 1000

    return config


# TODO: Could also try p_nets=[128]*4, st_nets=[10]*2
@provides("cond-affine")
def cond_affine(dataset, model, use_baseline):
    assert not use_baseline

    return {
        "schema_type": "cond-affine",
        "num_density_layers": 10,

        "batch_norm": False,

        "st_nets": [128] * 2,
        "p_nets": [128] * 2,
        "q_nets": GridParams([10] * 2, [100] * 4)
    }


# Gives z = x + mu(x) + sigma(x)*w, w ~ N(0, I)
@provides("linear-cond-affine-like-resflow")
def linear_cond_affine_like_resflow(dataset, model, use_baseline):
    assert not use_baseline

    num_u_channels = {
        "miniboone": 43,
        "hepmass": 21,
        "gas": 8,
        "power": 6
    }[dataset]

    config = {
        "schema_type": "cond-affine",
        "num_density_layers": 10,

        "num_u_channels": num_u_channels,

        "batch_norm": False,

        "s_nets": "fixed-constant",
        "t_nets": "identity",
        "p_nets": [128] * 4,
        "q_nets": GridParams([10] * 2, [100] * 4)
    }

    if not use_baseline:
        config["valid_batch_size"] = 1000
        config["test_batch_size"] = 1000

    return config


# Gives z = x + t(mu(x) + sigma(x)*w), w ~ N(0, I)
@provides("nonlinear-cond-affine-like-resflow")
def nonlinear_cond_affine_like_resflow(dataset, model, use_baseline):
    assert not use_baseline

    num_u_channels = {
        "miniboone": 43,
        "hepmass": 21,
        "gas": 8,
        "power": 6
    }[dataset]


    config = {
        "schema_type": "cond-affine",
        "num_density_layers": 10,

        "num_u_channels": num_u_channels,

        "batch_norm": False,

        "s_nets": "fixed-constant",
        "t_nets": [128] * 2,
        "p_nets": [128] * 2,
        "q_nets": GridParams([10] * 2, [100] * 4)
    }

    if not use_baseline:
        config["valid_batch_size"] = 1000
        config["test_batch_size"] = 1000

    return config


@provides("resflow-no-g")
def resflow_no_g(dataset, model, use_baseline):
    assert not use_baseline
    assert dataset == "miniboone"

    config = {
        "schema_type": "flat-resflow",
        "num_density_layers": 10,
        "hidden_channels": None,
        "lipschitz_constant": None,

        "batch_norm": False,

        "use_cond_affine": True,
        "pure_cond_affine": True,

        "num_u_channels": 43,
        "st_nets": [100] * 4,
        "p_mu_nets": "identity",
        "p_sigma_nets": "learned-constant",
        "q_nets": [100] * 4
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

    elif dataset in ["hepmass", "miniboone", "bsds300"]:
        config = {
            "num_density_layers": 10,
            "ar_map_hidden_channels": [512] * 2,

            "st_nets": [128] * 2,
            "p_nets": [128] * 2,
            "q_nets": [128] * 2
        }

    config["schema_type"] = "maf"
    config["batch_norm"] = use_baseline

    if dataset == "bsds300":
        config["lr"] = 1e-4

    return config


@provides("realnvp")
def realnvp(dataset, model, use_baseline):
    return {
        "schema_type": "flat-realnvp",

        "num_density_layers": 20,
        "coupler_shared_nets": True,
        "coupler_hidden_channels": [1024] * 2,

        "st_nets": [100] * 2,
        "p_nets": [100] * 2,
        "q_nets": [100] * 2,
    }


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
        "num_density_layers": 10,
        "tail_bound": 3,

        "batch_norm": False,

        "opt": "adam",
        "lr_schedule": "cosine",
        "weight_decay": 0.,
        "early_stopping": False,
        "max_grad_norm": 5,

        "valid_batch_size": 5000,
        "test_batch_size": 5000,

        "epochs_per_test": 5,

    }

    if dataset in ["power", "gas", "hepmass", "bsds300"]:
        dropout = {"power": 0., "gas": 0.1, "hepmass": 0.2, "bsds300": 0.2}[dataset]

        dset_size = {"power": 1_615_917, "gas": 852_174, "hepmass": 315_123, "bsds300": 1_000_000}[dataset]
        batch_size = 512
        train_steps = 400_000

        config = {
            "lr": 0.0005,
            "num_hidden_layers": 2,
            "num_hidden_channels": 512 if dataset == "bsds300" else 256,
            "num_bins": 8,
            "dropout_probability": dropout,
            
            "st_nets": [100] * 3,
            "p_nets": [200] * 3,
            "q_nets": [10] * 2,
        }

    elif dataset == "miniboone":
        dset_size = 29_556
        batch_size = 64
        train_steps = 250_000

        config = {
            "lr": 0.0003,
            "num_hidden_layers": 1,
            "num_hidden_channels": 64,
            "num_bins": 4,
            "dropout_probability": 0.2,

            "st_nets": [25] * 3,
            "p_nets": [50] * 3,
            "q_nets": [10] * 2,
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

