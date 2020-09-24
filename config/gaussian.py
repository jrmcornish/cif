from .dsl import group, base, provides, GridParams


group(
    "gaussian",
    [
        "linear-gaussian"
    ]
)


@base
def config(dataset, use_baseline):
    assert not use_baseline
    return {
        "pure_cond_affine": False,

        "dequantize": False,

        "batch_norm": False,
        "act_norm": False,

        "train_objective": "ml-ll-ss",
        "ml_ll_geom_prob": 0.65,

        # Other possible training objectives:
        #
        # "train_objective": "iwae",
        # "iwae_num_importance_samples": 10,
        #
        # "train_objective": "rws",
        # "rws_num_importance_samples": 10,

        "max_epochs": 2000,
        "max_grad_norm": None,
        "early_stopping": True,
        "max_bad_valid_epochs": 50,
        "train_batch_size": 1000,
        "valid_batch_size": 1000,
        "test_batch_size": 10000,

        "opt": "adam",
        "lr": 1e-2,
        "lr_schedule": "none",
        "weight_decay": 0.,
        "epochs_per_test": 5,

        "num_valid_elbo_samples": 10,
        "num_test_elbo_samples": 100
    }


@provides("vae")
def vae(dataset, model, use_baseline):
    return {
        "schema_type": "gaussian-vae",
        "use_cond_affine": False,
        "num_z_channels": 1,

        # Gives a linear-Gaussian (a.k.a. factor analysis) model
        "p_mu_nets": [],
        "p_sigma_nets": "learned-constant",

        "q_nets": [10, 10]
    }
