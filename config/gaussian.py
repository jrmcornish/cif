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
        "schema_type": "cond-affine",
        "num_density_layers": 1,
        "num_u_channels": 1,

        "use_cond_affine": True,
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
        "max_bad_valid_epochs": 250,
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


@provides("linear-gaussian")
def linear_gaussian(dataset, model, use_baseline):
    return {
        # Gives a linear-Gaussian (a.k.a. factor analysis) model
        "t_nets": [],
        "s_nets": "learned-constant",
        "p_nets": "fixed-constant",

        # # Well-specified for 1D latent and 2D data
        # "q_mu_nets": [],
        # "q_sigma_nets": "learned-constant",

        # "q_nets": [10, 10],

        "q_nets": "fixed-constant"
    }


@provides("vae")
def vae(dataset, model, use_baseline):
    return {
        "st_nets": [10, 10],
        "p_nets": "fixed-constant",
        "q_nets": [10, 10],
    }
