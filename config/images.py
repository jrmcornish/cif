import warnings

from .dsl import group, base, provides


group(
    "images",
    [
        "mnist",
        "fashion-mnist",
        "cifar10",
        "svhn"
    ]
)


@base
def config(dataset, use_baseline):
    return {
        "num_u_channels": 1,
        "use_cond_affine": True,
        "pure_cond_affine": False,

        "dequantize": True,

        "act_norm": False,
        "batch_norm": True,
        "batch_norm_apply_affine": use_baseline,
        "batch_norm_use_running_averages": True,
        "batch_norm_momentum": 0.1,

        "lr_schedule": "none",
        "max_bad_valid_epochs": 50,
        "max_grad_norm": None,
        "max_epochs": 1000,
        "epochs_per_test": 1,
        "early_stopping": True,

        "train_objective": "iwae",
        "num_train_importance_samples": 1,
        "num_valid_importance_samples": 5,
        "num_test_importance_samples": 10
    }


@provides("bernoulli-vae")
def bernoulli_vae(dataset, model, use_baseline):
    assert not use_baseline, "Cannot use baseline model for this config"

    return {
        "schema_type": "bernoulli-vae",

        "dequantize": False,
        "binarize_scale": 255,

        "logit_net": [200]*2,
        "q_nets": [200]*2,
        "num_z_channels": 50,

        "train_batch_size": 100,
        "valid_batch_size": 500,
        "test_batch_size": 500,
        "opt": "adam",
        "lr": 1e-4,
        "weight_decay": 0.
    }


@provides("realnvp")
def realnvp(dataset, model, use_baseline):
    config = {
        "schema_type": "multiscale-realnvp",

        "g_hidden_channels": [64]*8 if use_baseline else [64]*4,

        "st_nets": [8] * 2,
        "p_nets": [64] * 2,
        "q_nets": [64] * 2,

        "train_batch_size": 100,
        "valid_batch_size": 500,
        "test_batch_size": 500,
        "opt": "adam",
        "lr": 1e-4,
        "weight_decay": 0.
    }

    if dataset in ["cifar10", "svhn"]:
        config["logit_tf_lambda"] = 0.05
        config["logit_tf_scale"] = 256

    elif dataset in ["mnist", "fashion-mnist"]:
        config["logit_tf_lambda"] = 1e-6
        config["logit_tf_scale"] = 256

    return config


@provides("glow")
def glow(dataset, model, use_baseline):
    warnings.warn("Glow may quickly diverge for certain random seeds - if this happens just retry. This behaviour appears to be consistent with that in https://github.com/openai/glow and https://github.com/y0ast/Glow-PyTorch")

    if use_baseline:
        config = {
            "num_scales": 3,
            "num_steps_per_scale": 32,
            "g_num_hidden_channels": 512,
            "valid_batch_size": 500,
            "test_batch_size": 500
        }

    else:
        config = {
            "num_scales": 2,
            "num_steps_per_scale": 32,
            "g_num_hidden_channels": 256,
            "st_nets": 64,
            "p_nets": 128,
            "q_nets": 128,
            "valid_batch_size": 100,
            "test_batch_size": 100
        }

    config["schema_type"] = "glow"

    config["early_stopping"] = False
    config["train_batch_size"] = 64
    config["opt"] = "adamax"
    config["lr"] = 5e-4

    if dataset in ["cifar10"]:
        config["weight_decay"] = 0.1
    else:
        config["weight_decay"] = 0.

    config["centering_tf_scale"] = 256

    return config


# NOTE: We differ from the ResFlows paper in the following ways:
#
#   * We do not do Polyak averaging
#   * We have weight_decay = 0. (Note that if we want to set this ourselves,
#     we would need to correct for the fact that their objective is in bpd.)
#   * Our logit transform for MNIST has parameter 1e-6 instead of 1e-5 (although
#     1e-6 is what is used in the `residual-flows` implementation).
#
# Unlike the ResFlows paper, we do not do Polyak averaging
# Additionally, although not mentioned in the ResFlows paper, we differ 
# from the `residual-flows` implementation in the following ways:
#
#   * We use default hyperparameters for Adam, i.e. betas = (0.9, 0.999). In
#     `residual-flows`, they use betas = (0.9, 0.99).
#   * We do not use learning rate warmup
#   * They train on the full training set and take the model with best test score,
#     whereas we use a validation set that is extracted from the training set
#   * They resize MNIST to 32x32 whereas we keep the dimension at 28x28
#   * They do some kind of gradient normalisation, as well as gradient clipping
#
@provides("resflow-small")
def resflow(dataset, model, use_baseline):
    logit_tf_lambda = {
        "mnist": 1e-6,
        "fashion-mnist": 1e-6,
        "cifar10": 0.05,
    }[dataset]

    return {
        "schema_type": "multiscale-resflow",

        "train_batch_size": 64,
        "valid_batch_size": 128,
        "test_batch_size": 128,
        "epochs_per_test": 5,

        "opt": "adam",
        "lr": 1e-3,
        "weight_decay": 0.,

        "logit_tf_lambda": logit_tf_lambda,
        "logit_tf_scale": 256,

        "batch_norm": False,
        "act_norm": True,

        "reduce_memory": True,
        "scales": [4] * 3,
        "num_hidden_channels": 128,
        "lipschitz_constant": 0.98,
        "max_train_lipschitz_iters": None,
        "max_test_lipschitz_iters": None,
        "lipschitz_tolerance": 1e-3,
        "num_output_fc_blocks": 4,
        "output_fc_hidden_channels": [64] * 2,

        "st_nets": [32] * 2,
        "p_nets": [32] * 2,
        "q_nets": [32] * 2
    }


# Larger version of "resflow" designed to have comparable parameters to our method
@provides("resflow-big")
def resflow(dataset, model, use_baseline):
    assert use_baseline, "Must use baseline model for this config"

    logit_tf_lambda = {
        "mnist": 1e-6,
        "fashion-mnist": 1e-6,
        "cifar10": 0.05,
    }[dataset]

    return {
        "schema_type": "multiscale-resflow",

        "train_batch_size": 64,
        "valid_batch_size": 128,
        "test_batch_size": 128,
        "epochs_per_test": 5,

        "opt": "adam",
        "lr": 1e-3,
        "weight_decay": 0.,

        "logit_tf_lambda": logit_tf_lambda,
        "logit_tf_scale": 256,

        "batch_norm": False,
        "act_norm": True,

        "reduce_memory": True,
        "scales": [6] * 3,
        "num_hidden_channels": 256,
        "lipschitz_constant": 0.98,
        "max_train_lipschitz_iters": None,
        "max_test_lipschitz_iters": None,
        "lipschitz_tolerance": 1e-3,
        "num_output_fc_blocks": 4,
        "output_fc_hidden_channels": [128] * 2,

        "st_nets": [32] * 2,
        "p_nets": [32] * 2,
        "q_nets": [32] * 2
    }


# Parameters used in "Residual flows for invertible generative modeling" by Chen et al. (2009)
@provides("resflow-chen")
def resflow(dataset, model, use_baseline):
    assert use_baseline, "Must use baseline model for this config"

    logit_tf_lambda = {
        "mnist": 1e-6,
        "fashion-mnist": 1e-6,
        "cifar10": 0.05,
    }[dataset]

    return {
        "schema_type": "multiscale-resflow",

        "train_batch_size": 64,
        "valid_batch_size": 128,
        "test_batch_size": 128,
        "epochs_per_test": 5,

        "opt": "adam",
        "lr": 1e-3,
        "weight_decay": 0.,

        "logit_tf_lambda": logit_tf_lambda,
        "logit_tf_scale": 256,

        "batch_norm": False,
        "act_norm": True,

        "reduce_memory": True,
        "scales": [16] * 3,
        "num_hidden_channels": 512,
        "lipschitz_constant": 0.98,
        "max_train_lipschitz_iters": None,
        "max_test_lipschitz_iters": None,
        "lipschitz_tolerance": 1e-3,
        "num_output_fc_blocks": 4,
        "output_fc_hidden_channels": [128] * 2,
    }

