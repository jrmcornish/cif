from config import get_config


def test_baseline_glow_config():
    config = get_config(dataset="cifar10", model="glow", use_baseline=True)

    true_config = {
        "schema_type": "glow",
        "use_cond_affine": False,
        "pure_cond_affine": False,
        "num_scales": 3,
        "num_steps_per_scale": 32,
        "g_num_hidden_channels": 512,
        "num_u_channels": 0,
        "valid_batch_size": 500,
        "test_batch_size": 500,
        "early_stopping": False,
        "train_batch_size": 64,
        "opt": "adamax",
        "lr": 0.0005,
        "weight_decay": 0.1,
        "centering_tf_scale": 256,
        "dequantize": True,
        "act_norm": False, # TODO: Should be True
        "batch_norm": True, # TODO: Should be False
        "batch_norm_apply_affine": True,
        "batch_norm_use_running_averages": True,
        "batch_norm_momentum": 0.1,
        "lr_schedule": "none",
        "max_bad_valid_epochs": 50,
        "max_grad_norm": None,
        "max_epochs": 1000,
        "epochs_per_test": 1,
        "num_valid_elbo_samples": 1,
        "num_test_elbo_samples": 1
    }

    assert config == true_config


def test_lgf_glow_config():
    config = get_config(dataset="cifar10", model="glow", use_baseline=False)

    true_config = {
        "schema_type": "glow",
        "use_cond_affine": True,
        "pure_cond_affine": False,
        "num_scales": 2,
        "num_steps_per_scale": 32,
        "g_num_hidden_channels": 256,
        "num_u_channels": 1,
        "st_nets": 64,
        "p_nets": 128,
        "q_nets": 128,
        "valid_batch_size": 100,
        "test_batch_size": 100,
        "early_stopping": False,
        "train_batch_size": 64,
        "opt": "adamax",
        "lr": 0.0005,
        "weight_decay": 0.1,
        "centering_tf_scale": 256,
        "dequantize": True,
        "act_norm": False,
        "batch_norm": True,
        "batch_norm_apply_affine": False,
        "batch_norm_use_running_averages": True,
        "batch_norm_momentum": 0.1,
        "lr_schedule": "none",
        "max_bad_valid_epochs": 50,
        "max_grad_norm": None,
        "max_epochs": 1000,
        "epochs_per_test": 1,
        "num_valid_elbo_samples": 5,
        "num_test_elbo_samples": 10,
    }

    assert config == true_config


def test_lgf_realnvp_config():
    config = get_config(dataset="mnist", model="realnvp", use_baseline=False)

    true_config = {
        "schema_type": "multiscale-realnvp",
        "use_cond_affine": True,
        "pure_cond_affine": False,
        "g_hidden_channels": [
            64,
            64,
            64,
            64
        ],
        "num_u_channels": 1,
        "st_nets": [
            8,
            8
        ],
        "p_nets": [
            64,
            64
        ],
        "q_nets": [
            64,
            64
        ],
        "early_stopping": True,
        "train_batch_size": 100,
        "valid_batch_size": 500,
        "test_batch_size": 500,
        "opt": "adam",
        "lr": 0.0001,
        "weight_decay": 0.0,
        "logit_tf_lambda": 1e-06,
        "logit_tf_scale": 256,
        "dequantize": True,
        "act_norm": False,
        "batch_norm": True,
        "batch_norm_apply_affine": False,
        "batch_norm_use_running_averages": True,
        "batch_norm_momentum": 0.1,
        "lr_schedule": "none",
        "max_bad_valid_epochs": 50,
        "max_grad_norm": None,
        "max_epochs": 1000,
        "epochs_per_test": 1,
        "num_valid_elbo_samples": 5,
        "num_test_elbo_samples": 10,
    }

    assert true_config == config


def test_baseline_realnvp_config():
    config = get_config(dataset="mnist", model="realnvp", use_baseline=True)

    true_config = {
        "schema_type": "multiscale-realnvp",
        "use_cond_affine": False,
        "pure_cond_affine": False,
        "g_hidden_channels": [
            64,
            64,
            64,
            64,
            64,
            64,
            64,
            64
        ],
        "num_u_channels": 0,
        "early_stopping": True,
        "train_batch_size": 100,
        "valid_batch_size": 500,
        "test_batch_size": 500,
        "opt": "adam",
        "lr": 0.0001,
        "weight_decay": 0.0,
        "logit_tf_lambda": 1e-06,
        "logit_tf_scale": 256,
        "dequantize": True,
        "act_norm": False,
        "batch_norm": True,
        "batch_norm_apply_affine": True,
        "batch_norm_use_running_averages": True,
        "batch_norm_momentum": 0.1,
        "lr_schedule": "none",
        "max_bad_valid_epochs": 50,
        "max_grad_norm": None,
        "max_epochs": 1000,
        "epochs_per_test": 1,
        "num_valid_elbo_samples": 1,
        "num_test_elbo_samples": 1,
    }

    assert true_config == config
