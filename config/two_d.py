from .dsl import group, base, provides, GridParams


group(
    "2d",
    [
        "2uniforms",
        "2lines",
        "8gaussians",
        "checkerboard",
        "2spirals",
        "rings"
    ]
)


@base
def config(dataset, use_baseline):
    return {
        "num_u_channels": 1,
        "use_cond_affine": not use_baseline,

        "dequantize": False,

        "batch_norm": False,

        "max_epochs": 2000,
        "max_grad_norm": None,
        "early_stopping": True,
        "max_bad_valid_epochs": 100,
        "train_batch_size": 1000,
        "valid_batch_size": 1000,
        "test_batch_size": 10000,

        "opt": "adam",
        "lr": 1e-3,
        "lr_schedule": "none",
        "weight_decay": 0.,
        "epochs_per_test": 25,

        "num_train_elbo_samples": 10,
        "num_valid_elbo_samples": 10,
        "num_test_elbo_samples": 100
    }


@provides(
    "dlgm-deep",
    "dlgm-deep-nvp",
    "dlgm-shallow",
    "dlgm-shallow-nvp"
)
def dlgm(dataset, model, use_baseline):
    assert not use_baseline

    modifiers = model.split("-")[1:]

    depth = 8

    if "deep" in modifiers:
        layers = 4
    elif "shallow" in modifiers:
        layers = 1

    if "nvp" in modifiers:
        st = {
            "s_nets": "identity",
            "t_nets": "fixed-constant"
        }

    else:
        st = {
            "s_nets": "fixed-constant",
            "t_nets": "identity"
        }

    net_spec = [10] * (depth // layers)

    return {
        "schema_type": "cond-affine",

        "num_u_channels": 2,

        **st,

        "num_density_layers": layers,
        "p_nets": net_spec,
        "q_nets": net_spec
    }


@provides(
    "cond-affine-deep-s",
    "cond-affine-deep-t",
    "cond-affine-deep-st",
    "cond-affine-shallow-s",
    "cond-affine-shallow-t",
    "cond-affine-shallow-st"
)
def cond_affine(dataset, model, use_baseline):
    assert not use_baseline

    modifiers = model.split("-")[2:]

    depth = 8

    if "deep" in modifiers:
        layers = 2
    elif "shallow" in modifiers:
        layers = 1

    net_spec = [10] * (depth // (layers*2))

    if "s" in modifiers:
        st = {
            "s_nets": net_spec,
            "t_nets": "fixed-constant"
        }
    elif "t" in modifiers:
        st = {
            "s_nets": "fixed-constant",
            "t_nets": net_spec
        }
    elif "st" in modifiers:
        st = {
            "st_nets": net_spec
        }

    return {
        "schema_type": "cond-affine",

        "num_u_channels": 2,

        **st,

        "num_density_layers": layers,
        "p_nets": net_spec,
        "q_nets": net_spec
    }


@provides("affine")
def affine(dataset, model, use_baseline):
    assert use_baseline
    return {
        "schema_type": "affine",
        "num_density_layers": 10
    }


# TODO: Make 2uniforms match paper
@provides("maf")
def maf(dataset, model, use_baseline):
    return  {
        "schema_type": "maf",

        "num_density_layers": 20 if use_baseline else 5,
        "g_hidden_channels": [50] * 4,

        "st_nets": [10] * 2,
        "p_nets": [50] * 4,
        "q_nets": [50] * 4,
    }


@provides("realnvp")
def realnvp(dataset, model, use_baseline):
    return {
        "schema_type": "flat-realnvp",

        "num_density_layers": 2,
        "coupler_shared_nets": True,
        "coupler_hidden_channels": [10] * 2,

        "use_cond_affine": False,

        "st_nets": [10] * 2,
        "p_nets": [10] * 2,
        "q_nets": [4] * 1,
    }


@provides("sos")
def sos(dataset, model, use_baseline):
    return {
        "schema_type": "sos",
        
        "num_density_layers": 3 if use_baseline else 2,
        "num_polynomials_per_layer": 2,
        "polynomial_degree": 4,

        "st_nets": [40] * 2,
        "p_nets": [40] * 4,
        "q_nets":  [40] * 4
    }


@provides("planar")
def planar(dataset, model, use_baseline):
    return {
        "schema_type": "planar",

        "num_density_layers": 10,

        "use_cond_affine": False,
        "cond_hidden_channels": [10] * 2,

        "p_nets": [50] * 4,
        # TODO: Make [50] * 4
        "q_nets": [10] * 2
    }


@provides("nsf")
def nsf(dataset, model, use_baseline):
    return {
        "schema_type": "nsf",

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


@provides("bnaf")
def bnaf(dataset, model, use_baseline):
    return {
        "schema_type": "bnaf",

        "num_density_layers": 1,
        "num_hidden_layers": 2,
        "hidden_channels_factor": 50 if use_baseline else 45,
        "activation": "soft-leaky-relu",

        "st_nets": [24] * 2,
        "p_nets": [24] * 3,
        "q_nets": [24] * 3
    }


@provides("ffjord")
def ffjord(dataset, model, use_baseline):
    return {
        "schema_type": "ffjord",

        "num_density_layers": 1,
        "hidden_channels": [64] * 3,
        "numerical_tolerance": 1e-5,

        "st_nets": [24] * 2,
        "p_nets": [24] * 3,
        "q_nets": [24] * 3
    }
