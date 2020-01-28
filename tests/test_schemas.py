from config import get_config, get_schema


def test_baseline_multiscale_realnvp_schema():
    config = get_config(dataset="mnist", model="realnvp", use_baseline=True)
    schema = get_schema(config)

    true_schema = [
        {
            "type": "dequantization"
        },
        {
            "type": "scalar-mult",
            "value": 0.0039062421875
        },
        {
            "type": "scalar-add",
            "value": 1e-06
        },
        {
            "type": "logit"
        },
        {
            "type": "acl",
            "mask_type": "checkerboard",
            "reverse_mask": False,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "acl",
            "mask_type": "checkerboard",
            "reverse_mask": True,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "acl",
            "mask_type": "checkerboard",
            "reverse_mask": False,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "squeeze",
            "factor": 2
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": True,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": True,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "split"
        },
        {
            "type": "acl",
            "mask_type": "checkerboard",
            "reverse_mask": False,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "acl",
            "mask_type": "checkerboard",
            "reverse_mask": True,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "acl",
            "mask_type": "checkerboard",
            "reverse_mask": False,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "acl",
            "mask_type": "checkerboard",
            "reverse_mask": True,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        }
    ]

    assert schema == true_schema


def test_lgf_multiscale_realnvp_schema():
    config = get_config(dataset="mnist", model="realnvp", use_baseline=False)
    schema = get_schema(config)

    true_schema = [
        {
            "type": "dequantization"
        },
        {
            "type": "scalar-mult",
            "value": 0.0039062421875
        },
        {
            "type": "scalar-add",
            "value": 1e-06
        },
        {
            "type": "logit"
        },
        {
            "type": "acl",
            "mask_type": "checkerboard",
            "reverse_mask": False,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        8,
                        8
                    ]
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "acl",
            "mask_type": "checkerboard",
            "reverse_mask": True,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        8,
                        8
                    ]
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "acl",
            "mask_type": "checkerboard",
            "reverse_mask": False,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        8,
                        8
                    ]
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "squeeze",
            "factor": 2
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": True,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        8,
                        8
                    ]
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        8,
                        8
                    ]
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": True,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        8,
                        8
                    ]
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "split"
        },
        {
            "type": "acl",
            "mask_type": "checkerboard",
            "reverse_mask": False,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        8,
                        8
                    ]
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "acl",
            "mask_type": "checkerboard",
            "reverse_mask": True,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        8,
                        8
                    ]
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "acl",
            "mask_type": "checkerboard",
            "reverse_mask": False,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        8,
                        8
                    ]
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "acl",
            "mask_type": "checkerboard",
            "reverse_mask": True,
            "num_u_channels": 0,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64,
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        8,
                        8
                    ]
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "resnet",
                    "hidden_channels": [
                        64,
                        64
                    ]
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        }
    ]

    assert schema == true_schema


def test_baseline_glow_schema():
    config = get_config(dataset="cifar10", model="glow", use_baseline=True)
    schema = get_schema(config)

    true_schema = [
        {
            "type": "dequantization"
        },
        {
            "type": "scalar-mult",
            "value": 0.00390625
        },
        {
            "type": "scalar-add",
            "value": -0.5
        },
        {
            "type": "squeeze",
            "factor": 2
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "split"
        },
        {
            "type": "squeeze",
            "factor": 2
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "split"
        },
        {
            "type": "squeeze",
            "factor": 2
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": True
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 512
                }
            },
            "num_u_channels": 0
        }
    ]

    assert schema == true_schema


def test_lgf_glow_schema():
    config = get_config(dataset="cifar10", model="glow", use_baseline=False)
    schema = get_schema(config)

    true_schema = [
        {
            "type": "dequantization"
        },
        {
            "type": "scalar-mult",
            "value": 0.00390625
        },
        {
            "type": "scalar-add",
            "value": -0.5
        },
        {
            "type": "squeeze",
            "factor": 2
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "split"
        },
        {
            "type": "squeeze",
            "factor": 2
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        },
        {
            "type": "cond-affine",
            "num_u_channels": 1,
            "st_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 64
                }
            },
            "p_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            },
            "q_coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 128
                }
            }
        },
        {
            "type": "batch-norm",
            "per_channel": True,
            "momentum": 0.1,
            "apply_affine": False
        },
        {
            "type": "invconv",
            "lu": True
        },
        {
            "type": "acl",
            "mask_type": "split-channel",
            "reverse_mask": False,
            "coupler": {
                "independent_nets": False,
                "shift_log_scale_net": {
                    "type": "glow-cnn",
                    "num_hidden_channels": 256
                }
            },
            "num_u_channels": 0
        }
    ]
 
    assert schema == true_schema
