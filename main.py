#!/usr/bin/env python3

import argparse
import json
import time
import ast

import sys
sys.setrecursionlimit(3000)


parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=[
    "maf", "flat-realnvp", "sos", "bnaf", "nsf",
    "multiscale-realnvp", "glow", "ffjord"
], required=True)
parser.add_argument("--dataset", choices=[
    "2uniforms", "2lines", "8gaussians", "checkerboard", "2spirals", "rings",
    "power", "gas", "hepmass", "miniboone",
    "mnist", "fashion-mnist", "cifar10", "svhn"
], required=True)
parser.add_argument("--baseline", action="store_true", help="Run baseline flow instead of LGF")
parser.add_argument("--pure-cond-affine", action="store_true", help="Only use the conditional affine layers in the model. (Requires not using the baseline.)")
parser.add_argument("--seed", type=int, help="Random seed to use. Defaults to using current time.")
parser.add_argument("--print-model", action="store_true", help="Print the model and exit")
parser.add_argument("--print-schema", action="store_true", help="Print the model schema and exit")
parser.add_argument("--print-config", action="store_true", help="Print the full config and exit")
parser.add_argument("--nochkpt", action="store_true", help="Disable checkpointing")
parser.add_argument("--checkpoints", choices=["best-valid", "latest", "both", "none"], default="both", help="Type of checkpoints to save (default: %(default)s)")
parser.add_argument("--nosave", action="store_true", help="Don't save anything to disk")
parser.add_argument("--data-root", default="data/", help="Location of training data (default: %(default)s)")
parser.add_argument("--logdir-root", default="runs/", help="Location of log files (default: %(default)s)")
parser.add_argument("--config", default=[], action="append", help="Override config entries. Specify as `key=value`.")

args = parser.parse_args()

from config import get_config

config = get_config(
    model=args.model,
    dataset=args.dataset,
    use_baseline=args.baseline
)

if args.baseline:
    assert not args.pure_cond_affine
else:
    assert config["num_u_channels"] > 0

def parse_config_arg(key_value):
    assert "=" in key_value, "Must specify config items with format `key=value`"

    k, v = key_value.split("=", maxsplit=1)

    assert k, "Config item can't have empty key"
    assert v, "Config item can't have empty value"

    try:
        v = ast.literal_eval(v)
    except ValueError:
        v = str(v)

    return k, v

args_config = dict(parse_config_arg(kv) for kv in args.config)

if args.seed is None:
    seed = int(time.time() * 1e6) % 2**32
else:
    seed = args.seed

config = {
    **config,
    **args_config,
    "pure_cond_affine": args.pure_cond_affine,
    "seed": seed,
    "should_checkpoint_best_valid": args.checkpoints in ["best-valid", "both"],
    "should_checkpoint_latest": args.checkpoints in ["latest", "both"],
    "write_to_disk": not args.nosave,
    "data_root": args.data_root,
    "logdir_root": args.logdir_root
}

should_train = True

if args.print_config:
    print(json.dumps(config, indent=4))
    should_train = False

if args.print_model:
    from lgf.experiment import print_model
    print_model({**config, "write_to_disk": False})
    should_train = False

if args.print_schema:
    from lgf.models.schemas import get_schema
    print(json.dumps(get_schema(config), indent=4))
    should_train = False

if should_train:
    from lgf.experiment import train
    train(config)
