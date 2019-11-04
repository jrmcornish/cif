#!/usr/bin/env python3

import argparse
import json
import time

import sys
sys.setrecursionlimit(3000)

from lgf.experiment import train, print_density, print_schema

from config import get_config


parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["maf", "flat-realnvp", "multiscale-realnvp", "glow"])
parser.add_argument("--dataset", choices=[
    "2uniforms", "8gaussians", "checkerboard", "2spirals",
    "power", "gas", "hepmass", "miniboone",
    "mnist", "fashion-mnist", "cifar10", "svhn"
])
parser.add_argument("--print-density", action="store_true", help="Print the Pytorch Density and exit")
parser.add_argument("--print-schema", action="store_true", help="Print the model schema and exit")
parser.add_argument("--print-config", action="store_true", help="Print the full config and exit")
parser.add_argument("--baseline", action="store_true", help="Run baseline flow instead of LGF")
parser.add_argument("--nochkpt", action="store_true", help="Disable checkpointing")
parser.add_argument("--checkpoints", choices=["best-valid", "latest", "both", "none"], default="both", help="Type of checkpoints to save")
parser.add_argument("--nosave", action="store_true", help="Don't save anything to disk")
parser.add_argument("--data-root", default="data/", help="Location of training data (default: %(default)s)")
parser.add_argument("--logdir-root", default="runs/", help="Location of log files (default: %(default)s)")
parser.add_argument("--config", default="{}", help="Override config entries. Specify as JSON.")

args = parser.parse_args()

config = get_config(
    model=args.model,
    dataset=args.dataset,
    use_baseline=args.baseline
)

config = {
    **config,
    **json.loads(args.config),
    "dataset": args.dataset,
    "model": args.model,
    "seed": int(time.time() * 1e6) % 2**32,
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

if args.print_density:
    print_density({**config, "write_to_disk": False})
    should_train = False

if args.print_schema:
    print_schema(config)
    should_train = False

if should_train:
    train(config)
