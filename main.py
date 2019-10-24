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
parser.add_argument("--nosave", action="store_true", help="Don't save anything to disk (including checkpoints)")
parser.add_argument("--data-root", default="data/", help="Location of training data (default: %(default)s)")
parser.add_argument("--logdir-root", default="runs/", help="Location of log files (default: %(default)s)")
parser.add_argument("--config", default="{}", help="Override config entries (JSON)")

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
    "should_checkpoint": not args.nochkpt,
    "write_to_disk": (
        not args.nosave
        and not args.print_density
        and not args.print_schema
        and not args.print_config
    ),
    "data_root": args.data_root,
    "logdir_root": args.logdir_root
}

should_train = True

if args.print_config:
    print(json.dumps(config, indent=4))
    should_train = False

if args.print_density:
    print_density(config)
    should_train = False

if args.print_schema:
    print_schema(config)
    should_train = False

if should_train:
    train(config)
