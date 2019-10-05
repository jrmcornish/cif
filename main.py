#!/usr/bin/env python3

import argparse

from lgf.experiment import train, print_model

from config import two_uniforms, two_d, uci, images


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="Random seed to use")
parser.add_argument("--print", action="store_true", help="Print the model structure and exit")
parser.add_argument("--baseline", action="store_true", help="Run the baseline instead of the LGF")
parser.add_argument("--nosave", action="store_true", help="Don't save anything to disk")
parser.add_argument("--nochkpt", action="store_true", help="Disable checkpointing")
parser.add_argument("--data-root", default="data", help="Location of training data")
parser.add_argument("--logdir-root", default="runs", help="Location of log files")
parser.add_argument("--dataset", choices=[
    "2uniforms",
    "8gaussians", "checkerboard", "2spirals",
    "power", "gas", "hepmass", "miniboone",
    "mnist", "fashion-mnist", "cifar10", "svhn"
], required=True)

args = parser.parse_args()

if args.dataset == "2uniforms":
    config = two_uniforms(args.baseline)

elif args.dataset in ["8gaussians", "checkerboard", "2spirals"]:
    config = two_d(args.dataset, args.baseline)

elif args.dataset in ["power", "gas", "hepmass", "miniboone"]:
    config = uci(args.dataset, args.baseline)

elif args.dataset in ["mnist", "fashion-mnist", "cifar10", "svhn"]:
    config = images(args.dataset, args.baseline)

config = {
    "seed": args.seed,
    "should_save_checkpoints": not args.nochkpt,
    "write_to_disk": not args.nosave,
    "data_root": args.data_root,
    "logdir_root": args.logdir_root,
    **config
}

if args.print:
    print_model(config)

else:
    train(config)
