#!/usr/bin/env python3

import pprint
import contextlib
import argparse
import json
import time
import ast
from pathlib import Path

import sys
sys.setrecursionlimit(3000)

from config import get_datasets, get_models, get_config, get_schema, expand_grid


parser = argparse.ArgumentParser()

parser.add_argument("--resume", help="Directory of run to resume. Ignores other command-line settings for run.")

parser.add_argument("--model", choices=get_models())
parser.add_argument("--dataset", choices=get_datasets())
parser.add_argument("--baseline", action="store_true", help="Run baseline flow instead of CIF")
parser.add_argument("--num-seeds", type=int, default=1, help="Number of random seeds to use.")
parser.add_argument("--checkpoints", choices=["best-valid", "latest", "both", "none"], default="both", help="Type of checkpoints to save (default: %(default)s)")
parser.add_argument("--nosave", action="store_true", help="Don't save anything to disk")
parser.add_argument("--data-root", default="data/", help="Location of training data (default: %(default)s)")
parser.add_argument("--logdir-root", default="runs/", help="Location of log files (default: %(default)s)")
parser.add_argument("--config", default=[], action="append", help="Override config entries. Specify as `key=value`.")

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

parser.add_argument("--print-config", action="store_true", help="Print the full config and exit")
parser.add_argument("--print-schema", action="store_true", help="Print the model schema and exit")
parser.add_argument("--print-model", action="store_true", help="Print the model and exit")
parser.add_argument("--print-num-params", action="store_true", help="Print the number of parameters and exit")
parser.add_argument("--test", action="store_true", help="Test model and exit instead of training.")

args = parser.parse_args()


if args.resume is None:
    assert args.model is not None and args.dataset is not None

    config = get_config(
        model=args.model,
        dataset=args.dataset,
        use_baseline=args.baseline
    )

    assert "model" not in config, "Should not specify model in config"
    assert "datatset" not in config, "Should not specify dataset in config"
    config = {
        "model": args.model,
        "dataset": args.dataset,
        **config
    }

    config = {**config, **dict(parse_config_arg(kv) for kv in args.config)}

    config = {
        **config,
        "should_checkpoint_best_valid": args.checkpoints in ["best-valid", "both"],
        "should_checkpoint_latest": args.checkpoints in ["latest", "both"],
        "write_to_disk": not args.nosave,
        "data_root": args.data_root,
        "logdir_root": args.logdir_root
    }

else:
    with open(Path(args.resume) / "config.json", "r") as f:
        config = json.load(f)

    args.num_seeds = 1


should_train = True

if args.print_config:
    pprint.sorted = lambda x, key=None: x
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    should_train = False

grid = expand_grid(config)

if args.print_model:
    from cif.experiment import print_model
    for c in grid:
        print_model(c)
    should_train = False

if args.print_num_params:
    from cif.experiment import print_num_params
    for c in grid:
        print_num_params(c)
    should_train = False

if args.print_schema:
    if len(grid) == 1:
        print(json.dumps(get_schema(grid[0]), indent=4))
    else:
        for i, c in enumerate(grid):
            if i > 0:
                print()
            sep_width = 10
            print(("=" * sep_width) + f" Schema {i} " + ("=" * sep_width) + "\n")
            print(json.dumps(get_schema(c), indent=4))
    should_train = False

if should_train:
    from cif.experiment import train, print_test_metrics
    with contextlib.suppress(KeyboardInterrupt):
        for c in grid:
            for _ in range(args.num_seeds):
                # TODO: A bit misleading to log changed seed if we resume like this
                c = {**c, "seed": int(time.time() * 1e6) % 2**32}

                if args.test:
                    print_test_metrics(config=c, resume_dir=args.resume)
                else:
                    train(config=c, resume_dir=args.resume)
