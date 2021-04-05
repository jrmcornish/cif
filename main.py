#!/usr/bin/env python3

import pprint
import contextlib
import argparse
import json
import time
import ast
from pathlib import Path

import sys
sys.setrecursionlimit(3000) # Necessary for Glow

from config import get_datasets, get_models, get_config, get_schema, expand_grid


parser = argparse.ArgumentParser()

parser.add_argument("--model", choices=get_models())
parser.add_argument("--dataset", choices=get_datasets())
parser.add_argument("--baseline", action="store_true", help="Run baseline flow instead of CIF")
parser.add_argument("--num-seeds", type=int, default=1, help="Number of random seeds to use.")
parser.add_argument("--checkpoints", choices=["best-valid", "latest", "both", "none"], default="both", help="Type of checkpoints to save (default: %(default)s)")
parser.add_argument("--nosave", action="store_true", help="Don't save anything to disk")
parser.add_argument("--data-root", default="data/", help="Location of training data (default: %(default)s)")
parser.add_argument("--logdir-root", default="runs/", help="Location of log files (default: %(default)s)")

parser.add_argument("--config", default=[], action="append", help="Override config entries at runtime. Specify as `key=value' (e.g. `--config max_epochs=50'). Any config value can be overridden, but some (e.g. `model') may lead to unforeseen consequences, so this should be used with care.")

parser.add_argument("--load", help="Directory of a run to load. If this flag is specified, the following flags will be ignored silently: --model, --dataset, --baseline, --num-seeds, --checkpoints, --nosave, --data-root, --logdir-root. Their values *can* be overridden via --config, but this may lead to unforeseen consequences in some cases.")

parser.add_argument("--print-config", action="store_true", help="Print the full config and exit")
parser.add_argument("--print-schema", action="store_true", help="Print the model schema and exit")
parser.add_argument("--print-model", action="store_true", help="Print the model and exit")
parser.add_argument("--print-num-params", action="store_true", help="Print the number of parameters and exit")
parser.add_argument("--test", action="store_true", help="Test model and exit instead of training.")

args = parser.parse_args()


def parse_config_arg(key_value):
    assert "=" in key_value, "Must specify config items with format `key=value'"

    k, v = key_value.split("=", maxsplit=1)

    assert k, "Config item can't have empty key"
    assert v, "Config item can't have empty value"

    try:
        v = ast.literal_eval(v)
    except ValueError:
        v = str(v)

    return k, v


if args.load is None:
    assert args.model is not None, "Must specify --model"
    assert args.dataset is not None, "Must specify --dataset"

    config = get_config(
        model=args.model,
        dataset=args.dataset,
        use_baseline=args.baseline
    )

    config_to_merge = {
        "model": args.model,
        "dataset": args.dataset,
        "should_checkpoint_best_valid": args.checkpoints in ["best-valid", "both"],
        "should_checkpoint_latest": args.checkpoints in ["latest", "both"],
        "write_to_disk": not args.nosave,
        "data_root": args.data_root,
        "logdir_root": args.logdir_root
    }

    for key in config_to_merge:
        assert key not in config, f"Should not specify key `{key}' in config"

    config = {**config, **config_to_merge}

else:
    with open(Path(args.load) / "config.json", "r") as f:
        config = json.load(f)

    args.num_seeds = 1

config = {**config, **dict(parse_config_arg(kv) for kv in args.config)}

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
                    print_test_metrics(config=c, load_dir=args.load)
                else:
                    train(config=c, load_dir=args.load)
