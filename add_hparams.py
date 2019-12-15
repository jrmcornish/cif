#!/usr/bin/env python3

import sys
import json
import glob

import torch

from tensorboardX import SummaryWriter

from lgf.experiment import load_run

def paths(root):
    return glob.glob(f"{root}/*")


def get_config(path):
    with open(f"{path}/config.json", "r") as f:
        return json.load(f)


def get_configs(root):
    return [get_config(p) for p in paths(root)]


def all_keys(configs):
    result = []
    for c in configs:
        result += list(c)
    return list(set(result))


def should_ignore_key(key):
    return key in ["seed", "schema_type"]


def differing_keys(root):
    configs = get_configs(root)

    result = []
    for k in all_keys(configs):
        if not should_ignore_key(k) and key_value_differs(k, configs):
            result.append(k)

    return result


def key_value_differs(key, configs):
    if key not in configs[0]:
        return True

    val = configs[0][key]

    for c in configs:
        if key not in c or c[key] != val:
            return True

    return False


def get_config_values(keys, path):
    config = get_config(path)

    result = {}
    for k in keys:
        val = config.get(k)
        result[k] = simplify_value(val)

    return result


def simplify_value(val):
    if isinstance(val, list):
        items = set(val)
        if len(items) == 1:
            item, = items
            return f"[{item}]*{len(val)}"
        else:
            return f"[{','.join(items)}]"
    elif val is None:
        return "None"
    else:
        return val


root = sys.argv[1]

keys = differing_keys(root)

for path in paths(root):
    vals = get_config_values(keys, path)

    density, x_train, x_valid, x_test, config, checkpoint = load_run(
        run_dir=path,
        device=torch.device("cpu")
    )

    density.eval()
    with torch.no_grad():
        all_metrics = density.metrics(x_test, num_elbo_samples=100)
    
    metrics = {
        "log-prob": all_metrics["log-prob"].mean().item(),
        "elbo-gap": all_metrics["elbo-gap"].mean().item(),
        "epoch": checkpoint["epoch"]
    }

    writer = SummaryWriter(logdir=path)
    writer.add_hparams(hparam_dict=vals, metric_dict=metrics)
