#!/usr/bin/env python3

import os
import sys
import json
import glob

import torch

from tensorboardX import SummaryWriter

import tqdm

from lgf.experiment import load_run, num_params
from lgf.metrics import metrics
from config_grouping_functions import differing_keys, get_config_values

root = sys.argv[1]

keys = differing_keys(root)


for path in tqdm.tqdm(glob.glob(f"{root}/*")):
    metrics_path = os.path.join(path, "metrics.json")
    if os.path.exists(metrics_path):
        print("Skipping {}".format(path))
        continue

    vals = get_config_values(keys, path)

    try:
        density, train_loader, valid_loader, test_loader, config, checkpoint = load_run(
            run_dir=path,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            data_parallel=torch.cuda.device_count() > 1
        )
    except KeyError as e:
        import ipdb; ipdb.set_trace()
        print("Error {0} for path {1}".format(e, path))

    num_elbo_samples = 1 if config["num_u_channels"] == 0 else 100

    density.eval()
    sum_log_prob = 0.
    sum_elbo_gap = 0.

    with torch.no_grad():
        for (x, _) in tqdm.tqdm(test_loader):
            try:
                all_metrics = metrics(density, x, num_elbo_samples)
                sum_log_prob += all_metrics["log-prob"].sum().item()
                sum_elbo_gap += all_metrics["elbo-gap"].sum().item()
            except Exception as e:
                import ipdb; ipdb.set_trace()
                print("Error {0} for path {1}".format(e, path))

    points_in_test = test_loader.dataset.x.shape[0]
    metrics = {
        "log-prob": sum_log_prob / points_in_test,
        "elbo-gap": sum_elbo_gap / points_in_test,
        "epoch": checkpoint["epoch"],
        "num-params": num_params(density),
        "test-elbo-samples": num_elbo_samples
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    metrics = {f"hparams/{k}": v for k, v in metrics.items()}

    writer = SummaryWriter(logdir=path)
    writer.add_hparams(hparam_dict=vals, metric_dict=metrics)
