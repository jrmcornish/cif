#!/usr/bin/env python3

import os
import sys
import json
import glob

import torch

from tensorboardX import SummaryWriter

import tqdm

from lgf.experiment import load_run, num_params

root = sys.argv[1]


for path in tqdm.tqdm(glob.glob(f"{root}/*/*")):
    if not os.path.isdir(path):
        continue

    metrics_path = os.path.join(path, "valid_metrics.json")
    if os.path.exists(metrics_path):
        print("Skipping {}".format(path))
        continue

    try:
        density, train_loader, valid_loader, test_loader, config, checkpoint = load_run(
            run_dir=path,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    except KeyError as e:
        import ipdb; ipdb.set_trace()
        print("Error {0} for path {1}".format(e, path))

    num_elbo_samples = 1 if config["num_u_channels"] == 0 else config["num_valid_elbo_samples"]

    density.eval()
    sum_log_prob = 0.
    sum_elbo_gap = 0.

    with torch.no_grad():
        for (x, _) in tqdm.tqdm(valid_loader):
            try:
                all_metrics = density.metrics(x, num_elbo_samples=num_elbo_samples)
                sum_log_prob += all_metrics["log-prob"].sum().item()
                sum_elbo_gap += all_metrics["elbo-gap"].sum().item()
            except Exception as e:
                import ipdb; ipdb.set_trace()
                print("Error {0} for path {1}".format(e, path))

    points_in_test = valid_loader.dataset.x.shape[0]
    metrics = {
        "log-prob": sum_log_prob / points_in_test,
        "elbo-gap": sum_elbo_gap / points_in_test,
        "epoch": checkpoint["epoch"],
        "num-params": num_params(density),
        "test-elbo-samples": num_elbo_samples
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

