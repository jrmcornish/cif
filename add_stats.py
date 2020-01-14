#!/usr/bin/env python3

import os
import sys
import glob
import json

import numpy as np


root = sys.argv[1]


for group in glob.glob(f"{root}/*/*/*"):
    if os.path.exists(f"{group}/stats.json"):
        os.unlink(f"{group}/stats.json")


for group in glob.glob(f"{root}/*/*/*"):
    if not os.path.isdir(group):
        continue

    log_probs = []
    

    for run in glob.glob(f"{group}/*"):
        try:
            with open(f"{run}/metrics.json", "r") as f:
                metrics = json.load(f)
            log_probs.append(metrics["log-prob"])
        except (FileNotFoundError, NotADirectoryError):
            print(f"Skipping {group} because no metrics.json in {run}")
            break

    else:
        stats = { "mean": np.mean(log_probs) }

        if len(log_probs) >= 3:
            stats["stderr"] = np.std(log_probs) / np.sqrt(len(log_probs))
        else:
            print(f"No stderr for {group} because less than 3 samples")

        with open(f"{group}/stats.json", "w") as f:
            json.dump(stats, f)

