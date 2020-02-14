#!/usr/bin/env python3

import glob
import os
import sys
import shutil

from config_grouping_functions import get_config, group_runs

root = sys.argv[1]


for path in glob.glob(f"{root}/*"):
    dataset = get_config(path)["dataset"]
    dataset_dir = f"{os.path.dirname(path)}/{dataset}"
    os.makedirs(dataset_dir, exist_ok=True)
    shutil.move(path, dataset_dir)


for path in glob.glob(f"{root}/*/*"):
    model = get_config(path)["model"]
    model_dir = f"{os.path.dirname(path)}/{model}"
    os.makedirs(model_dir, exist_ok=True)
    shutil.move(path, model_dir)


for model_dir in glob.glob(f"{root}/*/*"):
    groups = group_runs(model_dir)

    for i, group in enumerate(groups):
        group_dir = f"{model_dir}/{i}"
        os.makedirs(group_dir)
        shutil.copyfile(f"{group[0]}/config.json", f"{group_dir}/config.json")

        for run in group:
            shutil.move(run, group_dir)


