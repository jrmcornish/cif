import json
import random
from pathlib import Path
import sys
import subprocess

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from .trainer import Trainer
from .datasets import get_loaders
from .visualizer import DummyDensityVisualizer, ImageDensityVisualizer, TwoDimensionalDensityVisualizer
from .models import get_density
from .writer import Writer, DummyWriter
from .metrics import metrics

from config import get_schema


def train(config, resume_dir):
    density, trainer, writer = setup_experiment(config=config, resume_dir=resume_dir)

    writer.write_json("config", config)

    writer.write_json("model", {
        "num_params": num_params(density),
        "schema": get_schema(config)
    })

    writer.write_textfile("git-head", subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii"))
    writer.write_textfile("git-diff", subprocess.check_output(["git", "diff"]).decode("ascii"))

    print("\nConfig:")
    print(json.dumps(config, indent=4))
    print(f"\nNumber of parameters: {num_params(density):,}\n")

    trainer.train()


def print_test_metrics(config, resume_dir):
    _, trainer, _ = setup_experiment(
        config={**config, "write_to_disk": False},
        resume_dir=resume_dir
    )

    with torch.no_grad():
        test_metrics = trainer.test()

    test_metrics = {k: v.item() for k, v in test_metrics.items()}

    print(json.dumps(test_metrics, indent=4))


def print_model(config):
    density, _, _, _ = setup_density_and_loaders(
        config={**config, "write_to_disk": False},
        device=torch.device("cpu")
    )
    print(density)


def print_num_params(config):
    density, _, _, _ = setup_density_and_loaders(
        config={**config, "write_to_disk": False},
        device=torch.device("cpu")
    )
    print(f"Number of parameters: {num_params(density):,}")


def setup_density_and_loaders(config, device):
    train_loader, valid_loader, test_loader = get_loaders(
        dataset=config["dataset"],
        device=device,
        data_root=config["data_root"],
        make_valid_loader=config["early_stopping"],
        train_batch_size=config["train_batch_size"],
        valid_batch_size=config["valid_batch_size"],
        test_batch_size=config["test_batch_size"]
    )

    density = get_density(
        schema=get_schema(config=config),
        x_train=train_loader.dataset.x
    )

    # TODO: Could do lazily inside Trainer
    density.to(device)

    return density, train_loader, valid_loader, test_loader


def load_run(run_dir, device):
    run_dir = Path(run_dir)

    with open(run_dir / "config.json", "r") as f:
        config = json.load(f)

    density, train_loader, valid_loader, test_loader = setup_density_and_loaders(
        config=config,
        device=device
    )

    try:
        checkpoint = torch.load(run_dir / "checkpoints" / "best_valid.pt", map_location=device)
    except FileNotFoundError:
        checkpoint = torch.load(run_dir / "checkpoints" / "latest.pt", map_location=device)

    print("Loaded checkpoint after epoch", checkpoint["epoch"])

    density.load_state_dict(checkpoint["module_state_dict"])

    return density, train_loader, valid_loader, test_loader, config, checkpoint


def setup_experiment(config, resume_dir):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"]+1)
    random.seed(config["seed"]+2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    density, train_loader, valid_loader, test_loader = setup_density_and_loaders(
        config=config,
        device=device
    )

    if config["opt"] == "sgd":
        opt_class = optim.SGD
    elif config["opt"] == "adam":
        opt_class = optim.Adam
    elif config["opt"] == "adamax":
        opt_class = optim.Adamax
    else:
        assert False, f"Invalid optimiser type {config['opt']}"

    opt = opt_class(
        density.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    if config["lr_schedule"] == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=opt,
            T_max=config["max_epochs"]*len(train_loader),
            eta_min=0.
        )
    elif config["lr_schedule"] == "none":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=opt,
            lr_lambda=lambda epoch: 1.
        )
    else:
        assert False, f"Invalid learning rate schedule `{config['lr_schedule']}'"

    if config["write_to_disk"]:
        if resume_dir is None:
            logdir = config["logdir_root"]
            make_subdir = True
        else:
            logdir = resume_dir
            make_subdir = False

        writer = Writer(logdir=logdir, make_subdir=make_subdir, tag_group=config["dataset"])
    else:
        writer = DummyWriter(logdir=resume_dir)

    if config["dataset"] in ["cifar10", "svhn", "fashion-mnist", "mnist"]:
        visualizer = ImageDensityVisualizer(writer=writer)
    elif train_loader.dataset.x.shape[1:] == (2,):
        visualizer = TwoDimensionalDensityVisualizer(
            writer=writer,
            x_train=train_loader.dataset.x,
            num_elbo_samples=config["num_test_elbo_samples"],
            device=device
        )
    else:
        visualizer = DummyDensityVisualizer(writer=writer)

    if config["schema_type"] == "ffjord":
        def train_metrics(density, x):
            train_info = density("elbo", x)
            loss = -train_info["elbo"].mean()

            nfes = torch.tensor(0.)
            while "prior-dict" in train_info:
                nfes += train_info["bijection-info"].get("nfes", torch.tensor(0.))
                train_info = train_info["prior-dict"]

            return {"loss": loss, "nfes": nfes}

    else:
        def train_metrics(density, x):
            return {"loss": -density("elbo", x)["elbo"].mean()}

    def valid_loss(density, x):
        return -metrics(density, x, config["num_valid_elbo_samples"])["log-prob"]

    def test_metrics(density, x):
        return metrics(density, x, config["num_test_elbo_samples"])

    trainer = Trainer(
        module=density,
        train_metrics=train_metrics,
        valid_loss=valid_loss,
        test_metrics=test_metrics,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        opt=opt,
        lr_scheduler=lr_scheduler,
        max_epochs=config["max_epochs"],
        max_grad_norm=config["max_grad_norm"],
        early_stopping=config["early_stopping"],
        max_bad_valid_epochs=config["max_bad_valid_epochs"],
        visualizer=visualizer,
        writer=writer,
        epochs_per_test=config["epochs_per_test"],
        should_checkpoint_latest=config["should_checkpoint_latest"],
        should_checkpoint_best_valid=config["should_checkpoint_best_valid"],
        device=device
    )

    return density, trainer, writer


def num_params(module):
    return sum(p.view(-1).shape[0] for p in module.parameters())
