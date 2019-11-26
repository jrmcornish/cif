import contextlib
import json
import random

import numpy as np

import torch
import torch.optim as optim

from .trainer import Trainer
from .datasets import get_loaders
from .visualizer import DummyDensityVisualizer, ImageDensityVisualizer, TwoDimensionalDensityVisualizer
from .models import get_schema, get_density
from .writer import Writer, DummyWriter


def train(config):
    density, trainer, writer = setup_experiment(config)

    writer.write_json("config", config)
    writer.write_json("model", {
        "num_params": num_params(density),
        "schema": get_schema(config)
    })

    print("\nConfig:")
    print(json.dumps(config, indent=4))
    print(f"\nNumber of parameters: {num_params(density):,}\n")

    with contextlib.suppress(KeyboardInterrupt):
        trainer.train()


def print_model(config):
    density, _, _, _ = setup_density_and_loaders(config, torch.device("cpu"))
    print(density)
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
    ).to(device)

    return density, train_loader, valid_loader, test_loader


def setup_experiment(config):
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
        writer = Writer(logdir_root=config["logdir_root"], tag_group=config["dataset"])
    else:
        writer = DummyWriter()

    if config["dataset"] in ["cifar10", "svhn", "fashion-mnist", "mnist"]:
        visualizer = ImageDensityVisualizer(writer=writer)
    elif train_loader.dataset.x.shape[1:] == (2,):
        visualizer = TwoDimensionalDensityVisualizer(
            writer=writer,
            train_loader=train_loader,
            num_elbo_samples=config["num_test_elbo_samples"],
            device=device
        )
    else:
        visualizer = DummyDensityVisualizer(writer=writer)

    train_loss = lambda density, x: -density.metrics(x, config["num_train_elbo_samples"])["elbo"]
    valid_loss = lambda density, x: -density.metrics(x, config["num_valid_elbo_samples"])["log-prob"]
    test_metrics = lambda density, x: density.metrics(x, config["num_test_elbo_samples"])

    trainer = Trainer(
        module=density,
        train_loss=train_loss,
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
