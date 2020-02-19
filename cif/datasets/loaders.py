import sys

import torch

from .two_d import get_2d_datasets
from .image import get_image_datasets
from .tabular import get_tabular_datasets


def get_loader(dset, device, batch_size, drop_last):
    return torch.utils.data.DataLoader(
        dset.to(device),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=False
    )


def get_loaders(
        dataset,
        device,
        data_root,
        make_valid_loader,
        train_batch_size,
        valid_batch_size,
        test_batch_size
):
    print("Loading data...", end="", flush=True, file=sys.stderr)

    if dataset in ["cifar10", "svhn", "mnist", "fashion-mnist"]:
        train_dset, valid_dset, test_dset = get_image_datasets(dataset, data_root, make_valid_loader)

    elif dataset in ["miniboone", "hepmass", "power", "gas", "bsds300"]:
        # TODO: Make make_valid_loader apply here too
        train_dset, valid_dset, test_dset = get_tabular_datasets(dataset, data_root)

    else:
        # TODO: Make make_valid_loader apply here too
        train_dset, valid_dset, test_dset = get_2d_datasets(dataset)

    print("Done.", file=sys.stderr)

    train_loader = get_loader(train_dset, device, train_batch_size, drop_last=True)

    if make_valid_loader:
        valid_loader = get_loader(valid_dset, device, valid_batch_size, drop_last=False)
    else:
        valid_loader = None

    test_loader = get_loader(test_dset, device, test_batch_size, drop_last=False)

    return train_loader, valid_loader, test_loader
