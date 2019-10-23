import torch

from .two_d import get_2d_datasets
from .image import get_image_datasets
from .uci import get_UCI_datasets


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
        train_batch_size,
        valid_batch_size,
        test_batch_size
):
    print("Loading data...", end="", flush=True)

    if dataset in ["cifar10", "svhn", "mnist", "fashion-mnist"]:
        train_dset, valid_dset, test_dset = get_image_datasets(dataset, data_root)
    elif dataset in ["miniboone", "hepmass", "power", "gas"]:
        train_dset, valid_dset, test_dset = get_UCI_datasets(dataset, data_root)
    else:
        train_dset, valid_dset, test_dset = get_2d_datasets(dataset)

    print("Done.")

    return (
        get_loader(train_dset, device, train_batch_size, drop_last=True),
        get_loader(valid_dset, device, valid_batch_size, drop_last=False),
        get_loader(test_dset, device, test_batch_size, drop_last=False)
    )
