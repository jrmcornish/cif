import torch

from .supervised_dataset import SupervisedDataset


def get_gaussian_dataset(role, size, dim, std):
    x = std * torch.randn(size, dim)
    y = torch.zeros(size).long()
    return SupervisedDataset(f"gaussian-dim{dim}-std{std}", role, x, y)


def get_well_conditioned_gaussian_datasets(dim, std, oos_std):
    train_dset = get_gaussian_dataset(role="train", size=50000, dim=dim, std=std)
    valid_dset = get_gaussian_dataset(role="valid", size=5000, dim=dim, std=std)
    test_dsets = [
        get_gaussian_dataset(role="test", size=10000, dim=dim, std=std),
        get_gaussian_dataset(role="test", size=10000, dim=dim, std=oos_std)
    ]

    return train_dset, valid_dset, test_dsets
