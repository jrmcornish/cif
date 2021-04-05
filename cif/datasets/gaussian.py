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


def get_linear_gaussian_dataset(role, size):
    # A = torch.randn(20, 10)
    # b = torch.randn(20)
    # sigma = 1e-2

    A = torch.tensor([[-4.], [1.]])
    b = torch.tensor([1., -3.])
    sigma = 0.1

    z = torch.randn(size, A.shape[1], 1)
    Az = torch.matmul(A, z).view(size, A.shape[0])
    x = Az + b + sigma*torch.randn_like(Az)

    return SupervisedDataset(name="linear-gaussian", role=role, x=x)


def get_linear_gaussian_datasets():
    train_dset = get_linear_gaussian_dataset(role="train", size=100_000)
    valid_dset = get_linear_gaussian_dataset(role="valid", size=10_000)
    test_dset = get_linear_gaussian_dataset(role="test", size=10_000)
    return train_dset, valid_dset, test_dset
