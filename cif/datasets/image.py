import os

import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.datasets
import torchvision.transforms as transforms

import imageio

from .supervised_dataset import SupervisedDataset


class NotMNIST(Dataset):
    # `root` should contain the notMNIST_small/ directory, which is available at
    #   http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz
    #   MD5sum: c9890a473a9769fda4bdf314aaf500dd
    # See https://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html for full information
    #
    # `train` and `download` arguments are ignored. I add them for compatibility with e.g.
    # MNIST and FashionMNIST. In the future it would be good to make a train/test split
    def __init__(self, root, train=False, download=False):
        assert not train, "Only test set available for NotMNIST"
        self.data, self.targets = self._load_tensors(root)

    def _load_tensors(self, root):
        data_path = os.path.join(root, "data.pt")
        targets_path = os.path.join(root, "targets.pt")

        try:
            with open(data_path, "rb") as f:
                data = torch.load(f)

            with open(targets_path, "rb") as f:
                targets = torch.load(f)

        except FileNotFoundError:
            data, targets = self._load_raw_images(os.path.join(root, "notMNIST_small"))

            torch.save(data, data_path)
            torch.save(targets, targets_path)

        return data, targets

    def _load_raw_images(self, root):
        data = []
        targets = []
        for letter in os.listdir(root):
            folder_path = os.path.join(root, letter)
            for basename in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, basename)
                    data.append(np.array(imageio.imread(img_path)))

                    # Convert letter label to numerical label in 0-9
                    targets.append("ABCDEFGHIJ".index(letter))
                except ValueError:
                    # Two images in the dataset, namely:
                    #   A/RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png
                    #   F/Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png
                    print("File {}/{} is broken".format(letter, basename), flush=True)

        data = torch.tensor(data)
        targets = torch.tensor(targets)

        return data, targets


# Returns tuple of form `(images, labels)`. Both are uint8 tensors.
# `images` has shape `(nimages, nchannels, nrows, ncols)`, and has
# entries in {0, ..., 255}
def get_raw_image_tensors(dataset_name, train, data_root):
    data_dir = os.path.join(data_root, dataset_name)

    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True)
        images = torch.tensor(dataset.data).permute((0, 3, 1, 2))
        labels = torch.tensor(dataset.targets)

    elif dataset_name == "svhn":
        dataset = torchvision.datasets.SVHN(root=data_dir, split="train" if train else "test", download=True)
        images = torch.tensor(dataset.data)
        labels = torch.tensor(dataset.labels)

    elif dataset_name in ["mnist", "fashion-mnist"]:
        dataset_class = {
            "mnist": torchvision.datasets.MNIST,
            "fashion-mnist": torchvision.datasets.FashionMNIST
        }[dataset_name]
        dataset = dataset_class(root=data_dir, train=train, download=True)
        images = dataset.data.unsqueeze(1)
        labels = dataset.targets

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return images.to(torch.uint8), labels.to(torch.uint8)


def image_tensors_to_supervised_dataset(dataset_name, dataset_role, images, labels):
    images = images.to(dtype=torch.get_default_dtype())
    labels = labels.long()
    return SupervisedDataset(dataset_name, dataset_role, images, labels)


def get_train_valid_image_datasets(dataset_name, data_root, valid_fraction, add_train_hflips):
    images, labels = get_raw_image_tensors(dataset_name, train=True, data_root=data_root)

    perm = torch.randperm(images.shape[0])
    shuffled_images = images[perm]
    shuffled_labels = labels[perm]

    valid_size = int(valid_fraction * images.shape[0])
    valid_images = shuffled_images[:valid_size]
    valid_labels = shuffled_labels[:valid_size]
    train_images = shuffled_images[valid_size:]
    train_labels = shuffled_labels[valid_size:]

    if add_train_hflips:
        train_images = torch.cat((train_images, train_images.flip([3])))
        train_labels = torch.cat((train_labels, train_labels))

    train_dset = image_tensors_to_supervised_dataset(dataset_name, "train", train_images, train_labels)
    valid_dset = image_tensors_to_supervised_dataset(dataset_name, "valid", valid_images, valid_labels)

    return train_dset, valid_dset


def get_test_image_dataset(dataset_name, data_root):
    images, labels = get_raw_image_tensors(dataset_name, train=False, data_root=data_root)
    return image_tensors_to_supervised_dataset(dataset_name, "test", images, labels)


def get_image_datasets(dataset_name, data_root, make_valid_dset):
    # Currently hardcoded; could make configurable
    valid_fraction = 0.1 if make_valid_dset else 0
    add_train_hflips = False

    train_dset, valid_dset = get_train_valid_image_datasets(dataset_name, data_root, valid_fraction, add_train_hflips)
    test_dset = get_test_image_dataset(dataset_name, data_root)
    return train_dset, valid_dset, test_dset
