import torch


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, name, role, x, y=None):
        if y is None:
            y = torch.zeros(x.shape[0]).long()

        assert x.shape[0] == y.shape[0]
        assert role in ["train", "valid", "test"]

        self.name = name
        self.role = role

        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def to(self, device):
        return SupervisedDataset(
            self.name,
            self.role,
            self.x.to(device),
            self.y.to(device)
        )
