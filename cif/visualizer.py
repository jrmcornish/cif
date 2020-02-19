from collections import defaultdict

import numpy as np

import torch
import torch.utils.data
import torchvision.utils

import matplotlib.pyplot as plt

import tqdm

from .metrics import metrics


# TODO: Make return a matplotlib figure instead. Writing can be done outside.
class DensityVisualizer:
    def __init__(self, writer):
        self._writer = writer

    def visualize(self, density, epoch):
        raise NotImplementedError


class DummyDensityVisualizer(DensityVisualizer):
    def visualize(self, density, epoch):
        return


class ImageDensityVisualizer(DensityVisualizer):
    def visualize(self, density, epoch):
        imgs = density.fixed_sample()

        num_images = imgs.shape[0]
        num_rows = int(np.ceil(num_images / min(np.sqrt(num_images), 10)))

        grid = torchvision.utils.make_grid(
            imgs, nrow=num_rows, pad_value=1,
            normalize=True, scale_each=True
        )

        self._writer.write_image("samples", grid, global_step=epoch)


class TwoDimensionalDensityVisualizer(DensityVisualizer):
    _GRID_SIZE = 300
    _CONTOUR_LEVELS = 50
    _NUM_TRAIN_POINTS_TO_SHOW = 500
    _PADDING = .2
    _BATCH_SIZE = 1000

    def __init__(self, writer, x_train, num_elbo_samples, device):
        super().__init__(writer=writer)

        self._x = x_train

        self._x1_lims = self._lims(self._x[:, 0])
        self._x2_lims = self._lims(self._x[:, 1])

        self._num_elbo_samples = num_elbo_samples

        self._device = device

    def _lims(self, t):
        return (
            t.min().item() - self._PADDING,
            t.max().item() + self._PADDING
        )

    def visualize(self, density, epoch):
        grid_x1, grid_x2 = torch.meshgrid((
            torch.linspace(*self._x1_lims, self._GRID_SIZE),
            torch.linspace(*self._x2_lims, self._GRID_SIZE)
        ))

        x1_x2 = torch.stack((grid_x1, grid_x2), dim=2).view(-1, 2)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x1_x2.to(self._device)),
            batch_size=self._BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False
        )

        probs = []
        for x1_x2_batch, in tqdm.tqdm(loader, leave=False, desc="Plotting"):
            with torch.no_grad():
                log_prob = metrics(density, x1_x2_batch, self._num_elbo_samples)["log-prob"]
            probs.append(torch.exp(log_prob))

        probs = torch.cat(probs, dim=0).view(*grid_x1.shape).cpu()

        plt.figure()

        contours = plt.contourf(grid_x1, grid_x2, probs, levels=self._CONTOUR_LEVELS, cmap="coolwarm")
        for c in contours.collections:
            c.set_edgecolor("face")
        cb = plt.colorbar()
        cb.solids.set_edgecolor("face")

        x = self._x.cpu()
        x = x[torch.randint(high=x.shape[0], size=(self._NUM_TRAIN_POINTS_TO_SHOW,))]
        plt.scatter(x[:, 0], x[:, 1], c="k", marker=".", s=7, linewidth=0.5, alpha=0.5)

        self._writer.write_figure("density", plt.gcf(), epoch)
