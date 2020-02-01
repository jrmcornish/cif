import numpy as np

import torch
import torch.nn as nn


class Density(nn.Module):
    def forward(self, mode, *args):
        if mode == "elbo":
            return self._elbo(*args)

        elif mode == "sample":
            return self._sample(*args)

        elif mode == "fixed-sample":
            return self._fixed_sample(*args)

        else:
            assert False, f"Invalid mode {mode}"

    def fix_random_u(self):
        fixed_density, _ = self._fix_random_u()
        return fixed_density

    def _fix_random_u(self):
        raise NotImplementedError

    def fix_u(self, u):
        raise NotImplementedError

    def elbo(self, x):
        return self("elbo", x)

    def sample(self, num_samples):
        return self("sample", num_samples)

    def fixed_sample(self, noise=None):
        return self("fixed-sample", noise)

    def _elbo(self, x):
        raise NotImplementedError

    def _sample(self, num_samples):
        raise NotImplementedError

    def _fixed_sample(self, noise):
        raise NotImplementedError
