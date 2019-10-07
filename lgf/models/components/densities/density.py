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

    def metrics(self, x, num_elbo_samples=1):
        x_samples = x.repeat_interleave(num_elbo_samples, dim=0)
        result = self.elbo(x_samples)

        elbo_samples = result["elbo"].view(x.shape[0], num_elbo_samples, 1)
        elbo = elbo_samples.mean(dim=1)

        log_prob = elbo_samples.logsumexp(dim=1) - np.log(num_elbo_samples)

        dim = int(np.prod(x.shape[1:]))
        bpd = -log_prob / dim / np.log(2)

        elbo_gap = log_prob - elbo

        return {
            "elbo": elbo,
            "log-prob": log_prob,
            "elbo-gap": elbo_gap,
            "bpd": bpd
        }

    def elbo(self, x):
        return self("elbo", x)

    def sample(self, num_samples):
        return self("sample", num_samples)

    def fixed_sample(self):
        return self("fixed-sample")

    def _elbo(self, x):
        raise NotImplementedError

    def _sample(self, num_samples):
        raise NotImplementedError

    def _fixed_sample(self):
        raise NotImplementedError


class MapDataBeforeEvalDensity(Density):
    def __init__(self, density, x):
        super().__init__()
        self._density = density
        self.register_buffer("_x", x)

    # We need to do it like this, i.e. we can't just override self.eval(), since
    # nn.Module.eval() just calls train(train_mode=False), so it wouldn't be called
    # recursively by modules containing this one.
    def train(self, train_mode=True):
        if not train_mode:
            self.training = True
            with torch.no_grad():
                self.log_prob(self._x)
        super().train(train_mode)

    def _log_prob(self, x):
        return self._density.log_prob(x)

    def _sample(self, num_samples):
        return self._density.sample(num_samples)

    def _fixed_sample(self):
        return self._density.fixed_sample()
