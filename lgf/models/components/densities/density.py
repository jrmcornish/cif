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
