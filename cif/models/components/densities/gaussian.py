import numpy as np

import torch
import torch.nn as nn

from .density import Density


def diagonal_gaussian_log_prob(w, means, stddevs):
    assert means.shape == stddevs.shape == w.shape

    flat_w = w.flatten(start_dim=1)
    flat_means = means.flatten(start_dim=1)
    flat_vars = stddevs.flatten(start_dim=1)**2

    _, dim = flat_w.shape

    const_term = -.5*dim*np.log(2*np.pi)
    log_det_terms = -.5*torch.sum(torch.log(flat_vars), dim=1, keepdim=True)
    product_terms = -.5*torch.sum((flat_w - flat_means)**2 / flat_vars, dim=1, keepdim=True)

    return const_term + log_det_terms + product_terms


def diagonal_gaussian_sample(means, stddevs):
    return stddevs*torch.randn_like(means) + means


def diagonal_gaussian_entropy(stddevs):
    flat_stddevs = stddevs.flatten(start_dim=1)
    _, dim = flat_stddevs.shape
    return torch.sum(torch.log(flat_stddevs), dim=1, keepdim=True) + .5*dim*(1 + np.log(2*np.pi))


class DiagonalGaussianDensity(Density):
    def __init__(self, mean, stddev, num_fixed_samples=0):
        super().__init__()
        assert mean.shape == stddev.shape
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

        if num_fixed_samples > 0:
            self.register_buffer("_fixed_samples", self.sample(num_fixed_samples))

    @property
    def shape(self):
        return self.mean.shape

    def _fix_random_u(self):
        return self, self.sample(num_samples=1)[0]

    def fix_u(self, u):
        assert not u
        return self

    def _elbo(self, z, reparam):
        log_prob = diagonal_gaussian_log_prob(
            z,
            self.mean.expand_as(z),
            self.stddev.expand_as(z),
        )
        return {
            "elbo": log_prob,
            "log_p_u": log_prob,
            "log_q_u": 0.,
            "z": z
        }

    def _sample(self, num_samples):
        return diagonal_gaussian_sample(
            self.mean.expand(num_samples, *self.shape),
            self.stddev.expand(num_samples, *self.shape)
        )

    def _fixed_sample(self, noise):
        return noise if noise is not None else self._fixed_samples


class DiagonalGaussianConditionalDensity(nn.Module):
    def __init__(
            self,
            coupler
    ):
        super().__init__()
        self.coupler = coupler

    def forward(self, mode, *args):
        if mode == "log-prob":
            return self._log_prob(*args)
        elif mode == "sample":
            return self._sample(*args)
        elif mode == "entropy":
            return self._entropy(*args)
        else:
            assert False, f"Invalid mode {mode}"

    def log_prob(self, inputs, cond_inputs):
        return self("log-prob", inputs, cond_inputs)

    def sample(self, cond_inputs, detach=False):
        return self("sample", cond_inputs, detach)

    def _log_prob(self, inputs, cond_inputs):
        means, stddevs = self._means_and_stddevs(cond_inputs)
        return {
            "log-prob": diagonal_gaussian_log_prob(inputs, means, stddevs)
        }

    def _sample(self, cond_inputs, detach):
        means, stddevs = self._means_and_stddevs(cond_inputs)
        samples = diagonal_gaussian_sample(means, stddevs)

        if detach:
            samples = samples.detach()

        log_probs = diagonal_gaussian_log_prob(samples, means, stddevs)

        return {
            "sample": samples,
            # We return in addition to samples so that we can avoid two forward
            # passes through self.coupler
            "log-prob": log_probs
        }

    def entropy(self, cond_inputs):
        _, stddevs = self._means_and_stddevs(cond_inputs)
        return diagonal_gaussian_entropy(stddevs)

    def _means_and_stddevs(self, cond_inputs):
        result = self.coupler(cond_inputs)
        return result["shift"], torch.exp(result["log-scale"])
