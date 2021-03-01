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

    def p_parameters(self):
        return []

    def q_parameters(self):
        return []

    def _fix_random_u(self):
        return self, self.sample(num_samples=1)[0]

    def fix_u(self, u):
        assert not u
        return self

    def _elbo(self, z, detach_q_params, detach_q_samples):
        log_prob = diagonal_gaussian_log_prob(
            z,
            self.mean.expand_as(z),
            self.stddev.expand_as(z),
        )

        return {
            "log-p": log_prob,
            "log-q": z.new_zeros((z.shape[0], 1)),
            "z": z
        }

    def _sample(self, num_samples):
        return diagonal_gaussian_sample(
            self.mean.expand(num_samples, *self.shape),
            self.stddev.expand(num_samples, *self.shape)
        )

    def _fixed_sample(self, noise):
        return noise if noise is not None else self._fixed_samples
