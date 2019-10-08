import numpy as np

import torch
import torch.nn as nn

from .density import Density


def concrete_log_prob(u, alphas, lam):
    assert alphas.shape == u.shape

    flat_u = u.flatten(start_dim=1)
    flat_alphas = alphas.flatten(start_dim=1)

    _, dim = flat_u.shape

    const_term = np.sum(np.log(np.arange(1, dim))) + (dim - 1)*np.log(lam)
    log_denominator = torch.logsumexp(torch.log(flat_alphas) - lam*torch.log(flat_u), 
        dim=1, keepdim=True)
    log_numerator = torch.log(flat_alphas) - (lam + 1)*torch.log(flat_u)
    log_product_quotient = torch.sum(log_numerator - log_denominator, dim=1, keepdim=True)
    return const_term + log_product_quotient


def concrete_sample(alphas, lam):
    standard_gumbel = torch.distributions.gumbel.Gumbel(
        torch.zeros_like(alphas),
        torch.ones_like(alphas)
    )
    gumbels = standard_gumbel.sample()
    log_numerator = (torch.log(alphas) + gumbels) / lam
    log_denominator = torch.logsumexp(log_numerator, dim=1, keepdim=True)
    return torch.exp(log_numerator - log_denominator)


class ConcreteDensity(Density):
    def __init__(self, alphas, lam, num_fixed_samples=0):
        super().__init__()
        assert len(alphas.shape) == 1
        self.alphas = nn.Parameter(alphas)
        self.lam = lam

        if num_fixed_samples > 0:
            self.register_buffer("_fixed_samples", self.sample(num_fixed_samples))

    @property
    def shape(self):
        return self.alphas.shape

    def _elbo(self, u):
        log_prob = concrete_log_prob(
            u,
            self.alphas.expand_as(u),
            self.lam
        )
        return {"elbo": log_prob}

    # TODO: Pass through forward
    def _sample(self, num_samples):
        return concrete_sample(
            self.alphas.expand(num_samples, *self.shape),
            self.lam
        )

    def _fixed_sample(self):
        return self._fixed_samples


class ConcreteConditionalDensity(nn.Module):
    def __init__(
            self,
            log_alpha_map,
            lam
    ):
        super().__init__()
        self.log_alpha_map = log_alpha_map
        self.lam = lam

    # TODO: Should pass through forward
    def log_prob(self, inputs, cond_inputs):
        return {
            "log-prob": concrete_log_prob(inputs, self._alphas(cond_inputs), self.lam)
        }

    # TODO: Should pass through forward
    def sample(self, cond_inputs):
        return concrete_sample(self._alphas(cond_inputs), self.lam)

    def entropy(self, cond_inputs):
        raise NotImplementedError

    def _alphas(self, cond_inputs):
        return torch.exp(self.log_alpha_map(cond_inputs))
