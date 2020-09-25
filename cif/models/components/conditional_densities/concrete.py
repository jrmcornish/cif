import numpy as np

import torch
import torch.nn as nn

from .conditional_density import ConditionalDensity


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


class ConcreteConditionalDensity(ConditionalDensity):
    def __init__(
            self,
            log_alpha_map,
            lam
    ):
        super().__init__()
        self.log_alpha_map = log_alpha_map
        self.lam = lam

    def _log_prob(self, inputs, cond_inputs):
        return {
            "log-prob": concrete_log_prob(inputs, self._alphas(cond_inputs), self.lam)
        }

    def _sample(self, cond_inputs, detach_params, detach_samples):
        alphas = self._alphas(cond_inputs)
        samples = concrete_sample(alphas, self.lam)

        if detach_params:
            # NOTE: We assume self.lam is not a parameter
            alphas = alphas.detach()

        if detach_samples:
            samples = samples.detach()

        log_probs = concrete_log_prob(samples, alphas, self.lam)

        return {
            "log-prob": log_probs,
            "sample": samples
        }

    def _alphas(self, cond_inputs):
        return torch.exp(self.log_alpha_map(cond_inputs))
