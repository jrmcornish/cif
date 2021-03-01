import torch
import torch.nn as nn

from .conditional_density import ConditionalDensity
from ..densities.gaussian import diagonal_gaussian_log_prob, diagonal_gaussian_sample, diagonal_gaussian_entropy


class DiagonalGaussianConditionalDensity(ConditionalDensity):
    def __init__(
            self,
            coupler
    ):
        super().__init__()
        self.coupler = coupler

    def _log_prob(self, inputs, cond_inputs):
        means, stddevs = self._means_and_stddevs(cond_inputs)
        return {
            "log-prob": diagonal_gaussian_log_prob(inputs, means, stddevs)
        }

    def _sample(self, cond_inputs, detach_params, detach_samples):
        means, stddevs = self._means_and_stddevs(cond_inputs)
        samples = diagonal_gaussian_sample(means, stddevs)

        if detach_params:
            means = means.detach()
            stddevs = stddevs.detach()

        if detach_samples:
            samples = samples.detach()

        log_probs = diagonal_gaussian_log_prob(samples, means, stddevs)

        return {
            "sample": samples,
            # We return in addition to samples so that we can avoid two forward
            # passes through self.coupler
            "log-prob": log_probs
        }

    def _means_and_stddevs(self, cond_inputs):
        result = self.coupler(cond_inputs)
        return result["shift"], torch.exp(result["log-scale"])
