import torch
import torch.nn as nn

from ..densities.gaussian import diagonal_gaussian_log_prob, diagonal_gaussian_sample, diagonal_gaussian_entropy


class DiagonalGaussianConditionalDensity(nn.Module):
    def __init__(
            self,
            coupler
    ):
        super().__init__()
        self.coupler = coupler

    def forward(self, mode, *args, **kwargs):
        if mode == "log-prob":
            return self._log_prob(*args, **kwargs)
        elif mode == "sample":
            return self._sample(*args, **kwargs)
        elif mode == "entropy":
            return self._entropy(*args, **kwargs)
        else:
            assert False, f"Invalid mode {mode}"

    def log_prob(self, inputs, cond_inputs):
        return self("log-prob", inputs, cond_inputs)

    def sample(self, cond_inputs, detach_params=False, detach_samples=False):
        return self(
            "sample",
            cond_inputs,
            detach_params=detach_params,
            detach_samples=detach_samples
        )

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

    def entropy(self, cond_inputs):
        _, stddevs = self._means_and_stddevs(cond_inputs)
        return diagonal_gaussian_entropy(stddevs)

    def _means_and_stddevs(self, cond_inputs):
        result = self.coupler(cond_inputs)
        return result["shift"], torch.exp(result["log-scale"])
