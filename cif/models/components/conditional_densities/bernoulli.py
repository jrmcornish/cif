import torch
import torch.nn as nn
import torch.distributions as dist

from .conditional_density import ConditionalDensity


class BernoulliConditionalDensity(ConditionalDensity):
    def __init__(
            self,
            logit_net
    ):
        super().__init__()
        self.logit_net = logit_net

    def _log_prob(self, inputs, cond_inputs):
        logits = self.logit_net(cond_inputs)
        log_probs = dist.bernoulli.Bernoulli(logits=logits).log_prob(inputs)
        return {
            "log-prob": log_probs.flatten(start_dim=1).sum(dim=1, keepdim=True)
        }

    def _sample(self, cond_inputs, detach_params, detach_samples):
        logits = self.logit_net(cond_inputs)
        bernoulli = dist.bernoulli.Bernoulli(logits=logits)
        samples = bernoulli.sample()

        # Doesn't really make sense to do this for Bernoullis, but also
        # no harm done
        if detach_params:
            logits = logits.detach()

        if detach_samples:
            samples = samples.detach()

        log_probs = bernoulli.log_prob(samples).flatten(start_dim=1).sum(dim=1, keepdim=True)

        return {
            "sample": samples,
            # We return in addition to samples so that we can avoid two forward
            # passes through self.coupler
            "log-prob": log_probs
        }
