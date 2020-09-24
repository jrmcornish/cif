import torch
import torch.nn as nn
import torch.distributions as dist


class BernoulliConditionalDensity(nn.Module):
    def __init__(
            self,
            logit_net
    ):
        super().__init__()
        self.logit_net = logit_net

    # TODO: DRY
    def forward(self, mode, *args, **kwargs):
        if mode == "log-prob":
            return self._log_prob(*args, **kwargs)
        elif mode == "sample":
            return self._sample(*args, **kwargs)
        else:
            assert False, f"Invalid mode {mode}"

    def log_prob(self, inputs, cond_inputs):
        return self("log-prob", inputs, cond_inputs)

    def sample(self, cond_inputs, detach_params=False, detach_samples=False):
        return self(
            "sample",
            cond_inputs=cond_inputs,
            detach_params=detach_params,
            detach_samples=detach_samples
        )

    def _log_prob(self, inputs, cond_inputs):
        logits = self.logit_net(cond_inputs)
        log_probs = dist.bernoulli.Bernoulli(logits=logits).log_prob(inputs)
        return {
            "log-prob": log_probs.flatten(start_dim=1).sum(dim=1, keepdim=True)
        }

    def _sample(self, cond_inputs, detach_params, detach_samples):
        logits = self.logit_net(cond_inputs)

        # Doesn't really make sense to do this for Bernoullis, but also
        # no harm done
        if detach_params:
            logits = logits.detach()

        bernoulli = dist.bernoulli.Bernoulli(logits=logits)

        samples = bernoulli.sample()

        if detach_samples:
            samples = samples.detach()

        # TODO: DRY
        log_probs = bernoulli.log_prob(samples).flatten(start_dim=1).sum(dim=1, keepdim=True)

        return {
            "sample": samples,
            # We return in addition to samples so that we can avoid two forward
            # passes through self.coupler
            "log-prob": log_probs
        }
