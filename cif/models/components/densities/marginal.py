from itertools import chain

from ..conditional_densities.conditional_density import ConditionalDensity
from .density import Density


class MarginalDensity(Density):
    def __init__(
            self,
            prior: Density,
            likelihood: ConditionalDensity,
            approx_posterior: ConditionalDensity
    ):
        super().__init__()
        self.prior = prior
        self.likelihood = likelihood
        self.approx_posterior = approx_posterior

    def p_parameters(self):
        return [*self.prior.parameters(), *self.likelihood.parameters()]

    def q_parameters(self):
        return [
            *self.approx_posterior.parameters(),
            *self.prior.q_parameters()
        ]

    def _elbo(self, x, detach_q_params, detach_q_samples):
        approx_posterior = self.approx_posterior.sample(
            cond_inputs=x,
            detach_params=detach_q_params,
            detach_samples=detach_q_samples
        )

        likelihood = self.likelihood.log_prob(
            inputs=x,
            cond_inputs=approx_posterior["sample"]
        )

        prior = self.prior.elbo(
            approx_posterior["sample"],
            detach_q_params=detach_q_params,
            detach_q_samples=detach_q_samples
        )

        log_p = likelihood["log-prob"] + prior["log-p"]
        log_q = approx_posterior["log-prob"] + prior["log-q"]

        return {
            "log-p": log_p,
            "log-q": log_q,
            # TODO: Don't really need this any more
            "elbo": log_p - log_q
        }

    def _sample(self, num_samples):
        z = self.prior.sample(num_samples)
        return self.likelihood.sample(cond_inputs=z)["sample"]

    def _fixed_sample(self, noise):
        z = self.prior.fixed_sample(noise=noise)
        return self.likelihood.sample(cond_inputs=z)["sample"]
