import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from .density import Density


class FlowDensity(Density):
    def __init__(self, prior, bijection):
        super().__init__()
        self.bijection = bijection
        self.prior = prior

    def p_parameters(self):
        return [
            *self.bijection.parameters(),
            *self.prior.p_parameters()
        ]

    def q_parameters(self):
        return self.prior.q_parameters()

    def _fix_random_u(self):
        fixed_prior, z = self.prior._fix_random_u()
        new_z = self.bijection.z_to_x(z.unsqueeze(0))["x"].squeeze(0)
        return FlowDensity(bijection=self.bijection, prior=fixed_prior), new_z

    def fix_u(self, u):
        fixed_prior = self.prior.fix_u(u=u)
        return FlowDensity(bijection=self.bijection, prior=fixed_prior)

    def _elbo(self, x, detach_q_params, detach_q_samples):
        result = self.bijection.x_to_z(x)

        prior_dict = self.prior.elbo(
            result["z"],
            detach_q_params=detach_q_params,
            detach_q_samples=detach_q_samples
        )

        return {
            "elbo": prior_dict["elbo"] + result["log-jac"],
            "log-p": prior_dict["log-p"] + result["log-jac"],
            "log-q": prior_dict["log-q"],
            "bijection-info": result,
            "prior-dict": prior_dict
        }

    def _sample(self, num_samples):
        z = self.prior.sample(num_samples)
        return self.bijection.z_to_x(z)["x"]

    def _fixed_sample(self, noise):
        z = self.prior.fixed_sample(noise=noise)
        return self.bijection.z_to_x(z)["x"]
