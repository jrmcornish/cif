import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from .density import Density


class BijectionDensity(Density):
    def __init__(self, prior, bijection):
        super().__init__()
        self.bijection = bijection
        self.prior = prior

    def _elbo(self, x):
        result = self.bijection.x_to_z(x)
        prior_term = self.prior.elbo(result["z"])["elbo"]
        return {"elbo": prior_term + result["log-jac"]}

    def _sample(self, num_samples):
        z = self.prior.sample(num_samples)
        return self.bijection.z_to_x(z)["x"]

    def _fixed_sample(self):
        z = self.prior.fixed_sample()
        return self.bijection.z_to_x(z)["x"]
