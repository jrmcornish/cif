import torch

from .density import Density


class SplitDensity(Density):
    def __init__(self, density_1, density_2, dim):
        super().__init__()

        self.density_1 = density_1
        self.density_2 = density_2
        self.dim = dim

    def _elbo(self, x):
        # TODO: Implement reparam and log_p_u, log_q_u outputs
        raise NotImplementedError
        x1, x2 = torch.chunk(x, chunks=2, dim=self.dim)
        elbo1 = self.density_1.elbo(x1)["elbo"]
        elbo2 = self.density_2.elbo(x2)["elbo"]
        return {"elbo": elbo1 + elbo2}

    def _fixed_sample(self, noise):
        x1 = self.density_1.fixed_sample(noise=noise)
        x2 = self.density_2.fixed_sample(noise=noise)
        return torch.cat((x1, x2), dim=self.dim)

    def _sample(self, num_samples):
        x1 = self.density_1.sample(num_samples)
        x2 = self.density_2.sample(num_samples)
        return torch.cat((x1, x2), dim=self.dim)
