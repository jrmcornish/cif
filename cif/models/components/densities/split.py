import torch

from .density import Density


class SplitDensity(Density):
    def __init__(self, density_1, density_2, dim):
        super().__init__()

        self.density_1 = density_1
        self.density_2 = density_2
        self.dim = dim

    def _elbo(self, x, detach_q_params, detach_q_samples):
        x1, x2 = torch.chunk(x, chunks=2, dim=self.dim)

        result_1 = self.density_1(
            "elbo",
            x1,
            detach_q_params=detach_q_params,
            detach_q_samples=detach_q_samples
        )

        result_2 = self.density_2(
            "elbo",
            x2,
            detach_q_params=detach_q_params,
            detach_q_samples=detach_q_samples
        )

        return {
            "log-p": result_1["log-p"] + result_2["log-p"],
            "log-q": result_1["log-q"] + result_2["log-q"],
        }

    def _fixed_sample(self, noise):
        if noise is not None:
            # This should split the noise -- i.e. noise should be chunked on self.dim
            raise NotImplementedError("Proper splitting of noise is not yet implemented")

        x1 = self.density_1.fixed_sample(noise=noise)
        x2 = self.density_2.fixed_sample(noise=noise)
        return torch.cat((x1, x2), dim=self.dim)

    def _sample(self, num_samples):
        x1 = self.density_1.sample(num_samples)
        x2 = self.density_2.sample(num_samples)
        return torch.cat((x1, x2), dim=self.dim)
