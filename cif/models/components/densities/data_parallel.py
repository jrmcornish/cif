import torch.nn as nn


class DataParallelDensity(nn.DataParallel):
    def elbo(self, x, detach_q_params=False, detach_q_samples=False):
        return self(
            "elbo",
            x,
            detach_q_params=detach_q_params,
            detach_q_samples=detach_q_samples
        )

    def p_parameters(self):
        return self.module.p_parameters()

    def q_parameters(self):
        return self.module.q_parameters()

    def sample(self, num_samples):
        # Bypass DataParallel
        return self.module.sample(num_samples)

    def fixed_sample(self, noise=None):
        # Bypass DataParallel
        return self.module.fixed_sample(noise=noise)
