import torch

from .density import Density


class DequantizationDensity(Density):
    def __init__(self, density):
        super().__init__()
        self.density = density
 
    def _elbo(self, x):
        return self.density.elbo(x.add_(torch.rand_like(x)))

    def _sample(self, num_samples):
        return self.density.sample(num_samples)

    def _fixed_sample(self):
        return self.density.fixed_sample()


class MapDataBeforeEvalDensity(Density):
    def __init__(self, density, x):
        super().__init__()
        self._density = density
        self.register_buffer("_x", x)

    # We need to do it like this, i.e. we can't just override self.eval(), since
    # nn.Module.eval() just calls train(train_mode=False), so it wouldn't be called
    # recursively by modules containing this one.
    def train(self, train_mode=True):
        if not train_mode:
            self.training = True
            with torch.no_grad():
                self.log_prob(self._x)
        super().train(train_mode)

    def _log_prob(self, x):
        return self._density.log_prob(x)

    def _sample(self, num_samples):
        return self._density.sample(num_samples)

    def _fixed_sample(self):
        return self._density.fixed_sample()
