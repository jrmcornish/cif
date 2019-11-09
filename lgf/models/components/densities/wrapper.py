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


class PassthroughBeforeEvalDensity(Density):
    def __init__(self, density, x):
        super().__init__()

        self.density = density

        # XXX: It is inefficient to store the data separately, but this will work for # the (non-image) datasets we consider
        self.register_buffer("x", x)

    # We need to do it like this, i.e. we can't just override self.eval(), since
    # nn.Module.eval() just calls train(train_mode=False), so it wouldn't be called
    # recursively by modules containing this one.
    def train(self, train_mode=True):
        if not train_mode:
            self.training = True
            with torch.no_grad():
                self.elbo(self.x)
        super().train(train_mode)

    def _elbo(self, x):
        return self.density.elbo(x)

    def _sample(self, num_samples):
        return self.density.sample(num_samples)

    def _fixed_sample(self):
        return self.density.fixed_sample()
