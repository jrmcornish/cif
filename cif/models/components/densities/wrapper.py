import torch
import torch.nn as nn
import torch.distributions as dist

from .density import Density
from ..networks import LipschitzNetwork


class WrapperDensity(Density):
    def __init__(self, density):
        super().__init__()
        self.density = density

    def p_parameters(self):
        return self.density.p_parameters()

    def q_parameters(self):
        return self.density.q_parameters()

    def _elbo(self, x, detach_q_params, detach_q_samples):
        return self.density.elbo(
            x,
            detach_q_params=detach_q_params,
            detach_q_samples=detach_q_samples
        )

    def _sample(self, num_samples):
        return self.density.sample(num_samples)

    def _fixed_sample(self, noise):
        return self.density.fixed_sample(noise=noise)


class DequantizationDensity(WrapperDensity):
    def _elbo(self, x, detach_q_params, detach_q_samples):
        return super()._elbo(
            x.add_(torch.rand_like(x)),
            detach_q_params=detach_q_params,
            detach_q_samples=detach_q_samples
        )


class BinarizationDensity(WrapperDensity):
    def __init__(self, density, scale):
        super().__init__(density)
        self.scale = scale

    def _elbo(self, x, detach_q_params, detach_q_samples):
        bernoulli = dist.bernoulli.Bernoulli(probs=x / self.scale)
        return super()._elbo(
            bernoulli.sample(),
            detach_q_params=detach_q_params,
            detach_q_samples=detach_q_samples
        )


class PassthroughBeforeEvalDensity(WrapperDensity):
    def __init__(self, density, x):
        super().__init__(density)

        # XXX: It is inefficient to store the data separately, but this will work for # the (non-image) datasets we consider
        self.register_buffer("x", x)

    # We need to do it like this, i.e. we can't just override self.eval(), since
    # nn.Module.eval() just calls train(train_mode=False), so it wouldn't be called
    # recursively by modules containing this one.
    # TODO: Could do with hooks
    def train(self, train_mode=True):
        # TODO: Implement - need to account for reparam
        raise NotImplementedError
        if not train_mode:
            self.training = True
            with torch.no_grad():
                self.elbo(self.x)
        super().train(train_mode)
