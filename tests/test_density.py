import unittest

import numpy as np
import scipy.stats as stats 

import torch
import torch.nn as nn

from cif.models.components.densities import (
    DiagonalGaussianDensity,
    DiagonalGaussianConditionalDensity,
    ELBODensity,
    ConcreteConditionalDensity
)
from cif.models.components.couplers import ChunkedSharedCoupler
from cif.models.components.networks import get_mlp
from cif.models.components.bijections import ConditionalAffineBijection
from cif.models.factory import get_coupler


class TestDiagonalGaussianDensity(unittest.TestCase):
    def setUp(self):
        self.shape = (10, 4, 2)
        self.mean = torch.rand(self.shape)
        self.stddev = 1 + torch.rand(self.shape)**2
        self.density = DiagonalGaussianDensity(self.mean, self.stddev, num_fixed_samples=64)

        flat_mean = self.mean.flatten().numpy()
        flat_vars = (self.stddev**2).flatten().numpy()
        self.scipy_density = stats.multivariate_normal(mean=flat_mean, cov=flat_vars)

    def test_elbo(self):
        batch_size = 1000
        w = torch.rand(batch_size, *self.shape)
        with torch.no_grad():
            log_prob = self.density.elbo(w)["elbo"]

        flat_w = w.flatten(start_dim=1).numpy()
        scipy_log_prob = self.scipy_density.logpdf(flat_w).reshape(batch_size, 1)

        self.assertEqual(log_prob.shape, (batch_size, 1))
        self.assertLessEqual(abs((log_prob.numpy() - scipy_log_prob).max()), 1e-4)

    def test_samples(self):
        num_samples = 100000

        samples = self.density.sample(num_samples)

        self.assertEqual(samples.shape, (num_samples, *self.shape))

        flat_samples = samples.flatten(start_dim=1)
        flat_mean = self.mean.flatten()
        flat_stddev = self.stddev.flatten()
        self._assert_moments_accurate(flat_samples, flat_mean, flat_stddev)

    def _assert_moments_accurate(self, flat_samples, flat_mean, flat_stddev):
        num_moments = 4
        eps = 0.5

        _, dim = flat_samples.shape

        tot_errors = 0
        tot_trials = 0
        for m in range(1, num_moments+1):
            moments = torch.mean(flat_samples**m, dim=0)
            for i in range(dim):
                tot_trials += 1
                ground_truth = stats.norm.moment(m, loc=flat_mean[i], scale=flat_stddev[i])
                if (ground_truth - moments[i]).abs() > eps:
                    tot_errors += 1

        self.assertLess(tot_errors / tot_trials, 0.05)


class TestDiagonalGaussianConditionalDensity(unittest.TestCase):
    def setUp(self):
        dim = 25
        cond_dim = 15
        self.shape = (dim,)
        self.cond_shape = (cond_dim,)

        self.mean_log_std_map = ChunkedSharedCoupler(
            shift_log_scale_net=get_mlp(
                num_input_channels=cond_dim,
                hidden_channels=[10, 10, 10],
                num_output_channels=2*dim,
                activation=nn.Tanh
            )
        )
        self.density = DiagonalGaussianConditionalDensity(self.mean_log_std_map)

    def test_log_prob(self):
        batch_size = 100
        inputs = torch.rand(batch_size, *self.shape)
        cond_inputs = torch.rand(batch_size, *self.cond_shape)

        with torch.no_grad():
            log_prob = self.density.log_prob(inputs, cond_inputs)["log-prob"]
            mean_log_std = self.mean_log_std_map(cond_inputs)

        means = mean_log_std["shift"]
        stds = torch.exp(mean_log_std["log-scale"])

        scipy_log_probs = stats.norm.logpdf(inputs, loc=means, scale=stds)
        scipy_log_prob = scipy_log_probs.reshape((batch_size, -1)).sum(axis=1, keepdims=True)

        self.assertLessEqual(abs((log_prob.numpy() - scipy_log_prob).max()), 1e-4)

    def test_samples(self):
        batch_size = 10
        num_samples = 10000
        num_moments = 2

        cond_inputs = torch.rand(batch_size, *self.cond_shape)

        with torch.no_grad():
            result = self.density.sample(cond_inputs.repeat_interleave(num_samples, dim=0))
            mean_log_std = self.mean_log_std_map(cond_inputs)

        samples = result["sample"]

        self.assertEqual(samples.shape, (batch_size * num_samples, *self.shape))

        samples = samples.view(batch_size, num_samples, *self.shape)
        means = mean_log_std["shift"].flatten()
        stds = torch.exp(mean_log_std["log-scale"]).flatten()
        for m in range(1, num_moments+1):
            moments = torch.mean(samples**m, dim=1)
            ground_truth = torch.empty_like(moments)
            for i, x in enumerate(moments.flatten()):
                ground_truth.view(-1)[i] = stats.norm.moment(
                    m, loc=means[i], scale=stds[i]
                )

            errs = (moments - ground_truth).abs()
            self.assertLess(errs.max(), 0.5)
            self.assertLess(errs.mean(), 0.05)

    def test_entropy(self):
        batch_size = 1000
        cond_inputs = torch.rand(batch_size, *self.cond_shape)

        with torch.no_grad():
            entropies = self.density.entropy(cond_inputs)
            mean_log_std = self.mean_log_std_map(cond_inputs)

        self.assertEqual(entropies.shape, (batch_size, 1))

        log_stddev = mean_log_std["log-scale"]
        ground_truth_entropies = stats.norm.entropy(scale=torch.exp(log_stddev)).reshape(batch_size, -1).sum(axis=1, keepdims=True)
        errors = np.abs(ground_truth_entropies - entropies.numpy())
        self.assertLess(errors.max(), 1e-4)


class TestELBODensity(unittest.TestCase):
    def test_log_prob_format(self):
        batch_size = 1000
        x_dim = 40
        input_shape = (x_dim,)
        u_dim = 15

        prior = DiagonalGaussianDensity(
            mean=torch.zeros(input_shape),
            stddev=torch.ones(input_shape)
        )

        p_u_density = self._u_density(u_dim, x_dim)

        bijection = ConditionalAffineBijection(
            x_shape=input_shape,
            coupler=get_coupler(
                input_shape=(u_dim,),
                num_channels_per_output=x_dim,
                config={
                    "independent_nets": False,
                    "shift_log_scale_net": {
                        "type": "mlp",
                        "hidden_channels": [40, 30],
                        "activation": "tanh"
                    }
                }
            )
        )

        q_u_density = self._u_density(u_dim, x_dim)

        density = ELBODensity(
            prior=prior,
            p_u_density=p_u_density,
            bijection=bijection,
            q_u_density=q_u_density,
        )

        x = torch.rand(batch_size, *input_shape)
        elbo = density.elbo(x)["elbo"]

        self.assertEqual(elbo.shape, (batch_size, 1))

    def _u_density(self, u_dim, x_dim):
        return DiagonalGaussianConditionalDensity(
            coupler=ChunkedSharedCoupler(
                shift_log_scale_net=get_mlp(
                    num_input_channels=x_dim,
                    hidden_channels=[10, 10, 10],
                    num_output_channels=2*u_dim,
                    activation=nn.Tanh
                )
            )
        )


# TODO: Find reference distribution to compare concrete against
class TestConcreteConditionalDensity(unittest.TestCase):
    def setUp(self):
        self.shape = (25,)
        self.cond_shape = (5,)
        self.lam = torch.exp(torch.rand(1))

        self.log_alpha_map = get_mlp(
            num_input_channels=np.prod(self.cond_shape),
            hidden_channels=[10, 10, 10],
            num_output_channels=np.prod(self.shape),
            activation=nn.Tanh
        )
        self.density = ConcreteConditionalDensity(self.log_alpha_map, self.lam)

    def test_samples(self):
        batch_size = 10
        num_samples = 10000
        num_moments = 2

        cond_inputs = torch.rand(batch_size, *self.cond_shape)

        with torch.no_grad():
            samples = self.density.sample(cond_inputs.repeat_interleave(num_samples, dim=0))["sample"]

        self.assertEqual(samples.shape, (batch_size * num_samples, *self.shape))


if __name__ == "__main__":
    unittest.main()
