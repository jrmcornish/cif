from itertools import chain

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from .density import Density


class BijectionDensity(Density):
    def __init__(self, prior, bijection):
        super().__init__()
        self.bijection = bijection
        self.prior = prior

    def p_parameters(self):
        return chain(self.bijection.parameters(), self.prior.p_parameters())

    def q_parameters(self):
        return self.prior.q_parameters()

    def _fix_random_u(self):
        fixed_prior, z = self.prior._fix_random_u()
        new_z = self.bijection.z_to_x(z.unsqueeze(0))["x"].squeeze(0)
        return BijectionDensity(bijection=self.bijection, prior=fixed_prior), new_z

    def fix_u(self, u):
        fixed_prior = self.prior.fix_u(u=u)
        return BijectionDensity(bijection=self.bijection, prior=fixed_prior)

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


class BijectionMixtureDensity(Density):
    def __init__(self, prior, bijections, weight_map):
        super().__init__()
        assert bijections, "Must have at least one bijection"
        self.prior = prior
        self.bijections = nn.ModuleList(bijections)
        self.weight_map = weight_map

    def _elbo(self, x):
        # TODO: Implement reparam and log_p_u, log_q_u outputs
        raise NotImplementedError
        K = len(self.bijections)

        results = [b.x_to_z(x) for b in self.bijections]

        log_jac_terms = torch.stack([result["log-jac"] for result in results])

        # This will have the z values for each x interleaved across bijections
        zs = torch.cat([result["z"] for result in results])

        log_prior_terms = self.prior.elbo(zs)["elbo"].view(K, x.shape[0], 1)

        log_weight_terms = self._log_weights(zs).view(K, x.shape[0], K)
        log_weight_terms = [w[:, k].view(x.shape[0], 1) for k, w in enumerate(log_weight_terms)]
        log_weight_terms = torch.stack(log_weight_terms)

        logsumexp_terms = log_jac_terms + log_prior_terms + log_weight_terms
        elbo = torch.logsumexp(logsumexp_terms, dim=0)

        return {"elbo": elbo}

    # TODO: Variable names here are a mess
    def _fixed_sample(self, noise):
        z = self.prior.fixed_sample(noise=noise)

        bucketed_z = [[] for _ in self.bijections]
        k = []
        for i, (zi, log_weights) in enumerate(zip(z, self._log_weights(z))):
            ki = Categorical(logits=log_weights).sample().item()
            bucketed_z[ki].append(zi)
            k.append(ki)

        results = []
        for ki, z_batch in enumerate(bucketed_z):
            if len(z_batch) > 0:
                z_batch = torch.stack(z_batch)
                results.append(self.bijections[ki].z_to_x(z_batch))
            else:
                results.append(None)

        result = []
        for ki in k:
            result.append(results[ki]["x"][0])
            results[ki]["x"] = results[ki]["x"][1:]

        return torch.stack(result)

    def _log_weights(self, z):
        return self.weight_map(z.flatten(start_dim=1))
