import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from .density import Density


class FlowMixtureDensity(Density):
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
