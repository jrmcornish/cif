import numpy as np


def metrics(density, x, num_elbo_samples):
    x_samples = x.repeat_interleave(num_elbo_samples, dim=0)

    result = density.elbo(x_samples)

    elbo_samples = result["elbo"].view(x.shape[0], num_elbo_samples, 1)
    elbo = elbo_samples.mean(dim=1)

    log_prob = elbo_samples.logsumexp(dim=1) - np.log(num_elbo_samples)

    dim = int(np.prod(x.shape[1:]))
    bpd = -log_prob / dim / np.log(2)

    elbo_gap = log_prob - elbo

    return {
        "elbo": elbo,
        "log-prob": log_prob,
        "bpd": bpd,
        "elbo-gap": elbo_gap
    }
