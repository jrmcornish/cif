import numpy as np
import scipy.stats as stats


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


def ml_ll_ss(density, x, geom_prob):
        # TODO: Discuss parameterisation
    K_dist = stats.geom(p=geom_prob, loc=-1)

    # TODO: Correct/optimal to share K across batch?
    K = K_dist.rvs()

    # TODO: This will cause memory problems
    x_samples = x.repeat_interleave(2**(K+1), dim=0)
    log_w = density.elbo(x_samples)["elbo"].view(x.shape[0], 2**(K+1), 1)

    # This is in keeping with (9) of Blanchet et al.
    log_I0 = log_w[:, 0]

    # For this alternative, setting ml_ll_geom_prob=1 recovers IWAE
    # TODO: Comparison with other approaches?
    # log_I0 = (log_w[:, 0] + log_w[:, 1]) / 2

    upper_level_term = log_w.logsumexp(dim=1) - (K+1)*np.log(2)

    log_O = log_w[:, :2**K].logsumexp(dim=1) - K*np.log(2)
    log_E = log_w[:, 2**K:].logsumexp(dim=1) - K*np.log(2)
    lower_level_term = (log_O + log_E) / 2

    ml_ll_ss = log_I0 + (upper_level_term - lower_level_term) / K_dist.pmf(K)

    return {"loss": -ml_ll_ss.mean()}
