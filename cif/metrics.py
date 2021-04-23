import torch

import numpy as np
import scipy.stats as stats


def metrics(density, x, num_importance_samples):
    result = density.elbo(x, num_importance_samples, detach_q_params=False, detach_q_samples=False)

    elbo_samples = result["log-w"]
    elbo = elbo_samples.mean(dim=1)

    iwae = elbo_samples.logsumexp(dim=1) - np.log(num_importance_samples)

    dim = int(np.prod(x.shape[1:]))
    bpd = -iwae / dim / np.log(2)

    elbo_gap = iwae - elbo

    return {
        "elbo": elbo,
        f"iwae-{num_importance_samples}": iwae,
        "bpd": bpd,
        "elbo-gap": elbo_gap
    }


# NOTE: technically this isn't IwAE since we don't subtract off
# `np.log(num_importance_samples)` However, this isn't a problem since this
# function is intended only to be used only at training time (cf. `metrics()`
# above)
def iwae(density, x, num_importance_samples, detach_q):
    log_w = density.elbo(
        x=x,
        num_importance_samples=num_importance_samples,
        detach_q_params=detach_q,
        detach_q_samples=detach_q
    )["log-w"]

    return -log_w.logsumexp(dim=1).mean()


def iwae_alt(density, x, num_importance_samples, grad_weight_pow):
    log_w = density.elbo(
        x=x,
        num_importance_samples=num_importance_samples,
        detach_q_params=True,
        detach_q_samples=False
    )["log-w"]

    log_Z = log_w.logsumexp(dim=1).view(x.shape[0], 1, 1)
    grad_weight = (log_w - log_Z).exp() ** grad_weight_pow
    return -(grad_weight.detach() * log_w).sum(dim=1).mean()


def rws(density, x, num_importance_samples):
    log_w = density.elbo(
        x=x,
        num_importance_samples=num_importance_samples,
        detach_q_params=False,
        detach_q_samples=True
    )["log-w"]

    log_Z = log_w.logsumexp(dim=1).view(x.shape[0], 1, 1)
    grad_weight = (log_w - log_Z).exp()
    return (grad_weight.detach() * log_w).sum(dim=1).mean()


def rws_dreg(density, x, num_importance_samples):
    log_w = density.elbo(
        x=x,
        num_importance_samples=num_importance_samples,
        detach_q_params=True,
        detach_q_samples=False
    )["log-w"]

    log_Z = log_w.logsumexp(dim=1).view(x.shape[0], 1, 1)
    grad_weight = (log_w - log_Z).exp().detach()
    return -((grad_weight - grad_weight**2) * log_w).sum(dim=1).mean()
