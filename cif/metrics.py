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


def iwae(density, x, num_importance_samples, stl):
    log_p_u, log_q_u = _elbo(
        density=density,
        x=x,
        num_importance_samples=num_importance_samples,
        detach_q_params=stl,
        detach_q_samples=False
    )

    loss = -(log_p_u - log_q_u).logsumexp(dim=1).mean()

    return {"loss": loss}


def iwae_dreg(density, x, num_importance_samples):
    log_p_u, log_q_u = _elbo(
        density=density,
        x=x,
        num_importance_samples=num_importance_samples,
        detach_q_params=True,
        detach_q_samples=False
    )

    p_loss = -(log_p_u - log_q_u.detach()).logsumexp(dim=1).mean()

    log_w = log_p_u.detach() - log_q_u
    log_Z = log_w.logsumexp(dim=1).view(x.shape[0], 1, 1)
    grad_weight = (log_w - log_Z).exp().detach()
    q_loss = -(grad_weight**2 * log_w).sum(dim=1).mean()

    return {
        "p_loss": p_loss,
        "q_loss": q_loss
    }


def rws(density, x, num_importance_samples):
    log_p_u, log_q_u = _elbo(
        density=density,
        x=x,
        num_importance_samples=num_importance_samples,
        detach_q_params=False,
        detach_q_samples=True
    )

    p_loss = -(log_p_u - log_q_u.detach()).logsumexp(dim=1).mean()
    q_loss = (log_p_u.detach() - log_q_u).logsumexp(dim=1).mean()

    return {
        "p_loss": p_loss,
        "q_loss": q_loss
    }


def rws_dreg(density, x, num_importance_samples):
    x_samples = x.repeat_interleave(num_importance_samples, dim=0)

    result = density.elbo(x_samples, detach_q_params=True, detach_q_samples=False)

    log_p_u = result["log_p_u"].view(x.shape[0], num_importance_samples, 1)
    log_q_u = result["log_q_u"].view(x.shape[0], num_importance_samples, 1)

    p_loss = -(log_p_u - log_q_u.detach()).logsumexp(dim=1).mean()

    log_w = log_p_u.detach() - log_q_u
    log_Z = log_w.logsumexp(dim=1).view(x.shape[0], 1, 1)
    grad_weight = (log_w - log_Z).exp().detach()
    q_loss = -((grad_weight - grad_weight**2) * log_w).sum(dim=1).mean()

    return {
        "p_loss": p_loss,
        "q_loss": q_loss
    }


# TODO: This is broken now
def ml_ll_ss(density, x, geom_prob):
    K_dist = stats.geom(p=geom_prob, loc=-1)

    # TODO: Correct/optimal to share K across batch?
    K = K_dist.rvs()

    # TODO: This will cause memory problems
    x_samples = x.repeat_interleave(2**(K+1), dim=0)

    result = density.elbo(x_samples, reparam=False)

    log_p_u = result["log_p_u"].view(x.shape[0], 2**(K+1), 1)
    log_q_u = result["log_q_u"].view(x.shape[0], 2**(K+1), 1)

    p_loss = -_ml_ll_ss(log_p_u, log_q_u.detach(), K_dist, K)
    q_loss = _ml_ll_ss(log_p_u.detach(), log_q_u, K_dist, K)

    return {
        "p_loss": p_loss,
        "q_loss": q_loss
    }


def _ml_ll_ss(log_p_u, log_q_u, K_dist, K):
    log_w = log_p_u - log_q_u

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

    return ml_ll_ss.mean()


def _elbo(density, x, num_importance_samples, detach_q_params, detach_q_samples):
    x_samples = x.repeat_interleave(num_importance_samples, dim=0)

    result = density.elbo(
        x_samples,
        detach_q_params=detach_q_params,
        detach_q_samples=detach_q_samples
    )

    output_shape = (x.shape[0], num_importance_samples, 1)

    log_p_u = result["log_p_u"].view(output_shape)
    log_q_u = result["log_q_u"].view(output_shape)

    return log_p_u, log_q_u
