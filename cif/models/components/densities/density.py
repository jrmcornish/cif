import numpy as np

import torch
import torch.nn as nn


class Density(nn.Module):
    def forward(self, mode, *args, **kwargs):
        if mode == "elbo":
            return self._elbo(*args, **kwargs)

        elif mode == "sample":
            return self._sample(*args, **kwargs)

        elif mode == "fixed-sample":
            return self._fixed_sample(*args, **kwargs)

        else:
            assert False, f"Invalid mode {mode}"

    def p_parameters(self):
        raise NotImplementedError

    def q_parameters(self):
        raise NotImplementedError

    def fix_random_u(self):
        fixed_density, _ = self._fix_random_u()
        return fixed_density

    def fix_u(self, u):
        raise NotImplementedError

    def elbo(self, x, num_importance_samples, detach_q_params=False, detach_q_samples=False):
        result = self(
            "elbo",
            x.repeat_interleave(num_importance_samples, dim=0),
            detach_q_params=detach_q_params,
            detach_q_samples=detach_q_samples
        )

        output_shape = (x.shape[0], num_importance_samples, 1)

        log_p = result["log-p"].view(output_shape)
        log_q = result["log-q"].view(output_shape)
        log_w = log_p - log_q

        return {"log-p": log_p, "log-q": log_q, "log-w": log_w}

    def sample(self, num_samples):
        return self("sample", num_samples)

    def fixed_sample(self, noise=None):
        return self("fixed-sample", noise)

    def _fix_random_u(self):
        raise NotImplementedError

    def _elbo(self, x, detach_q_params, detach_q_samples):
        raise NotImplementedError

    def _sample(self, num_samples):
        raise NotImplementedError

    def _fixed_sample(self, noise):
        raise NotImplementedError
