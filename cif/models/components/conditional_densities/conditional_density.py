import torch.nn as nn

class ConditionalDensity(nn.Module):
    def forward(self, mode, *args, **kwargs):
        if mode == "log-prob":
            return self._log_prob(*args, **kwargs)
        elif mode == "sample":
            return self._sample(*args, **kwargs)
        else:
            assert False, f"Invalid mode {mode}"

    def log_prob(self, inputs, cond_inputs):
        return self("log-prob", inputs, cond_inputs)

    def sample(self, cond_inputs, detach_params=False, detach_samples=False):
        return self(
            "sample",
            cond_inputs,
            detach_params=detach_params,
            detach_samples=detach_samples
        )

    def _log_prob(self, inputs, cond_inputs):
        raise NotImplementedError

    def _sample(self, cond_inputs, detach_params, detach_samples):
        raise NotImplementedError
