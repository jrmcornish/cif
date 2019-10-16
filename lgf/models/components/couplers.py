import torch.nn as nn


class IndependentCoupler(nn.Module):
    def __init__(self, shift_net, log_scale_net):
        super().__init__()
        self.shift_net = shift_net
        self.log_scale_net = log_scale_net

    def forward(self, inputs):
        return {
            "shift": self.shift_net(inputs),
            "log-scale": self.log_scale_net(inputs)
        }


class SharedCoupler(nn.Module):
    _NUM_CHUNKS = 2
    _CHANNEL_DIM = 1

    def __init__(self, shift_log_scale_net):
        super().__init__()
        self.shift_log_scale_net = shift_log_scale_net

    def forward(self, inputs):
        result = self.shift_log_scale_net(inputs)
        num_channels = result.shape[self._CHANNEL_DIM]
        assert num_channels % 2 == 0
        return {
            "shift": result[:, :num_channels//2],
            "log-scale": result[:, num_channels//2:]
        }
