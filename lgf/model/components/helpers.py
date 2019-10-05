import torch
import torch.nn as nn


class ScaledTanh2dModule(nn.Module):
    def __init__(self, module, num_channels):
        super().__init__()
        self.module = module
        self.weights = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, inputs):
        out = self.module(inputs)
        out = self.weights * torch.tanh(out)
        return out


class ConstantMap(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("value", value)

    def forward(self, inputs):
        return self.value.expand(inputs.shape[0], *self.value.shape)


class SplittingModule(nn.Module):
    def __init__(self, module, output_names, dim):
        super().__init__()
        assert output_names
        self.module = module
        self.output_names = output_names
        self.dim = dim

    def forward(self, *args, **kwargs):
        outputs = self.module(*args, **kwargs)
        assert outputs.shape[self.dim] % len(self.output_names) == 0
        chunked_outputs = outputs.chunk(len(self.output_names), dim=self.dim)
        return dict(zip(self.output_names, chunked_outputs))


class JoiningModule(nn.Module):
    def __init__(self, modules, output_names):
        super().__init__()
        assert len(modules) == len(output_names)
        self.module_list = nn.ModuleList(modules)
        self.output_names = output_names

    def forward(self, *args, **kwargs):
        outputs = [m(*args, **kwargs) for m in self.module_list]
        return dict(zip(self.output_names, outputs))
