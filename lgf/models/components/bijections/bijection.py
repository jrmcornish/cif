import numpy as np

import torch
import torch.nn as nn


class Bijection(nn.Module):
    def __init__(self, x_shape, z_shape):
        super().__init__()
        self.x_shape = x_shape
        self.z_shape = z_shape

    def forward(self, inputs, direction, **kwargs):
        if direction == "x-to-z":
            assert inputs.shape[1:] == self.x_shape
            result = self._x_to_z(inputs, **kwargs)
            assert result["z"].shape[1:] == self.z_shape
            return result

        elif direction == "z-to-x":
            assert inputs.shape[1:] == self.z_shape
            result = self._z_to_x(inputs, **kwargs)
            assert result["x"].shape[1:] == self.x_shape
            return result

        else:
            assert False, f"Invalid direction {direction}"

    def x_to_z(self, x, **kwargs):
        return self(x, "x-to-z", **kwargs)

    def z_to_x(self, z, **kwargs):
        return self(z, "z-to-x", **kwargs)

    def inverse(self):
        return InverseBijection(self)

    # TODO: This is definitely not the best way to do things
    def condition(self, u):
        return ConditionedBijection(bijection=self, u=u)

    def _x_to_z(self, x, **kwargs):
        raise NotImplementedError

    def _z_to_x(self, z, **kwargs):
        raise NotImplementedError


class ConditionedBijection(Bijection):
    def __init__(self, bijection, u):
        super().__init__(x_shape=bijection.x_shape, z_shape=bijection.z_shape)

        self.bijection = bijection
        self.register_buffer("u", u)

    def _x_to_z(self, x, **kwargs):
        return self.bijection.x_to_z(x, u=self._expand_u(x))

    def _z_to_x(self, z, **kwargs):
        return self.bijection.z_to_x(z, u=self._expand_u(z))

    def _expand_u(self, inputs):
        return self.u.unsqueeze(0).expand(inputs.shape[0], *[-1 for _ in self.u.shape])


class InverseBijection(Bijection):
    def __init__(self, bijection):
        super().__init__(x_shape=bijection.z_shape, z_shape=bijection.x_shape)
        self.bijection = bijection

    def _x_to_z(self, x, **kwargs):
        result = self.bijection.z_to_x(x, **kwargs)
        z = result.pop("x")
        return {"z": z, **result}

    def _z_to_x(self, z, **kwargs):
        result = self.bijection.x_to_z(z, **kwargs)
        x = result.pop("z")
        return {"x": x, **result}


class IdentityBijection(Bijection):
    def __init__(self, x_shape):
        super().__init__(x_shape=x_shape, z_shape=x_shape)

    def _x_to_z(self, x, **kwargs):
        return {"z": x, "log-jac": self._log_jac_like(x)}

    def _z_to_x(self, z, **kwargs):
        return {"x": z, "log-jac": self._log_jac_like(z)}

    def _log_jac_like(self, inputs):
        return torch.zeros(inputs.shape[0], 1, dtype=inputs.dtype, device=inputs.device)


class CompositeBijection(Bijection):
    def __init__(self, layers, direction):
        if direction == "z-to-x":
            x_shape = layers[-1].x_shape
            z_shape = layers[0].z_shape

        elif direction == "x-to-z":
            x_shape = layers[0].x_shape
            z_shape = layers[-1].z_shape

        else:
            assert False, f"Invalid direction {direction}"

        super().__init__(x_shape, z_shape)

        if direction == "z-to-x":
            layers = reversed(layers)

        self._x_to_z_layers = nn.ModuleList(layers)

    def _x_to_z(self, x, **kwargs):
        z, log_jac = self._pass_through(x, "x-to-z", **kwargs)
        return {"z": z, "log-jac": log_jac}

    def _z_to_x(self, z, **kwargs):
        x, log_jac = self._pass_through(z, "z-to-x", **kwargs)
        return {"x": x, "log-jac": log_jac}

    def _pass_through(self, inputs, direction, **kwargs):
        assert direction in ["z-to-x", "x-to-z"]

        if direction == "x-to-z":
            output_name = "z"
            layer_order = self._x_to_z_layers
        else:
            output_name = "x"
            layer_order = reversed(self._x_to_z_layers)

        outputs = inputs
        log_jac = None
        for layer in layer_order:
            result = layer(outputs, direction, **kwargs)
            outputs = result[output_name]
            if log_jac is None:
                log_jac = result["log-jac"]
            else:
                log_jac += result["log-jac"]

        return outputs, log_jac
