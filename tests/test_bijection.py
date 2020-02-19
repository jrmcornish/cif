import unittest
from unittest.mock import Mock, MagicMock
import random
import warnings

import numpy as np

import torch
import torch.nn as nn

from cif.models.components.bijections import (
    CompositeBijection,
    FlipBijection,
    ViewBijection,
    ScalarAdditionBijection,
    ScalarMultiplicationBijection,
    LogitBijection,
    MADEBijection,
    BatchNormBijection,
    BruteForceInvertible1x1ConvBijection,
    LUInvertible1x1ConvBijection,
    Checkerboard2dAffineCouplingBijection,
    SplitChannelwiseAffineCouplingBijection,
    AlternatingChannelwiseAffineCouplingBijection,
    MaskedChannelwiseAffineCouplingBijection,
    Squeeze2dBijection,
    TanhBijection,
    RandomChannelwisePermutationBijection,
    FFJORDBijection,
    PlanarBijection,
    ResidualFlowBijection
)
from cif.models.components.networks import get_lipschitz_mlp
from cif.models.factory import get_coupler


class _TestBijection:
    def test_z_to_x_correct_output_format(self):
        z = torch.randn(self.batch_size, *self.bijection.z_shape)

        result = self._z_to_x(z, u=self._get_u())
        x = result["x"]
        log_jac = result["log-jac"]

        self.assertEqual(x.shape, (self.batch_size, *self.bijection.x_shape))
        self.assertEqual(log_jac.shape, (self.batch_size, 1))

    def test_x_to_z_correct_output_format(self):
        x = torch.randn(self.batch_size, *self.bijection.x_shape)

        result = self._x_to_z(x, u=self._get_u())
        z = result["z"]
        log_jac = result["log-jac"]

        self.assertEqual(z.shape, (self.batch_size, *self.bijection.z_shape))
        self.assertEqual(log_jac.shape, (self.batch_size, 1))

    def test_x_to_z_invertible(self):
        x = torch.randn(self.batch_size, *self.bijection.x_shape)
        u = self._get_u()
        z = self._x_to_z(x, u=u)["z"]
        x_alt = self._z_to_x(z, u=u)["x"]

        diffs = (x - x_alt).norm(dim=-1)
        self.assertLess(diffs.max().item(), self.eps)

    def test_z_to_x_invertible(self):
        z = torch.randn(self.batch_size, *self.bijection.z_shape)
        u = self._get_u()
        x = self._z_to_x(z, u=u)["x"]
        z_alt = self._x_to_z(x, u=u)["z"]

        diffs = (z - z_alt).norm(dim=-1)
        self.assertLess(diffs.max().item(), self.eps)

    def test_x_to_z_no_nans(self):
        x = torch.randn(self.batch_size, *self.bijection.x_shape)
        u = self._get_u()
        result = self._x_to_z(x, u=u)
        self.assertTrue(torch.all(torch.isfinite(result["z"])))
        self.assertTrue(torch.all(torch.isfinite(result["log-jac"])))

    def test_z_to_x_no_nans(self):
        z = torch.randn(self.batch_size, *self.bijection.z_shape)
        u = self._get_u()
        result = self._z_to_x(z, u=u)
        self.assertTrue(torch.all(torch.isfinite(result["x"])))
        self.assertTrue(torch.all(torch.isfinite(result["log-jac"])))

    def _x_to_z(self, x, u):
        if u is None:
            return self.bijection.x_to_z(x)
        else:
            return self.bijection.x_to_z(x, u=u)

    def _z_to_x(self, z, u):
        if u is None:
            return self.bijection.z_to_x(z)
        else:
            return self.bijection.z_to_x(z, u=u)

    def _get_u(self):
        if self.u_shape is None:
            return None
        else:
            return torch.randn(self.batch_size, *self.u_shape)


class TestMADEBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        self.batch_size = 1000
        self.u_shape = None
        self.eps = 1e-6
        self.bijection = MADEBijection(
            num_input_channels=5,
            hidden_channels=[11, 12, 10, 14],
            activation=nn.Tanh
        )

    def test_single_input_fails(self):
        with self.assertRaises(Exception):
            MADEBijection(
                num_input_channels=1,
                hidden_channels=[10],
                activation=nn.Tanh
            )


class TestCompositeBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        self.batch_size = 100
        self.eps = 1e-5

        x_shape = (5, 4)
        x_dim = np.prod(x_shape)
        u_dim = 3
        self.u_shape = (u_dim,)

        layers = [ViewBijection(x_shape, (x_dim,))]
        for i in range(5):
            made = MADEBijection(
                num_input_channels=x_dim,
                hidden_channels=[40, 40],
                activation=nn.Tanh
            )
            layers.append(made)
        layers.append(ViewBijection((x_dim,), x_shape))

        self.bijection = CompositeBijection(layers, "x-to-z")


class TestFlipBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        self.batch_size = 100
        self.eps = 1e-6
        self.bijection = FlipBijection(x_shape=(5,), dim=1)
        self.u_shape = None

    def test_flipped(self):
        inputs = torch.randn(self.batch_size, *self.bijection.x_shape)

        z = self.bijection.z_to_x(inputs)["x"]
        self.assert_flipped(inputs, z)

        x = self.bijection.x_to_z(inputs)["z"]
        self.assert_flipped(inputs, x)

    def assert_flipped(self, inputs, outputs):
        _, inputs_dim = inputs.size()
        _, outputs_dim = outputs.size()
        self.assertEqual(inputs_dim, outputs_dim)
        for i, j in zip(range(inputs_dim), reversed(range(outputs_dim))):
            self.assertTrue((inputs[:, i] == outputs[:, j]).all())

    def test_log_jacs(self):
        inputs = torch.randn(self.batch_size, *self.bijection.x_shape)

        log_jac = self.bijection.x_to_z(inputs)["log-jac"]
        self.assertTrue((log_jac == 0).all())

        log_jac = self.bijection.z_to_x(inputs)["log-jac"]
        self.assertTrue((log_jac == 0).all())


class TestViewBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        self.batch_size = 100
        self.u_shape = None
        self.eps = 1e-7
        self.bijection = ViewBijection((10, 4), (20, 2))


class TestBatchNormBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        self.batch_size = 100
        self.eps = 1e-5

        self.bijection = BatchNormBijection(
            x_shape=(10, 2),
            per_channel=True,
            apply_affine=True,
            momentum=0.1
        )
        self.u_shape = None

        # XXX: We have to do this because otherwise composing z_to_x with x_to_z won't be invertible 
        # (because when we call x_to_z, we change the function). Ideally we could test both instances.
        self.bijection.eval()


def get_channelwise_masked_acl_test(bijection_class, u_dim):
    class Test(_TestBijection, unittest.TestCase):
        def setUp(self):
            self.batch_size = 100
            self.eps = 1e-6

            self.u_shape = (u_dim,) if u_dim > 0 else None

            num_x_channels = 21

            def coupler_factory(num_passthrough_channels):
                return get_coupler(
                    input_shape=(num_passthrough_channels + u_dim,),
                    num_channels_per_output=num_x_channels - num_passthrough_channels,
                    config={
                        "independent_nets": False,
                        "shift_log_scale_net": {
                            "type": "mlp",
                            "hidden_channels": [10, 20],
                            "activation": "tanh"
                        }
                    }
                )

            self.bijection = bijection_class(
                x_shape=(num_x_channels,),
                coupler_factory=coupler_factory,
                reverse_mask=True
            )

    return Test


TestConditionalChannelwiseMaskedAffineCouplingBijection = get_channelwise_masked_acl_test(bijection_class=SplitChannelwiseAffineCouplingBijection, u_dim=0)
TestUnconditionalChannelwiseMaskedAffineCouplingBijection = get_channelwise_masked_acl_test(bijection_class=AlternatingChannelwiseAffineCouplingBijection, u_dim=10)


def get_checkerboard_acl_test(num_u_channels):
    class Test(_TestBijection, unittest.TestCase):
        def setUp(self):
            self.batch_size = 100
            self.eps = 1e-6

            self.u_shape = (num_u_channels, 28, 28) if num_u_channels > 0 else None
            num_x_channels = 1
            x_shape = (num_x_channels, 28, 28)
            self.bijection = Checkerboard2dAffineCouplingBijection(
                x_shape=x_shape,
                coupler=get_coupler(
                    input_shape=(num_x_channels + num_u_channels, *x_shape[1:]),
                    num_channels_per_output=num_x_channels,
                    config={
                        "independent_nets": False,
                        "shift_log_scale_net": {
                            "type": "resnet",
                            "hidden_channels": [24, 24]
                        }
                    }
                ),
                reverse_mask=True
            )

    return Test


TestConditionalCheckerboard2dAffineCouplingBijection = get_checkerboard_acl_test(num_u_channels=0)
TestUnconditionalCheckerboard2dAffineCouplingBijection = get_checkerboard_acl_test(num_u_channels=10)


class TestLogitBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        self.batch_size = 100
        self.u_shape = None
        self.eps = 1e-5

        x_shape = (10, 4)
        self.bijection = CompositeBijection([
            # Necessary to have these because we get nans otherwise for logit
            ScalarMultiplicationBijection(x_shape=x_shape, value=0.002),
            ScalarAdditionBijection(x_shape=x_shape, value=0.02),
            LogitBijection(x_shape=x_shape),
        ], "x-to-z")


class TestSqueeze2dBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        self.x_shape = (1, 4, 4)
        self.z_shape = (4, 2, 2)
        self.u_shape = None
        self.batch_size = 100
        self.eps = 1e-7
        self.bijection = Squeeze2dBijection(x_shape=self.x_shape, factor=2)

    def test_realnvp_paper_example(self):
        x = torch.tensor([
            [1, 2, 5, 6],
            [3, 4, 7, 8],
            [9, 10, 13, 14],
            [11, 12, 15, 16]
        ], dtype=torch.get_default_dtype())

        x = x.view(1, 1, 4, 4)

        z = self.bijection.x_to_z(x)["z"]

        self.assertEqual(z.shape, (1, 4, 2, 2))

        z = z.squeeze(0)

        self.assertTrue((z[0] == torch.tensor([
            [1., 5.],
            [9., 13.]
        ])).all())

        self.assertTrue((z[1] == torch.tensor([
            [2., 6.],
            [10., 14.]
        ])).all())

        self.assertTrue((z[2] == torch.tensor([
            [3., 7.],
            [11., 15.]
        ])).all())

        self.assertTrue((z[3] == torch.tensor([
            [4., 8.],
            [12., 16.]
        ])).all())

    def test_extension_of_realnvp_paper_example(self):
        x = torch.tensor([[
            [1, 2, 5, 6],
            [3, 4, 7, 8],
            [9, 10, 13, 14],
            [11, 12, 15, 16]
        ], [
            [17, 18, 21, 22],
            [19, 20, 23, 24],
            [25, 26, 29, 30],
            [27, 28, 31, 32]
        ]], dtype=torch.get_default_dtype())

        x = x.view(1, 2, 4, 4)

        bijection = Squeeze2dBijection(x_shape=x.shape[1:], factor=2)

        z = bijection.x_to_z(x)["z"]

        self.assertEqual(z.shape, (1, 8, 2, 2))

        z = z.squeeze(0)

        result1 = torch.tensor([1., 2., 3., 4., 17., 18., 19., 20.])
        self.assertTrue((z[:, 0, 0] == result1).all())

        result2 = torch.tensor([5., 6., 7., 8., 21., 22., 23., 24.])
        self.assertTrue((z[:, 0, 1] == result2).all())

        result3 = torch.tensor([9., 10., 11., 12., 25., 26., 27., 28.])
        self.assertTrue((z[:, 1, 0] == result3).all())

        result4 = torch.tensor([13., 14., 15., 16., 29., 30., 31., 32.])
        self.assertTrue((z[:, 1, 1] == result4).all())


class TestUnconditionalBruteForceInvertible1x1ConvBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        x_shape = (1, 4, 4)
        self.batch_size = 1000
        self.u_shape = None
        self.eps = 1e-6
        self.bijection = BruteForceInvertible1x1ConvBijection(
            x_shape=x_shape
        )


class TestConditionalBruteForceInvertible1x1ConvBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        x_shape = (1, 4, 4)
        self.batch_size = 1000
        self.u_shape = (2, 4, 4)
        self.eps = 1e-6
        self.bijection = BruteForceInvertible1x1ConvBijection(
            x_shape=x_shape,
            num_u_channels=self.u_shape[0]
        )


class TestUnconditionalLUInvertible1x1ConvBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        x_shape = (1, 4, 4)
        self.batch_size = 1000
        self.u_shape = None
        self.eps = 1e-6
        self.bijection = LUInvertible1x1ConvBijection(
            x_shape=x_shape
        )


class TestConditionalLUInvertible1x1ConvBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        x_shape = (1, 4, 4)
        self.batch_size = 1000
        self.u_shape = (2, 4, 4)
        self.eps = 1e-6
        self.bijection = LUInvertible1x1ConvBijection(
            x_shape=x_shape,
            num_u_channels=self.u_shape[0]
        )


class TestNonImageConditionalLUInvertible1x1ConvBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        x_shape = (2,)
        self.batch_size = 1000
        self.u_shape = (1,)
        self.eps = 1e-6
        self.bijection = LUInvertible1x1ConvBijection(
            x_shape=x_shape,
            num_u_channels=self.u_shape[0]
        )


class TestUnconditionalInvertible1x1ConvBijectionEquality(_TestBijection, unittest.TestCase):
    def setUp(self):
        x_shape = (3, 4, 4)
        self.batch_size = 1000
        self.u_shape = None
        self.eps = 5e-4
        self.jac_eps = 5e-4
        self.bijection = LUInvertible1x1ConvBijection(
            x_shape=x_shape
        )

        # Sample new s to ensure non-trivial Jacobian comparison
        new_s = torch.randn(x_shape[0])
        self.bijection.sign_s = torch.sign(new_s)
        self.bijection.log_s = nn.Parameter(torch.log(torch.abs(new_s)))

        self.bijection_to_compare = BruteForceInvertible1x1ConvBijection(
            x_shape=x_shape
        )
        self.bijection_to_compare.weights = nn.Parameter(self.bijection._get_weights())

    def test_LU_gives_same_result_forward(self):
        x = torch.randn(self.batch_size, *self.bijection.x_shape)
        z_LU, log_jac_LU  = self.bijection.x_to_z(x).values()
        z_non_LU, log_jac_non_LU =  self.bijection_to_compare.x_to_z(x).values()

        diffs_z = (z_LU - z_non_LU).norm(dim=-1)
        self.assertLess(diffs_z.max().item(), self.eps)

        diffs_jac = (log_jac_LU - log_jac_non_LU).norm(dim=-1)
        self.assertLess(diffs_jac.max().item(), self.jac_eps)

    def test_LU_gives_same_result_backward(self):
        z = torch.randn(self.batch_size, *self.bijection.x_shape)
        x_LU, log_jac_LU  = self.bijection.z_to_x(z).values()
        x_non_LU, log_jac_non_LU =  self.bijection_to_compare.z_to_x(z).values()

        diffs_x = (x_LU - x_non_LU).norm(dim=-1)
        self.assertLess(diffs_x.max().item(), self.eps)

        diffs_jac = (log_jac_LU - log_jac_non_LU).norm(dim=-1)
        self.assertLess(diffs_jac.max().item(), self.jac_eps)


class TestScalarMultiplicationBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        x_shape = (50, 4)
        self.batch_size = 1000
        self.u_shape = None
        self.eps = 1e-6
        self.bijection = ScalarMultiplicationBijection(
            x_shape=x_shape,
            value=5.3
        )


class TestScalarAdditionBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        x_shape = (50, 4)
        self.batch_size = 1000
        self.u_shape = None
        self.eps = 1e-6
        self.bijection = ScalarAdditionBijection(
            x_shape=x_shape,
            value=5.3
        )


class TestRandomChannelwisePermutationBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        self.batch_size = 1000
        self.u_shape = None
        self.eps = 1e-6
        self.bijection = RandomChannelwisePermutationBijection(x_shape=(40, 32, 1))


class TestResidualFlowBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        self.batch_size = 1000
        self.u_shape = None
        self.eps = 1e-4
        num_input_channels = 4
        self.bijection = ResidualFlowBijection(
            x_shape=(num_input_channels,),
            lipschitz_net=get_lipschitz_mlp(
                num_input_channels=num_input_channels,
                hidden_channels=[20, 30],
                num_output_channels=4,
                lipschitz_constant=0.9,
                max_train_lipschitz_iters=None,
                max_eval_lipschitz_iters=None,
                lipschitz_tolerance=1e-3
            ),
            reduce_memory=True
        )


class TestFFJORDBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        self.batch_size = 1000
        self.u_shape = None
        self.eps = 1e-4
        self.bijection = FFJORDBijection(
            x_shape=(10,),
            velocity_hidden_channels=[20]*2,
            num_u_channels=0,
            relative_tolerance=1e-5,
            absolute_tolerance=1e-5
        )


class TestFFJORDConditionalBijection(_TestBijection, unittest.TestCase):
    def setUp(self):
        self.batch_size = 1000
        self.u_shape = (1,)
        self.eps = 1e-4
        self.bijection = FFJORDBijection(
            x_shape=(10,),
            velocity_hidden_channels=[20]*4,
            num_u_channels=self.u_shape[0],
            relative_tolerance=1e-5,
            absolute_tolerance=1e-5
        )


if __name__ == "__main__":
    unittest.main()
