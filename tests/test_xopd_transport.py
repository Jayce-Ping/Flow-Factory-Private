# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for XOPD cross-VAE latent transport (pure tensor math, CPU only)."""

import unittest

import torch

from flow_factory.trainers.xopd.transport import (
    AdaLNTransport,
    IdentityTransport,
    LinearTransport,
    MLPTransport,
    WhiteningTransport,
    build_transport,
    channel_affine,
    fit_channel_affine_lstsq,
    moment_matching_affine,
    resample_spatial,
)


def _bchw_identity(z):
    """to/from_spatial for an adapter whose native layout is already BCHW."""
    return z


class TestResampleSpatial(unittest.TestCase):
    def test_noop_when_same_size(self):
        x = torch.randn(2, 4, 8, 8)
        y = resample_spatial(x, (8, 8))
        self.assertIs(y, x)

    def test_resizes(self):
        x = torch.randn(2, 4, 8, 8)
        y = resample_spatial(x, (16, 16))
        self.assertEqual(tuple(y.shape), (2, 4, 16, 16))


class TestChannelAffine(unittest.TestCase):
    def test_matches_manual(self):
        x = torch.randn(2, 3, 5, 5)
        A = torch.randn(4, 3)
        b = torch.randn(4)
        y = channel_affine(x, A, b)
        self.assertEqual(tuple(y.shape), (2, 4, 5, 5))
        # spot-check one position
        manual = A @ x[0, :, 2, 3] + b
        torch.testing.assert_close(y[0, :, 2, 3], manual)


class TestFitChannelAffine(unittest.TestCase):
    def test_recovers_known_affine(self):
        torch.manual_seed(0)
        C_in, C_out = 6, 4
        A_true = torch.randn(C_out, C_in)
        b_true = torch.randn(C_out)
        z_in = torch.randn(8, C_in, 4, 4)
        z_out = channel_affine(z_in, A_true, b_true)
        A, b = fit_channel_affine_lstsq(z_in, z_out, ridge=0.0)
        torch.testing.assert_close(A, A_true, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(b, b_true, atol=1e-3, rtol=1e-3)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            fit_channel_affine_lstsq(torch.randn(2, 3, 4, 4), torch.randn(2, 5, 8, 8))


class TestMomentMatching(unittest.TestCase):
    def test_aligns_per_channel_moments(self):
        torch.manual_seed(0)
        C = 4
        z_in = torch.randn(16, C, 8, 8)
        # target: per-channel scaled+shifted
        scale = torch.tensor([2.0, 0.5, 3.0, 1.0]).view(1, C, 1, 1)
        shift = torch.tensor([1.0, -1.0, 0.0, 5.0]).view(1, C, 1, 1)
        z_out = z_in * scale + shift
        A, b = moment_matching_affine(z_in, z_out)
        z_hat = channel_affine(z_in, A, b)
        # per-channel mean/std should match
        torch.testing.assert_close(
            z_hat.mean(dim=(0, 2, 3)), z_out.mean(dim=(0, 2, 3)), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            z_hat.std(dim=(0, 2, 3)), z_out.std(dim=(0, 2, 3)), atol=1e-2, rtol=1e-2
        )


class TestIdentityTransport(unittest.TestCase):
    def test_sample_and_mean_passthrough(self):
        t = IdentityTransport()
        z = torch.randn(2, 4, 8, 8)
        torch.testing.assert_close(t.transport_sample(z), z)
        called = {}

        def q(x):
            called["x"] = x
            return x * 2

        out = t.transition_mean_to_student(z, q)
        torch.testing.assert_close(out, z * 2)
        torch.testing.assert_close(called["x"], z)


class TestLinearTransport(unittest.TestCase):
    def _make(self):
        return LinearTransport(
            teacher_to_spatial=_bchw_identity,
            teacher_from_spatial=_bchw_identity,
            student_to_spatial=_bchw_identity,
            student_from_spatial=_bchw_identity,
            ridge=0.0,
        )

    def test_requires_fit(self):
        t = self._make()
        self.assertTrue(t.requires_warmup)
        self.assertFalse(t.is_fitted)
        with self.assertRaises(RuntimeError):
            t.transport_sample(torch.randn(1, 6, 8, 8))

    def test_fit_and_transport_same_grid(self):
        torch.manual_seed(0)
        C_T, C_S = 6, 4
        A_true = torch.randn(C_S, C_T)
        b_true = torch.randn(C_S)
        z_T = [torch.randn(4, C_T, 8, 8) for _ in range(3)]
        z_S = [channel_affine(z, A_true, b_true) for z in z_T]
        t = self._make()
        t.fit(z_T, z_S)
        self.assertTrue(t.is_fitted)
        # transport recovers the known affine
        probe = torch.randn(2, C_T, 8, 8)
        out = t.transport_sample(probe)
        torch.testing.assert_close(out, channel_affine(probe, A_true, b_true),
                                   atol=1e-3, rtol=1e-3)

    def test_transition_mean_exact_for_affine_teacher(self):
        # If teacher mean is identity in T space, mapping T->S and back is exact.
        torch.manual_seed(1)
        C_T, C_S = 4, 4
        A_true = torch.randn(C_S, C_T)
        b_true = torch.randn(C_S)
        z_T = [torch.randn(8, C_T, 6, 6) for _ in range(4)]
        z_S = [channel_affine(z, A_true, b_true) for z in z_T]
        t = self._make()
        t.fit(z_T, z_S)
        x_S = torch.randn(2, C_S, 6, 6)
        # teacher mean = identity -> transported mean should be ~ x_S (T(T^-1(x))=x)
        out = t.transition_mean_to_student(x_S, query_teacher_mean=lambda x_T: x_T)
        torch.testing.assert_close(out, x_S, atol=1e-3, rtol=1e-3)


class TestWhiteningTransport(unittest.TestCase):
    def _make(self):
        return WhiteningTransport(
            teacher_to_spatial=_bchw_identity,
            teacher_from_spatial=_bchw_identity,
            student_to_spatial=_bchw_identity,
            student_from_spatial=_bchw_identity,
        )

    def test_diagonal_moment_match_and_inverse(self):
        torch.manual_seed(0)
        C = 4
        scale = torch.tensor([2.0, 0.5, 3.0, 1.0]).view(1, C, 1, 1)
        shift = torch.tensor([1.0, -1.0, 0.0, 5.0]).view(1, C, 1, 1)
        z_T = [torch.randn(8, C, 6, 6) for _ in range(3)]
        z_S = [z * scale + shift for z in z_T]
        t = self._make()
        t.fit(z_T, z_S)
        # transported teacher latent should match per-channel moments of z_S
        out = t.transport_sample(z_T[0])
        torch.testing.assert_close(
            out.mean(dim=(0, 2, 3)), z_S[0].mean(dim=(0, 2, 3)), atol=1e-2, rtol=1e-2
        )
        # identity teacher mean -> exact round trip (analytic inverse)
        x_S = torch.randn(2, C, 6, 6)
        rt = t.transition_mean_to_student(x_S, query_teacher_mean=lambda x_T: x_T)
        torch.testing.assert_close(rt, x_S, atol=1e-3, rtol=1e-3)

    def test_identity_when_spaces_coincide(self):
        # Same space (z_S == z_T) -> moment matching gives gamma=1, beta=0.
        torch.manual_seed(1)
        C = 4
        z = [torch.randn(8, C, 5, 5) for _ in range(2)]
        t = self._make()
        t.fit(z, [zz.clone() for zz in z])
        probe = torch.randn(2, C, 5, 5)
        out = t.transport_sample(probe)
        torch.testing.assert_close(out, probe, atol=1e-2, rtol=1e-2)


class TestAdaLNTransport(unittest.TestCase):
    def _make(self, C_T=4, C_S=4):
        return AdaLNTransport(
            teacher_to_spatial=_bchw_identity,
            teacher_from_spatial=_bchw_identity,
            student_to_spatial=_bchw_identity,
            student_from_spatial=_bchw_identity,
            teacher_channels=C_T,
            student_channels=C_S,
        )

    def test_is_nn_module_with_params(self):
        import torch.nn as nn

        t = self._make()
        self.assertIsInstance(t, nn.Module)
        names = {n for n, _ in t.named_parameters()}
        self.assertEqual(names, {"log_gamma", "beta", "W"})
        self.assertTrue(t.requires_warmup)

    def test_default_is_identity(self):
        # log_gamma=0, beta=0 -> identity before any init.
        t = self._make()
        z = torch.randn(2, 4, 6, 6)
        torch.testing.assert_close(t.transport_sample(z), z, atol=1e-5, rtol=1e-5)

    def test_moment_init_and_analytic_inverse(self):
        torch.manual_seed(0)
        C = 4
        scale = torch.tensor([2.0, 0.5, 3.0, 1.0]).view(1, C, 1, 1)
        shift = torch.tensor([1.0, -1.0, 0.0, 5.0]).view(1, C, 1, 1)
        z_T = [torch.randn(8, C, 6, 6) for _ in range(3)]
        z_S = [z * scale + shift for z in z_T]
        t = self._make()
        t.fit(z_T, z_S)
        self.assertTrue(t.is_fitted)
        out = t.transport_sample(z_T[0])
        torch.testing.assert_close(
            out.mean(dim=(0, 2, 3)), z_S[0].mean(dim=(0, 2, 3)), atol=1e-2, rtol=1e-2
        )
        # analytic inverse round trip (identity teacher mean)
        x_S = torch.randn(2, C, 6, 6)
        rt = t.transition_mean_to_student(x_S, query_teacher_mean=lambda x_T: x_T)
        torch.testing.assert_close(rt, x_S, atol=1e-4, rtol=1e-4)

    def test_gradients_flow_to_params(self):
        t = self._make()
        z = torch.randn(2, 4, 5, 5)
        out = t.transport_sample(z)
        out.pow(2).mean().backward()
        self.assertIsNotNone(t.log_gamma.grad)
        self.assertIsNotNone(t.beta.grad)

    def test_channel_mismatch(self):
        t = self._make(C_T=6, C_S=4)
        z_T = torch.randn(2, 6, 5, 5)
        out = t.transport_sample(z_T)
        self.assertEqual(tuple(out.shape), (2, 4, 5, 5))


class TestMLPTransportPlaceholder(unittest.TestCase):
    def test_raises(self):
        with self.assertRaises(NotImplementedError):
            MLPTransport()


class TestBuildTransport(unittest.TestCase):
    def test_identity(self):
        self.assertIsInstance(build_transport("identity"), IdentityTransport)

    def test_whitening(self):
        t = build_transport(
            "whitening",
            teacher_to_spatial=_bchw_identity,
            teacher_from_spatial=_bchw_identity,
            student_to_spatial=_bchw_identity,
            student_from_spatial=_bchw_identity,
        )
        self.assertIsInstance(t, WhiteningTransport)
        self.assertTrue(t.requires_warmup)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            build_transport("bogus")

    def test_mlp_raises(self):
        with self.assertRaises(NotImplementedError):
            build_transport("mlp")


class TestTransportStateDict(unittest.TestCase):
    def _affine_converters(self):
        return dict(
            teacher_to_spatial=_bchw_identity,
            teacher_from_spatial=_bchw_identity,
            student_to_spatial=_bchw_identity,
            student_from_spatial=_bchw_identity,
        )

    def test_linear_roundtrip(self):
        torch.manual_seed(0)
        z_T = [torch.randn(4, 4, 6, 6) for _ in range(2)]
        z_S = [channel_affine(z, torch.randn(4, 4), torch.randn(4)) for z in z_T]
        t = LinearTransport(**self._affine_converters(), ridge=0.0)
        t.fit(z_T, z_S)
        sd = t.state_dict()
        t2 = LinearTransport(**self._affine_converters())
        t2.load_state_dict(sd)
        probe = torch.randn(2, 4, 6, 6)
        torch.testing.assert_close(t2.transport_sample(probe), t.transport_sample(probe))

    def test_adaln_roundtrip(self):
        torch.manual_seed(0)
        z_T = [torch.randn(4, 4, 6, 6) for _ in range(2)]
        z_S = [z * 2.0 + 1.0 for z in z_T]
        t = AdaLNTransport(**self._affine_converters(), teacher_channels=4, student_channels=4)
        t.fit(z_T, z_S)
        sd = t.state_dict()
        self.assertIn("_student_grid", sd)
        t2 = AdaLNTransport(**self._affine_converters(), teacher_channels=4, student_channels=4)
        t2.load_state_dict(sd)
        self.assertTrue(t2.is_fitted)
        probe = torch.randn(2, 4, 6, 6)
        torch.testing.assert_close(t2.transport_sample(probe), t.transport_sample(probe))


if __name__ == "__main__":
    unittest.main()
