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
    IdentityTransport,
    LinearTransport,
    MLPTransport,
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


class TestMLPTransportPlaceholder(unittest.TestCase):
    def test_raises(self):
        with self.assertRaises(NotImplementedError):
            MLPTransport()


class TestBuildTransport(unittest.TestCase):
    def test_identity(self):
        self.assertIsInstance(build_transport("identity"), IdentityTransport)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            build_transport("bogus")

    def test_mlp_raises(self):
        with self.assertRaises(NotImplementedError):
            build_transport("mlp")


if __name__ == "__main__":
    unittest.main()
