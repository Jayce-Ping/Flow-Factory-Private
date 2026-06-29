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
    ConvTransport,
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
        # Only the adaLN-Zero modulation MLP is a gradient parameter; the base
        # affine (A_base, b_base) is a frozen closed-form buffer.
        names = {n for n, _ in t.named_parameters()}
        self.assertEqual(
            names,
            {
                "mod_mlp.0.weight",
                "mod_mlp.0.bias",
                "mod_mlp.2.weight",
                "mod_mlp.2.bias",
            },
        )
        buf_names = {n for n, _ in t.named_buffers()}
        self.assertIn("A_base", buf_names)
        self.assertIn("b_base", buf_names)
        self.assertTrue(t.requires_warmup)

    def test_default_is_identity(self):
        # A_base=identity-selection, b_base=0, modulation zero-init (gamma=1,shift=0)
        # -> the untrained transport is the identity, with or without a noise level.
        t = self._make()
        z = torch.randn(2, 4, 6, 6)
        torch.testing.assert_close(t.transport_sample(z), z, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            t.transport_sample(z, sigma=0.5), z, atol=1e-5, rtol=1e-5
        )

    def test_zero_init_modulation_is_neutral(self):
        # adaLN-Zero: zero-init last Linear -> modulation neutral at any t, so the
        # transport equals the closed-form base affine right after fit (do-no-harm).
        torch.manual_seed(0)
        C = 4
        z_T = [torch.randn(8, C, 6, 6) for _ in range(3)]
        z_S = [z * 2.0 + 1.0 for z in z_T]
        t = self._make()
        t.fit(z_T, z_S)
        with_sigma = t.transport_sample(z_T[0], sigma=0.75)
        no_sigma = t.transport_sample(z_T[0], sigma=None)
        torch.testing.assert_close(with_sigma, no_sigma, atol=1e-5, rtol=1e-5)

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
        # analytic inverse round trip (identity teacher mean), at a fixed noise level
        x_S = torch.randn(2, C, 6, 6)
        rt = t.transition_mean_to_student(
            x_S, query_teacher_mean=lambda x_T: x_T, sigma=0.3
        )
        torch.testing.assert_close(rt, x_S, atol=1e-4, rtol=1e-4)

    def test_gradients_flow_to_params(self):
        # After a forward at a real noise level, gradients reach the modulation MLP.
        t = self._make()
        z = torch.randn(2, 4, 5, 5)
        out = t.transport_sample(z, sigma=0.5)
        out.pow(2).mean().backward()
        self.assertIsNotNone(t.mod_mlp[0].weight.grad)
        # The zero-init last layer has a defined (possibly zero at step 0) grad slot
        # once it participates in the graph.
        self.assertIsNotNone(t.mod_mlp[-1].weight.grad)

    def test_sigma_conditioning_changes_modulation(self):
        # After a few online updates the modulation should differ across noise levels
        # (the whole point of the conditioning) — train it to need per-sigma correction.
        torch.manual_seed(0)
        C = 4
        t = self._make()
        # target depends on sigma: low -> scale 2, high -> scale 0.5 (toy per-sigma map).
        for _ in range(50):
            z_T = [torch.randn(8, C, 6, 6), torch.randn(8, C, 6, 6)]
            z_S = [z_T[0] * 2.0, z_T[1] * 0.5]
            t.set_online_lr(1e-2)
            t.update_online(z_T, z_S, sigma_list=[0.05, 0.95])
        probe = torch.randn(4, C, 6, 6)
        out_lo = t.transport_sample(probe, sigma=0.05)
        out_hi = t.transport_sample(probe, sigma=0.95)
        # The two noise levels must produce different outputs.
        self.assertGreater((out_lo - out_hi).abs().mean().item(), 1e-3)

    def test_two_phase_base_then_mod(self):
        # Phase 1 (update_base=True, update_mod=False) fits ONLY the base; phase 2
        # (update_base=False, update_mod=True) freezes the base and trains ONLY the MLP.
        torch.manual_seed(0)
        C = 4
        t = self._make()
        z_T = [torch.randn(8, C, 6, 6) for _ in range(2)]
        z_S = [z * 2.0 + 1.0 for z in z_T]
        t.set_online_lr(1e-2)
        # Phase 1: base only -> base fitted, modulation MLP untouched (last layer zero).
        t.update_online(z_T, z_S, sigma_list=[0.1, 0.9], update_base=True, update_mod=False)
        self.assertTrue(t.is_fitted)
        base_after_p1 = t.A_base.clone()
        mlp_last_after_p1 = t.mod_mlp[-1].weight.clone()
        torch.testing.assert_close(
            mlp_last_after_p1, torch.zeros_like(mlp_last_after_p1)
        )
        # Phase 2: modulation only -> base must stay frozen, MLP must change.
        for _ in range(10):
            t.update_online(
                z_T, z_S, sigma_list=[0.1, 0.9], update_base=False, update_mod=True
            )
        torch.testing.assert_close(t.A_base, base_after_p1, atol=0.0, rtol=0.0)
        self.assertGreater(
            (t.mod_mlp[-1].weight - mlp_last_after_p1).abs().max().item(), 0.0
        )

    def test_pinv_cache_invalidates_on_base_change(self):
        # The cached pinv(A_base) is lazily built on first inverse and invalidated
        # whenever the base is re-solved (so a stale inverse is never reused).
        torch.manual_seed(0)
        C = 4
        t = self._make()
        z_T = [torch.randn(8, C, 6, 6) for _ in range(2)]
        z_S = [z * 2.0 + 1.0 for z in z_T]
        t.fit(z_T, z_S)
        self.assertIsNone(t._A_base_pinv)
        _ = t.transition_mean_to_student(
            torch.randn(2, C, 6, 6), query_teacher_mean=lambda x: x, sigma=0.3
        )
        self.assertIsNotNone(t._A_base_pinv)  # built on first inverse
        # Re-solving the base invalidates the cache.
        t.update_online(z_T, z_S, sigma_list=[0.1, 0.9], update_base=True, update_mod=False)
        self.assertIsNone(t._A_base_pinv)

    def test_channel_mismatch(self):
        t = self._make(C_T=6, C_S=4)
        z_T = torch.randn(2, 6, 5, 5)
        out = t.transport_sample(z_T)
        self.assertEqual(tuple(out.shape), (2, 4, 5, 5))


class TestConvTransport(unittest.TestCase):
    """Strictly-linear conv transport: do-no-harm init, affine pushforward, recon."""

    def _make(self, C_T=8, C_S=4, hidden=16, n_layers=2):
        return ConvTransport(
            teacher_to_spatial=_bchw_identity,
            teacher_from_spatial=_bchw_identity,
            student_to_spatial=_bchw_identity,
            student_from_spatial=_bchw_identity,
            teacher_channels=C_T,
            student_channels=C_S,
            hidden_channels=hidden,
            n_layers=n_layers,
        )

    @staticmethod
    def _true_upsample_net(C_T, C_S, f, seed=0):
        """A fixed LINEAR conv->PixelShuffle map (channels carry the sub-pixel detail).

        This is exactly the regime the per-pixel affine cannot fit (bilinear upsample
        + per-position channel mix) but a linear conv residual CAN.
        """
        import torch.nn as nn

        g = torch.Generator().manual_seed(seed)
        net = nn.Sequential(nn.Conv2d(C_T, C_S * f * f, 3, padding=1), nn.PixelShuffle(f))
        for p in net.parameters():
            p.data = torch.randn(p.shape, generator=g)
            p.requires_grad_(False)
        return net

    def test_is_nn_module_with_residual_params(self):
        import torch.nn as nn

        t = self._make()
        self.assertIsInstance(t, nn.Module)
        self.assertTrue(t.requires_warmup)
        # Residual nets are lazily built on fit; before that there are no params.
        self.assertEqual(len(list(t.parameters())), 0)
        z_T = [torch.randn(6, 8, 4, 4) for _ in range(2)]
        z_S = [torch.randn(6, 4, 8, 8) for _ in range(2)]  # 2x upscale
        t.fit(z_T, z_S)
        names = {n for n, _ in t.named_parameters()}
        self.assertTrue(any(n.startswith("_fwd") for n in names))
        self.assertTrue(any(n.startswith("_inv") for n in names))
        buf_names = {n for n, _ in t.named_buffers()}
        self.assertIn("A_base", buf_names)
        self.assertIn("b_base", buf_names)
        self.assertEqual(t._upscale, 2)

    def test_do_no_harm_init_equals_base_affine(self):
        # Zero-init residual -> right after fit the transport == the closed-form base
        # affine (bilinear resample + channel affine), i.e. >= LinearTransport.
        torch.manual_seed(0)
        C_T, C_S, f = 8, 4, 2
        z_T = [torch.randn(6, C_T, 4, 4) for _ in range(3)]
        z_S = [torch.randn(6, C_S, 4 * f, 4 * f) for _ in range(3)]
        t = self._make(C_T=C_T, C_S=C_S)
        t.fit(z_T, z_S)
        probe = torch.randn(2, C_T, 4, 4)
        out = t.transport_sample(probe)
        base = channel_affine(
            resample_spatial(probe, t._student_grid), t.A_base, t.b_base
        )
        torch.testing.assert_close(out, base, atol=1e-5, rtol=1e-5)

    def test_linearity_affine_pushforward(self):
        # Even after training the residual, T is AFFINE: it preserves affine combos,
        # so E[T(z)] = T(E[z]) and the L1 transition-mean pushforward stays exact.
        torch.manual_seed(0)
        C_T, C_S, f = 8, 4, 2
        true_net = self._true_upsample_net(C_T, C_S, f)
        t = self._make(C_T=C_T, C_S=C_S)
        t.set_online_lr(1e-2)
        for _ in range(30):
            z_T = [torch.randn(6, C_T, 4, 4) for _ in range(2)]
            z_S = [true_net(z).detach() for z in z_T]
            t.update_online(z_T, z_S)
        z1 = torch.randn(2, C_T, 4, 4)
        z2 = torch.randn(2, C_T, 4, 4)
        a = 0.3
        lhs = t.transport_sample(a * z1 + (1 - a) * z2)
        rhs = a * t.transport_sample(z1) + (1 - a) * t.transport_sample(z2)
        torch.testing.assert_close(lhs, rhs, atol=1e-4, rtol=1e-4)

    def test_warmup_recon_beats_affine_floor(self):
        # On a channel->sub-pixel target (what the affine CANNOT fit), training the
        # linear conv residual drops the recon far below the base-affine floor.
        torch.manual_seed(0)
        C_T, C_S, f = 8, 4, 2
        true_net = self._true_upsample_net(C_T, C_S, f)
        t = self._make(C_T=C_T, C_S=C_S, hidden=32, n_layers=2)
        t.set_online_lr(1e-2)
        # Fixed eval batch.
        z_T_eval = torch.randn(8, C_T, 4, 4)
        z_S_eval = true_net(z_T_eval).detach()

        def recon():
            with torch.no_grad():
                return float(
                    (t.transport_sample(z_T_eval) - z_S_eval).pow(2).mean()
                )

        # Phase 1: base only -> the affine floor.
        z_T = [torch.randn(6, C_T, 4, 4) for _ in range(2)]
        z_S = [true_net(z).detach() for z in z_T]
        t.update_online(z_T, z_S, update_base=True, update_mod=False)
        base_floor = recon()
        # Phase 2: train the residual on frozen base.
        for _ in range(200):
            z_T = [torch.randn(6, C_T, 4, 4) for _ in range(2)]
            z_S = [true_net(z).detach() for z in z_T]
            t.update_online(z_T, z_S, update_base=False, update_mod=True)
        final = recon()
        self.assertLess(final, 0.3 * base_floor)

    def test_inverse_recon_improves(self):
        # The paired inverse net is directly trained to map z_S -> z_T; training drops
        # its recon below the analytic base inverse (pinv(A_base) + bilinear downsample).
        torch.manual_seed(0)
        C_T, C_S, f = 8, 4, 2
        true_net = self._true_upsample_net(C_T, C_S, f)
        t = self._make(C_T=C_T, C_S=C_S, hidden=32, n_layers=2)
        t.set_online_lr(1e-2)
        # Fixed eval pair (teacher z_T and its true student image z_S).
        z_T_eval = torch.randn(8, C_T, 4, 4)
        z_S_eval = true_net(z_T_eval).detach()

        def inv_err():
            with torch.no_grad():
                return float(
                    (t._inverse_spatial(z_S_eval) - z_T_eval).pow(2).mean()
                )

        t.update_online(
            [torch.randn(6, C_T, 4, 4)],
            [true_net(torch.randn(6, C_T, 4, 4)).detach()],
            update_base=True,
            update_mod=False,
        )
        before = inv_err()
        for _ in range(200):
            z_T = [torch.randn(6, C_T, 4, 4) for _ in range(2)]
            z_S = [true_net(z).detach() for z in z_T]
            t.update_online(z_T, z_S, update_base=False, update_mod=True)
        after = inv_err()
        self.assertLess(after, before)

    def test_requires_update_base_before_mod(self):
        # update_mod before the residual nets exist (no base/grids yet) must raise.
        t = self._make()
        with self.assertRaises(RuntimeError):
            t.update_online(
                [torch.randn(2, 8, 4, 4)],
                [torch.randn(2, 4, 8, 8)],
                update_base=False,
                update_mod=True,
            )

    def test_non_integer_upscale_raises(self):
        # Teacher 3x3 -> student 4x4 is not an integer upscale -> fail fast.
        t = self._make()
        with self.assertRaises(ValueError):
            t.fit([torch.randn(2, 8, 3, 3)], [torch.randn(2, 4, 4, 4)])

    def test_state_dict_roundtrip(self):
        torch.manual_seed(0)
        C_T, C_S, f = 8, 4, 2
        true_net = self._true_upsample_net(C_T, C_S, f)
        t = self._make(C_T=C_T, C_S=C_S)
        t.set_online_lr(1e-2)
        for _ in range(10):
            z_T = [torch.randn(6, C_T, 4, 4) for _ in range(2)]
            z_S = [true_net(z).detach() for z in z_T]
            t.update_online(z_T, z_S)
        sd = t.state_dict()
        self.assertIn("_student_grid", sd)
        self.assertIn("_upscale", sd)
        t2 = self._make(C_T=C_T, C_S=C_S)
        t2.load_state_dict(sd)
        self.assertTrue(t2.is_fitted)
        probe = torch.randn(2, C_T, 4, 4)
        torch.testing.assert_close(t2.transport_sample(probe), t.transport_sample(probe))

    def test_build_via_factory(self):
        for name in ("conv", "conv_linear"):
            t = build_transport(
                name,
                teacher_to_spatial=_bchw_identity,
                teacher_from_spatial=_bchw_identity,
                student_to_spatial=_bchw_identity,
                student_from_spatial=_bchw_identity,
                teacher_channels=8,
                student_channels=4,
            )
            self.assertIsInstance(t, ConvTransport)
            self.assertTrue(t.requires_warmup)


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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU smoke test")
class TestTransportGPUSmoke(unittest.TestCase):
    """GPU smoke test for the A/B/C improvements at realistic channel counts.

    Uses the cross-VAE FLUX.2 (C_T=128) -> SD3.5 (C_S=16) channel sizes on a small
    spatial grid so it runs in <1GB and a couple of seconds, validating the
    timestep->sigma conditioning (A), the pinv cache (B) and the two-phase base/MLP
    schedule (C) on a CUDA device without loading any model.
    """

    C_T = 128
    C_S = 16
    H = W = 32
    B = 8

    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(0)
        # A fixed "true" teacher->student channel projection so targets are learnable.
        self.P = torch.randn(self.C_S, self.C_T, device=self.device)

    def _make(self):
        t = AdaLNTransport(
            teacher_to_spatial=_bchw_identity,
            teacher_from_spatial=_bchw_identity,
            student_to_spatial=_bchw_identity,
            student_from_spatial=_bchw_identity,
            teacher_channels=self.C_T,
            student_channels=self.C_S,
        )
        return t.to(self.device)

    def _pair(self, sigma_val):
        """A fresh (z_T, z_S) pair whose channel scale depends on the noise level."""
        zt = torch.randn(self.B, self.C_T, self.H, self.W, device=self.device)
        proj = torch.einsum("sc,bchw->bshw", self.P, zt)
        scale = 2.0 if sigma_val < 0.5 else 0.5  # per-sigma correction the MLP must learn
        return zt, proj * scale

    def test_a_sigma_conditioning_and_recon_decrease(self):
        t = self._make()
        t.set_online_lr(1e-2)
        recons = []
        for _ in range(40):
            lo, hi = self._pair(0.1), self._pair(0.9)
            z_T = [lo[0], hi[0]]
            z_S = [lo[1], hi[1]]
            recons.append(t.update_online(z_T, z_S, sigma_list=[0.1, 0.9]))
        # recon should improve as the per-sigma modulation learns the correction.
        self.assertLess(
            sum(recons[-5:]) / 5.0, sum(recons[:5]) / 5.0,
            msg=f"recon did not decrease: first5={recons[:5]} last5={recons[-5:]}",
        )
        # the modulation must produce DIFFERENT outputs at the two noise levels.
        probe = torch.randn(self.B, self.C_T, self.H, self.W, device=self.device)
        out_lo = t.transport_sample(probe, sigma=0.1)
        out_hi = t.transport_sample(probe, sigma=0.9)
        self.assertGreater((out_lo - out_hi).abs().mean().item(), 1e-3)

    def test_b_pinv_cache_hit(self):
        t = self._make()
        z_T = [self._pair(0.2)[0] for _ in range(2)]
        z_S = [torch.einsum("sc,bchw->bshw", self.P, z) for z in z_T]
        t.fit(z_T, z_S)
        self.assertIsNone(t._A_base_pinv)
        x_S = torch.randn(self.B, self.C_S, self.H, self.W, device=self.device)
        out1 = t.transition_mean_to_student(x_S, query_teacher_mean=lambda x: x, sigma=0.4)
        cached = t._A_base_pinv
        self.assertIsNotNone(cached)  # built on first inverse
        out2 = t.transition_mean_to_student(x_S, query_teacher_mean=lambda x: x, sigma=0.4)
        # same cached object reused (no recompute) and identical result.
        self.assertIs(t._A_base_pinv, cached)
        torch.testing.assert_close(out1, out2)

    def test_inverse_exact_at_fixed_sigma(self):
        # C_S < C_T -> A_base has full row rank -> forward(inverse(x_S)) == x_S exactly
        # (right inverse), at any fixed sigma, with identity teacher mean.
        t = self._make()
        z_T = [self._pair(0.3)[0] for _ in range(3)]
        z_S = [torch.einsum("sc,bchw->bshw", self.P, z) for z in z_T]
        t.fit(z_T, z_S)
        x_S = torch.randn(self.B, self.C_S, self.H, self.W, device=self.device)
        rt = t.transition_mean_to_student(x_S, query_teacher_mean=lambda x: x, sigma=0.6)
        torch.testing.assert_close(rt, x_S, atol=1e-3, rtol=1e-3)

    def test_c_two_phase_schedule(self):
        t = self._make()
        t.set_online_lr(1e-2)
        z_T = [self._pair(0.1)[0], self._pair(0.9)[0]]
        z_S = [self._pair(0.1)[1], self._pair(0.9)[1]]
        # Phase 1: base only.
        t.update_online(z_T, z_S, sigma_list=[0.1, 0.9], update_base=True, update_mod=False)
        self.assertTrue(t.is_fitted)
        base_after_p1 = t.A_base.clone()
        mlp_last_p1 = t.mod_mlp[-1].weight.clone()
        # Phase 2: modulation only -> base frozen, MLP changes.
        for _ in range(10):
            t.update_online(
                z_T, z_S, sigma_list=[0.1, 0.9], update_base=False, update_mod=True
            )
        torch.testing.assert_close(t.A_base, base_after_p1, atol=0.0, rtol=0.0)
        self.assertGreater((t.mod_mlp[-1].weight - mlp_last_p1).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
