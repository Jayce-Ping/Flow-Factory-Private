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

"""Unit tests for MoF weight-normalization modes (softmax / affine / none).

Covers the shared logits→weights mapping, the checkpoint-mode resolution
helper, mode-aware initialization of LUT and router mixing modules, and the
Σw≈1 soft-penalty value.
"""

import unittest

import torch
import torch.nn.functional as F

from flow_factory.trainers.mof.utils import (
    apply_weight_normalization,
    resolve_weight_normalization,
)
from flow_factory.trainers.mof.common import (
    MoFAdaLNRouter,
    MoFMLPRouter,
    MoFMixingModule,
    MoFMixingModuleSimple,
    MoFTimeRouter,
    MoFTrainerBase,
    _router_output_bias_init,
    create_mixing_module,
)

K, T, S = 3, 5, 3
D_POOL, D_HIDDEN, D_TIME = 32, 16, 8


class TestApplyWeightNormalization(unittest.TestCase):
    def test_softmax_matches_torch(self) -> None:
        logits = torch.randn(K, T, S)
        out = apply_weight_normalization(logits, "softmax", temperature=0.7, dim=0)
        torch.testing.assert_close(out, F.softmax(logits / 0.7, dim=0))
        torch.testing.assert_close(out.sum(dim=0), torch.ones(T, S))

    def test_affine_projects_onto_sum_one(self) -> None:
        logits = torch.randn(K, T, S)
        out = apply_weight_normalization(logits, "affine", dim=0)
        torch.testing.assert_close(out.sum(dim=0), torch.ones(T, S))

    def test_affine_identity_when_already_sum_one(self) -> None:
        logits = torch.randn(K, T, S)
        logits = logits - (logits.sum(dim=0, keepdim=True) - 1.0) / K
        out = apply_weight_normalization(logits, "affine", dim=0)
        torch.testing.assert_close(out, logits)

    def test_affine_allows_negative_weights(self) -> None:
        # CFG-style: [2, -1] sums to 1 and must pass through unchanged.
        logits = torch.tensor([[2.0], [-1.0]])
        out = apply_weight_normalization(logits, "affine", dim=0)
        torch.testing.assert_close(out, logits)

    def test_none_is_identity(self) -> None:
        logits = torch.randn(K, T, S)
        out = apply_weight_normalization(logits, "none", temperature=0.1, dim=0)
        self.assertIs(out, logits)

    def test_temperature_ignored_outside_softmax(self) -> None:
        logits = torch.randn(K, T, S)
        for mode in ("affine", "none"):
            a = apply_weight_normalization(logits, mode, temperature=1.0, dim=0)
            b = apply_weight_normalization(logits, mode, temperature=0.01, dim=0)
            torch.testing.assert_close(a, b)

    def test_router_axis(self) -> None:
        logits = torch.randn(4, K)  # (B, K) router convention
        out = apply_weight_normalization(logits, "affine", dim=-1)
        torch.testing.assert_close(out.sum(dim=-1), torch.ones(4))

    def test_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            apply_weight_normalization(torch.randn(K), "sigmoid", dim=0)


class TestResolveWeightNormalization(unittest.TestCase):
    def test_precedence_matrix(self) -> None:
        # (ckpt_value, override) → expected
        cases = [
            (None, None, "softmax"),  # legacy fallback
            ("none", None, "none"),
            ("affine", None, "affine"),
            (None, "none", "none"),
            ("softmax", "none", "none"),  # override wins (with warning)
            ("none", "softmax", "softmax"),
        ]
        for ckpt, override, expected in cases:
            self.assertEqual(
                resolve_weight_normalization(ckpt, override), expected,
                msg=f"ckpt={ckpt!r}, override={override!r}",
            )

    def test_invalid_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            resolve_weight_normalization("sigmoid", None)
        with self.assertRaises(ValueError):
            resolve_weight_normalization(None, "sigmoid")


class TestLUTModules(unittest.TestCase):
    def test_softmax_uniform_init(self) -> None:
        for cls in (MoFMixingModule, MoFMixingModuleSimple):
            mod = cls(K=K, T=T, S=S, weight_normalization="softmax", init_mode="uniform")
            w = mod.forward()
            self.assertEqual(tuple(w.shape), (K, T, S))
            torch.testing.assert_close(w, torch.full((K, T, S), 1.0 / K))

    def test_none_uniform_init_equals_softmax_uniform(self) -> None:
        for cls in (MoFMixingModule, MoFMixingModuleSimple):
            mod = cls(K=K, T=T, S=S, weight_normalization="none", init_mode="uniform")
            torch.testing.assert_close(mod.forward(), torch.full((K, T, S), 1.0 / K))

    def test_affine_uniform_init_is_identity_projection(self) -> None:
        mod = MoFMixingModule(K=K, T=T, S=S, weight_normalization="affine", init_mode="uniform")
        torch.testing.assert_close(mod.forward(), torch.full((K, T, S), 1.0 / K))

    def test_none_mode_weights_are_raw_logits(self) -> None:
        mod = MoFMixingModule(K=K, T=T, S=S, weight_normalization="none", init_mode="uniform")
        with torch.no_grad():
            mod.logits.fill_(2.0 / K)
        torch.testing.assert_close(
            mod.forward().sum(dim=0), torch.full((T, S), 2.0)
        )  # Σw=2 is impossible under softmax/affine → raw path proven

    def test_hard_init_rejected_under_softmax(self) -> None:
        with self.assertRaises(ValueError):
            MoFMixingModule(
                K=K, T=T, S=S, weight_normalization="softmax", init_mode="hard",
                teacher_set_mapping={0: 0, 1: 1, 2: 2},
            )

    def test_hard_init_one_hot_under_none(self) -> None:
        mapping = {0: 0, 1: 1, 2: 2}
        mod = MoFMixingModule(
            K=K, T=T, S=S, weight_normalization="none", init_mode="hard",
            teacher_set_mapping=mapping,
        )
        w = mod.forward()
        for s_id, k_idx in mapping.items():
            expected = torch.zeros(K)
            expected[k_idx] = 1.0
            torch.testing.assert_close(w[:, 0, s_id], expected)

    def test_teacher_biased_none_matches_softmax_values(self) -> None:
        mapping = {0: 0, 1: 1, 2: 2}
        mod = MoFMixingModule(
            K=K, T=T, S=S, weight_normalization="none",
            init_mode="teacher_biased", init_bias=2.0,
            teacher_set_mapping=mapping,
        )
        w = mod.forward()
        expected_col = F.softmax(torch.tensor([2.0, 0.0, 0.0]), dim=0)
        torch.testing.assert_close(w[:, 0, 0], expected_col)
        torch.testing.assert_close(w.sum(dim=0), torch.ones(T, S))


class TestRouterBiasInit(unittest.TestCase):
    def test_softmax_mode_zero_bias(self) -> None:
        bias = _router_output_bias_init(K, "softmax", init_mode="teacher_biased")
        torch.testing.assert_close(bias, torch.zeros(K))

    def test_none_mode_uniform(self) -> None:
        bias = _router_output_bias_init(K, "none", init_mode="uniform")
        torch.testing.assert_close(bias, torch.full((K,), 1.0 / K))

    def test_none_mode_teacher_biased_profile(self) -> None:
        bias = _router_output_bias_init(
            K, "none", init_mode="teacher_biased", init_bias=2.0,
            teacher_set_mapping={0: 0},
        )
        torch.testing.assert_close(bias, F.softmax(torch.tensor([2.0, 0.0, 0.0]), dim=0))

    def test_hard_rejected_for_routers(self) -> None:
        with self.assertRaises(ValueError):
            _router_output_bias_init(K, "none", init_mode="hard")


def _build_router(module_type: str, mode: str):
    return create_mixing_module(
        module_type=module_type,
        K=K,
        d_pool=D_POOL,
        d_hidden=D_HIDDEN,
        d_time=D_TIME,
        weight_normalization=mode,
        init_mode="uniform",
    )


def _router_forward(router, B: int = 4) -> torch.Tensor:
    t = torch.rand(B) * 1000
    prompt_embeds = torch.randn(B, 7, D_POOL)
    pooled = torch.randn(B, D_POOL)
    return router(t, prompt_embeds, pooled)  # (K, B)


ROUTER_TYPES = ("time_router", "adaln_router", "mlp_router")


class TestRouterModes(unittest.TestCase):
    def test_initial_mixture_uniform_all_modes(self) -> None:
        for module_type in ROUTER_TYPES:
            for mode in ("softmax", "affine", "none"):
                router = _build_router(module_type, mode)
                w = _router_forward(router)
                self.assertEqual(tuple(w.shape), (K, 4), msg=f"{module_type}/{mode}")
                torch.testing.assert_close(
                    w, torch.full((K, 4), 1.0 / K),
                    msg=f"{module_type}/{mode}: initial mixture not uniform",
                )

    def test_none_mode_raw_path(self) -> None:
        # Push the output bias to 2/K: column sums must become 2 — impossible
        # under softmax (Σ=1) or affine (projected back to Σ=1).
        for module_type in ROUTER_TYPES:
            router = _build_router(module_type, "none")
            out_layer = (
                router.out_proj if module_type == "time_router" else router.mlp[-1]
            )
            with torch.no_grad():
                out_layer.bias.fill_(2.0 / K)
            w = _router_forward(router)
            torch.testing.assert_close(
                w.sum(dim=0), torch.full((4,), 2.0),
                msg=f"{module_type}: 'none' mode did not bypass normalization",
            )

    def test_affine_mode_keeps_sum_one(self) -> None:
        for module_type in ROUTER_TYPES:
            router = _build_router(module_type, "affine")
            out_layer = (
                router.out_proj if module_type == "time_router" else router.mlp[-1]
            )
            with torch.no_grad():
                out_layer.bias.fill_(2.0 / K)  # Σbias=2 → projected back to Σ=1
            w = _router_forward(router)
            torch.testing.assert_close(w.sum(dim=0), torch.ones(4))

    def test_softmax_regression(self) -> None:
        for module_type in ROUTER_TYPES:
            router = _build_router(module_type, "softmax")
            with torch.no_grad():
                for p in router.parameters():
                    p.add_(torch.randn_like(p) * 0.05)
            w = _router_forward(router)
            torch.testing.assert_close(w.sum(dim=0), torch.ones(4))
            self.assertTrue((w >= 0).all())

    def test_adaln_identity_modulation_init(self) -> None:
        router = _build_router("adaln_router", "softmax")
        t_hidden = torch.randn(4, D_HIDDEN)
        gamma, beta = router.adaLN_modulation(t_hidden).chunk(2, dim=-1)
        torch.testing.assert_close(gamma, torch.ones(4, D_HIDDEN))
        torch.testing.assert_close(beta, torch.zeros(4, D_HIDDEN))
        # h = gamma * c + beta == c → text conditioning flows from step 0.
        c = torch.randn(4, D_HIDDEN)
        torch.testing.assert_close(gamma * c + beta, c)

    def test_factory_forwards_mode_to_all_module_types(self) -> None:
        for module_type in ("lut", "lut_simple") + ROUTER_TYPES:
            mod = create_mixing_module(
                module_type=module_type,
                K=K, T=T, S=S,
                d_pool=D_POOL, d_hidden=D_HIDDEN, d_time=D_TIME,
                weight_normalization="none",
                init_mode="uniform",
            )
            self.assertEqual(mod.weight_normalization, "none", msg=module_type)


class TestWeightSumPenalty(unittest.TestCase):
    def test_zero_on_simplex(self) -> None:
        w_kb = torch.full((K, 6), 1.0 / K)
        self.assertAlmostEqual(
            MoFTrainerBase._weight_sum_penalty_value(w_kb).item(), 0.0
        )

    def test_hand_computed_value(self) -> None:
        # Columns sum to 1.5 and 0.5 → penalty = ((0.5)² + (−0.5)²)/2 = 0.25
        w_kb = torch.tensor([[1.0, 0.5], [0.5, 0.0], [0.0, 0.0]])
        self.assertAlmostEqual(
            MoFTrainerBase._weight_sum_penalty_value(w_kb).item(), 0.25
        )

    def test_gradient_flows(self) -> None:
        w_kb = torch.rand(K, 4, requires_grad=True)
        MoFTrainerBase._weight_sum_penalty_value(w_kb).backward()
        self.assertIsNotNone(w_kb.grad)
        self.assertTrue(torch.isfinite(w_kb.grad).all())


class TestDistillLutSemantics(unittest.TestCase):
    """Exercise the resolution + mapping used by stage-2 distill on a
    minimal simulated mof_state dict (the silent-softmax regression)."""

    def test_none_checkpoint_keeps_one_hot(self) -> None:
        logits = torch.zeros(K, T, S)
        for s in range(S):
            logits[s, :, s] = 1.0  # hard route, raw 'none'-mode weights
        state = {"weight_normalization": "none", "temperature": 1.0}
        mode = resolve_weight_normalization(state.get("weight_normalization"), None)
        weights = apply_weight_normalization(
            logits, mode, state.get("temperature", 1.0), dim=0
        )
        torch.testing.assert_close(weights, logits)  # NOT softmaxed

    def test_legacy_checkpoint_defaults_to_softmax(self) -> None:
        logits = torch.zeros(K, T, S)
        state: dict = {}
        mode = resolve_weight_normalization(state.get("weight_normalization"), None)
        self.assertEqual(mode, "softmax")
        weights = apply_weight_normalization(logits, mode, 1.0, dim=0)
        torch.testing.assert_close(weights, torch.full((K, T, S), 1.0 / K))


if __name__ == "__main__":
    unittest.main()
