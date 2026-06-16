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

"""CPU unit tests for the MoF KL-min loss + regularizer math.

Covers the three differentiable staticmethods on ``MoFKLMinTrainer``:
``_klmin_loss`` (velocity-MSE to base), ``_entropy_bonus`` (weight entropy),
and ``_uniform_anchor`` (distance-to-uniform). No GPU / model is needed.
"""

import math
import unittest

import torch

from flow_factory.hparams import MoFKLMinTrainingArguments
from flow_factory.trainers.mof.klmin import MoFKLMinTrainer


class TestKLMinLoss(unittest.TestCase):
    def test_matches_convex_combo_of_task_vectors(self) -> None:
        """mean_spatial ||v_lambda - v_base||^2 == mean_spatial ||sum_k w_k tau_k||^2.

        With sum_k w_k = 1 the two forms are algebraically identical
        (tau_k = v_k - v_base); this pins the documented identity.
        """
        torch.manual_seed(0)
        K, B, C, H, W = 3, 4, 2, 5, 5
        teachers = torch.randn(K, B, C, H, W)
        v_base = torch.randn(B, C, H, W)
        w = torch.softmax(torch.randn(K), dim=0)
        w = w / w.sum()  # enforce exact sum-to-one for the identity

        v_lambda = (w.view(K, 1, 1, 1, 1) * teachers).sum(dim=0)  # (B, C, H, W)
        loss = MoFKLMinTrainer._klmin_loss(v_lambda, v_base)  # (B,)

        tau = teachers - v_base.unsqueeze(0)
        combo = (w.view(K, 1, 1, 1, 1) * tau).sum(dim=0)  # (B, C, H, W)
        expected = (combo ** 2).mean(dim=(1, 2, 3))  # (B,)

        self.assertEqual(loss.shape, (B,))
        torch.testing.assert_close(loss, expected)

    def test_zero_when_mixture_equals_base(self) -> None:
        v = torch.randn(2, 3, 4, 4)
        loss = MoFKLMinTrainer._klmin_loss(v.clone(), v.clone())
        torch.testing.assert_close(loss, torch.zeros(2))

    def test_returns_float32_from_bf16_inputs(self) -> None:
        v_lambda = torch.randn(2, 3, 4, 4, dtype=torch.bfloat16)
        v_base = torch.randn(2, 3, 4, 4, dtype=torch.bfloat16)
        loss = MoFKLMinTrainer._klmin_loss(v_lambda, v_base)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.shape, (2,))

    def test_gradient_flows_to_mixture_only(self) -> None:
        teachers = torch.randn(3, 2, 4, 4)
        v_base = torch.randn(2, 4, 4)
        logits = torch.zeros(3, requires_grad=True)
        w = torch.softmax(logits, dim=0)
        v_lambda = (w.view(3, 1, 1, 1) * teachers).sum(dim=0)
        MoFKLMinTrainer._klmin_loss(v_lambda, v_base).mean().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            MoFKLMinTrainer._klmin_loss(torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 5))


class TestEntropyBonus(unittest.TestCase):
    def test_uniform_is_log_k(self) -> None:
        K, B = 4, 6
        w = torch.full((K, B), 1.0 / K)
        h = MoFKLMinTrainer._entropy_bonus(w)
        self.assertAlmostEqual(h.item(), math.log(K), places=5)

    def test_one_hot_is_zero(self) -> None:
        K, B = 3, 5
        w = torch.zeros(K, B)
        w[0] = 1.0
        h = MoFKLMinTrainer._entropy_bonus(w)
        self.assertAlmostEqual(h.item(), 0.0, places=5)

    def test_uniform_maximizes_entropy(self) -> None:
        K, B = 3, 1
        w_uniform = torch.full((K, B), 1.0 / K)
        w_skew = torch.tensor([[0.7], [0.2], [0.1]])
        self.assertGreater(
            MoFKLMinTrainer._entropy_bonus(w_uniform).item(),
            MoFKLMinTrainer._entropy_bonus(w_skew).item(),
        )

    def test_returns_scalar(self) -> None:
        h = MoFKLMinTrainer._entropy_bonus(torch.softmax(torch.randn(3, 4), dim=0))
        self.assertEqual(h.dim(), 0)


class TestUniformAnchor(unittest.TestCase):
    def test_zero_at_uniform(self) -> None:
        K, B = 4, 5
        w = torch.full((K, B), 1.0 / K)
        a = MoFKLMinTrainer._uniform_anchor(w, K)
        self.assertAlmostEqual(a.item(), 0.0, places=6)

    def test_matches_manual_one_hot(self) -> None:
        K = 3
        w = torch.tensor([[1.0], [0.0], [0.0]])
        a = MoFKLMinTrainer._uniform_anchor(w, K)
        expected = ((w - 1.0 / K) ** 2).sum(dim=0).mean()
        torch.testing.assert_close(a, expected)
        # (1 - 1/3)^2 + 2 * (1/3)^2 = 4/9 + 2/9 = 6/9
        self.assertAlmostEqual(a.item(), 6.0 / 9.0, places=6)

    def test_positive_when_not_uniform(self) -> None:
        w = torch.softmax(torch.randn(3, 4) * 3, dim=0)
        self.assertGreater(MoFKLMinTrainer._uniform_anchor(w, 3).item(), 0.0)


class TestKLMinArgsEntropyGuard(unittest.TestCase):
    """The entropy bonus needs softmax weights (a valid distribution); affine /
    none admit negative weights that make H(w) ill-defined, so __post_init__
    must reject klmin_entropy_coeff>0 there."""

    @staticmethod
    def _make(**overrides) -> MoFKLMinTrainingArguments:
        kwargs = dict(
            teacher_paths=["/tmp/a", "/tmp/b"],
            unique_sample_num_per_epoch=1,
            group_size=1,
            per_device_batch_size=1,
        )
        kwargs.update(overrides)
        return MoFKLMinTrainingArguments(**kwargs)

    def test_entropy_allowed_with_softmax(self) -> None:
        args = self._make(weight_normalization="softmax", klmin_entropy_coeff=0.1)
        self.assertEqual(args.klmin_entropy_coeff, 0.1)

    def test_entropy_rejected_with_affine(self) -> None:
        with self.assertRaisesRegex(ValueError, "weight_normalization='softmax'"):
            self._make(weight_normalization="affine", klmin_entropy_coeff=0.1)

    def test_entropy_rejected_with_none(self) -> None:
        with self.assertRaisesRegex(ValueError, "weight_normalization='softmax'"):
            self._make(weight_normalization="none", klmin_entropy_coeff=0.1)

    def test_non_softmax_allowed_when_entropy_off(self) -> None:
        # The default (entropy off) must not trip the guard under affine/none;
        # the uniform anchor is the normalization-agnostic alternative.
        args = self._make(weight_normalization="affine", klmin_uniform_anchor_coeff=0.5)
        self.assertEqual(args.klmin_entropy_coeff, 0.0)
        self.assertEqual(args.klmin_uniform_anchor_coeff, 0.5)


if __name__ == "__main__":
    unittest.main()
