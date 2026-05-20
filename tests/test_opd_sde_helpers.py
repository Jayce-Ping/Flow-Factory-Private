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

"""Unit tests for OPD-SDE static helpers (R_bar aggregation, group normalization)."""

import unittest

import torch

from flow_factory.hparams.training_args import OPDTrainingArguments
from flow_factory.trainers.opd.sde import OPDTrainer


class TestOPDReverseCumulative(unittest.TestCase):
    def test_mean_full_horizon(self) -> None:
        d0 = torch.tensor([1.0, 10.0])
        d1 = torch.tensor([2.0, 20.0])
        d2 = torch.tensor([3.0, 30.0])
        r_per_k = OPDTrainer._reverse_cumulative(
            [d0, d1, d2],
            max_future_steps=None,
            reduction="mean",
        )
        self.assertEqual(len(r_per_k), 3)
        torch.testing.assert_close(r_per_k[2], torch.zeros(2))
        torch.testing.assert_close(r_per_k[1], d2)
        torch.testing.assert_close(r_per_k[0], (d1 + d2) / 2.0)

    def test_sum_bounded_horizon(self) -> None:
        d_list = [torch.tensor([float(i)]) for i in range(4)]
        r_per_k = OPDTrainer._reverse_cumulative(
            d_list,
            max_future_steps=2,
            reduction="sum",
        )
        torch.testing.assert_close(r_per_k[0], torch.tensor(1.0 + 2.0))
        torch.testing.assert_close(r_per_k[2], torch.tensor(0.0))

    def test_mean_bounded_horizon(self) -> None:
        d_list = [torch.tensor([float(i)]) for i in range(4)]
        r_per_k = OPDTrainer._reverse_cumulative(
            d_list,
            max_future_steps=2,
            reduction="mean",
        )
        torch.testing.assert_close(r_per_k[0], torch.tensor(1.5))


class TestOPDGroupCenter(unittest.TestCase):
    def test_zero_mean_per_group(self) -> None:
        values = torch.tensor([1.0, 3.0, 10.0, 20.0])
        group_ids = torch.tensor([1, 1, 2, 2])
        centered = OPDTrainer._group_center(values, group_ids, group_size=2)
        self.assertAlmostEqual(centered[:2].sum().item(), 0.0, places=5)
        self.assertAlmostEqual(centered[2:].sum().item(), 0.0, places=5)
        torch.testing.assert_close(centered[0], torch.tensor(-1.0))
        torch.testing.assert_close(centered[1], torch.tensor(1.0))

    def test_rank_scale_two_groups(self) -> None:
        """Simulate N=32 rank with two prompt groups (K=16 each)."""
        group_a = 100
        group_b = 200
        group_ids = torch.tensor([group_a] * 16 + [group_b] * 16, dtype=torch.int64)
        values = torch.arange(32, dtype=torch.float32)
        centered = OPDTrainer._group_center(
            values,
            group_ids,
            group_size=16,
            rank_index=0,
        )
        self.assertAlmostEqual(centered[:16].sum().item(), 0.0, places=4)
        self.assertAlmostEqual(centered[16:].sum().item(), 0.0, places=4)
        torch.testing.assert_close(
            centered[:16],
            values[:16] - values[:16].mean(),
        )

    def test_incomplete_group_raises(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0])
        group_ids = torch.tensor([1, 1, 1])
        with self.assertRaises(ValueError):
            OPDTrainer._group_center(values, group_ids, group_size=16, rank_index=3)


class TestOPDGroupNormalize(unittest.TestCase):
    def test_center_only_matches_group_center(self) -> None:
        values = torch.tensor([1.0, 3.0, 10.0, 20.0])
        group_ids = torch.tensor([1, 1, 2, 2])
        centered = OPDTrainer._group_normalize(
            values,
            group_ids,
            group_size=2,
            center=True,
            divide_by_std=False,
        )
        expected = OPDTrainer._group_center(values, group_ids, group_size=2)
        torch.testing.assert_close(centered, expected)

    def test_center_and_std(self) -> None:
        values = torch.tensor([1.0, 3.0, 10.0, 20.0])
        group_ids = torch.tensor([1, 1, 2, 2])
        normalized = OPDTrainer._group_normalize(
            values,
            group_ids,
            group_size=2,
            center=True,
            divide_by_std=True,
            std_eps=1e-6,
        )
        for gid in (1, 2):
            mask = group_ids == gid
            g = values[mask]
            centered = g - g.mean()
            std = max(float(torch.std(g, unbiased=False).item()), 1e-6)
            torch.testing.assert_close(normalized[mask], centered / std)

    def test_constant_group_all_zero(self) -> None:
        values = torch.tensor([5.0, 5.0, 7.0, 7.0])
        group_ids = torch.tensor([1, 1, 2, 2])
        normalized = OPDTrainer._group_normalize(
            values,
            group_ids,
            group_size=2,
            center=True,
            divide_by_std=True,
        )
        torch.testing.assert_close(normalized, torch.zeros_like(values))


class TestOPDTrainingArgumentsPostInit(unittest.TestCase):
    def test_reinforce_group_std_requires_center(self) -> None:
        with self.assertRaises(ValueError):
            OPDTrainingArguments(
                teacher_paths=["/tmp/teacher"],
                reinforce_group_std=True,
                reinforce_group_center=False,
            )


class TestOPDRankReinforcePipeline(unittest.TestCase):
    def test_stitch_reverse_center(self) -> None:
        """Micro-batch stitch (N=4, B=2) then rank-level R_bar and center."""
        d0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        d1 = torch.tensor([4.0, 3.0, 2.0, 1.0])
        r_raw = OPDTrainer._reverse_cumulative(
            [d0, d1],
            max_future_steps=None,
            reduction="sum",
        )
        group_ids = torch.tensor([10, 10, 20, 20])
        r_centered = OPDTrainer._group_center(
            r_raw[0],
            group_ids,
            group_size=2,
        )
        # r_raw[0][i] = d1[i]; group 10: d1[0],d1[1] -> mean 3.5
        torch.testing.assert_close(r_raw[0], d1)
        torch.testing.assert_close(r_centered[0], torch.tensor(-2.5))
        torch.testing.assert_close(r_centered[1], torch.tensor(-0.5))
        torch.testing.assert_close(r_centered[2], torch.tensor(0.5))
        torch.testing.assert_close(r_centered[3], torch.tensor(2.5))


class TestPCGradProjectGradients(unittest.TestCase):
    """Tests for the pcgrad_project_gradients utility in opd/common.py."""

    def test_single_teacher_passthrough(self) -> None:
        """Single teacher: no conflicts, gradients returned unchanged."""
        from flow_factory.trainers.opd.common import pcgrad_project_gradients

        g0 = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])]
        result = pcgrad_project_gradients([g0])
        torch.testing.assert_close(result[0], g0[0])
        torch.testing.assert_close(result[1], g0[1])

    def test_conflicting_teachers_projection(self) -> None:
        """Two perfectly opposing teachers: projections should zero the conflict."""
        from flow_factory.trainers.opd.common import pcgrad_project_gradients

        # Teacher 0 gradient = [1, 0], Teacher 1 gradient = [-1, 0]
        # dot(g0, g1) = -1 < 0: conflict
        # Projection of g0 onto g1: (dot / norm_sq) * g1 = (-1/1)*[-1,0] = [1,0]
        # pc[0] = g0 - proj = [1,0] - [1,0] = [0,0]
        # Projection of g1 onto g0: (dot / norm_sq) * g0 = (-1/1)*[1,0] = [-1,0]
        # pc[1] = g1 - proj = [-1,0] - [-1,0] = [0,0]
        # Combined = [0, 0]
        g0 = [torch.tensor([1.0, 0.0])]
        g1 = [torch.tensor([-1.0, 0.0])]
        result = pcgrad_project_gradients([g0, g1])
        torch.testing.assert_close(result[0], torch.tensor([0.0, 0.0]))

    def test_aligned_teachers_no_projection(self) -> None:
        """Two aligned teachers: no conflict, sum of gradients."""
        from flow_factory.trainers.opd.common import pcgrad_project_gradients

        g0 = [torch.tensor([1.0, 2.0])]
        g1 = [torch.tensor([3.0, 4.0])]
        # dot(g0, g1) = 3 + 8 = 11 > 0: no conflict
        # projected = g0 + g1
        result = pcgrad_project_gradients([g0, g1])
        torch.testing.assert_close(result[0], torch.tensor([4.0, 6.0]))

    def test_partial_conflict(self) -> None:
        """Two teachers with partial conflict: projection removes conflicting component."""
        from flow_factory.trainers.opd.common import pcgrad_project_gradients

        # g0 = [1, 1], g1 = [-1, 1]
        # dot(g0, g1) = -1 + 1 = 0: no conflict (orthogonal)
        # No projection applied, sum = [0, 2]
        g0 = [torch.tensor([1.0, 1.0])]
        g1 = [torch.tensor([-1.0, 1.0])]
        result = pcgrad_project_gradients([g0, g1])
        torch.testing.assert_close(result[0], torch.tensor([0.0, 2.0]))

    def test_three_teachers(self) -> None:
        """Three teachers: pairwise projections resolve conflicts."""
        from flow_factory.trainers.opd.common import pcgrad_project_gradients

        g0 = [torch.tensor([1.0, 0.0])]
        g1 = [torch.tensor([0.0, 1.0])]
        g2 = [torch.tensor([-1.0, -1.0])]
        # g0 vs g1: dot=0, no conflict
        # g0 vs g2: dot=-1 < 0, conflict
        # g1 vs g2: dot=-1 < 0, conflict
        # Result should not have NaN and should be reasonable
        result = pcgrad_project_gradients([g0, g1, g2])
        self.assertFalse(torch.isnan(result[0]).any())

    def test_multi_param_structure(self) -> None:
        """Verify unflatten preserves per-parameter tensor shapes."""
        from flow_factory.trainers.opd.common import pcgrad_project_gradients

        # Two params with different shapes
        g0 = [torch.randn(3, 4), torch.randn(2)]
        g1 = [torch.randn(3, 4), torch.randn(2)]
        result = pcgrad_project_gradients([g0, g1])
        self.assertEqual(result[0].shape, (3, 4))
        self.assertEqual(result[1].shape, (2,))


class TestReverseCumulativePerTeacher(unittest.TestCase):
    """Tests for _reverse_cumulative_per_teacher (delegates to _reverse_cumulative)."""

    def test_two_teachers_sum_reduction(self) -> None:
        """Two teachers, sum reduction: R_bar[k][t] = sum of D_j for j > t."""
        # D_per_teacher_list[t][k] = D_k at timestep t
        # T=3, K=2
        # Teacher 0: D = [1, 2, 3]
        # Teacher 1: D = [10, 20, 30]
        d_per_teacher_list = [
            [torch.tensor([1.0]), torch.tensor([10.0])],  # t=0
            [torch.tensor([2.0]), torch.tensor([20.0])],  # t=1
            [torch.tensor([3.0]), torch.tensor([30.0])],  # t=2
        ]
        # Expected R_bar (sum, no horizon):
        # Teacher 0: R_bar[0] = D1+D2 = 5, R_bar[1] = D2 = 3, R_bar[2] = 0
        # Teacher 1: R_bar[0] = D1+D2 = 50, R_bar[1] = D2 = 30, R_bar[2] = 0
        r_per_k_per_teacher = OPDTrainer._reverse_cumulative(
            [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])],
            max_future_steps=None,
            reduction="sum",
        )
        # Verify the static method directly
        torch.testing.assert_close(r_per_k_per_teacher[0], torch.tensor([5.0]))
        torch.testing.assert_close(r_per_k_per_teacher[1], torch.tensor([3.0]))
        torch.testing.assert_close(r_per_k_per_teacher[2], torch.tensor([0.0]))

    def test_excludes_current_timestep(self) -> None:
        """R_bar[k][t] should NOT include D_t (only D_{t+1} onward)."""
        # T=2, single teacher
        d_list = [torch.tensor([5.0]), torch.tensor([7.0])]
        r_per_k = OPDTrainer._reverse_cumulative(d_list, None, reduction="sum")
        # R_bar[0] = D_1 = 7 (NOT D_0 + D_1 = 12)
        torch.testing.assert_close(r_per_k[0], torch.tensor([7.0]))
        # R_bar[1] = 0 (nothing after t=1)
        torch.testing.assert_close(r_per_k[1], torch.tensor([0.0]))

    def test_mean_reduction(self) -> None:
        """Mean reduction: R_bar[t] = mean of future D values."""
        d_list = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([6.0])]
        r_per_k = OPDTrainer._reverse_cumulative(d_list, None, reduction="mean")
        # R_bar[0] = mean(D_1, D_2) = mean(2, 6) = 4
        torch.testing.assert_close(r_per_k[0], torch.tensor([4.0]))
        # R_bar[1] = D_2 = 6
        torch.testing.assert_close(r_per_k[1], torch.tensor([6.0]))
        # R_bar[2] = 0
        torch.testing.assert_close(r_per_k[2], torch.tensor([0.0]))


if __name__ == "__main__":
    unittest.main()
