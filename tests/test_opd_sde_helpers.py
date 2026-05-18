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

"""Unit tests for OPD-SDE static helpers (R_bar aggregation, group centering)."""

import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
