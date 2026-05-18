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


if __name__ == "__main__":
    unittest.main()
