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

"""Focused trainer-wiring tests for P-OPD behavior-transition caches."""

import unittest
from collections import defaultdict
from types import SimpleNamespace

import torch

from flow_factory.trainers.xopd.trainer import XOPDTrainer


class TestPOPDTrainerCache(unittest.TestCase):
    def _trainer(self) -> XOPDTrainer:
        trainer = XOPDTrainer.__new__(XOPDTrainer)
        trainer.training_args = SimpleNamespace(popd_alpha=0.5, popd_temperature=1.0)
        trainer.adapter = SimpleNamespace(
            scheduler=SimpleNamespace(dynamics_type="Flow-SDE"),
        )
        trainer.popd_alpha = 0.5
        trainer.popd_temperature = 1.0
        return trainer

    def test_builds_cache_from_the_same_rollout_transition(self) -> None:
        trainer = self._trainer()
        mu_old = torch.tensor([[[0.0, 0.0]]])
        next_latents = torch.tensor([[[3.0, 3.0], [1.0, 1.0]]])
        batch = {
            "all_latents": next_latents,
            "next_latents_mean": mu_old,
            "std_dev_t": torch.ones(1, 1, 1),
            "dt": -0.25 * torch.ones(1, 1, 1),
            "callback_index_map": torch.zeros(1, 1, dtype=torch.long),
        }
        caches = trainer._precompute_popd_step_caches(
            batch=batch,
            latents_index_map=torch.tensor([0, 1]),
            mu_teacher_list=[torch.ones(1, 2)],
            timestep_indices=[0],
        )
        self.assertEqual(len(caches), 1)
        torch.testing.assert_close(caches[0].mu_old, torch.zeros(1, 2))
        torch.testing.assert_close(caches[0].transition_variance, torch.tensor([0.25]))
        torch.testing.assert_close(caches[0].next_latents, torch.ones(1, 2))
        self.assertFalse(caches[0].responsibility.teacher_responsibility.requires_grad)

    def test_rejects_misaligned_teacher_cache_length(self) -> None:
        trainer = self._trainer()
        with self.assertRaises(ValueError):
            trainer._precompute_popd_step_caches(
                batch={},
                latents_index_map=torch.tensor([0, 1]),
                mu_teacher_list=[],
                timestep_indices=[0],
            )

    def test_diagnostics_keep_per_sample_timestep_values_and_global_quantiles(self) -> None:
        trainer = self._trainer()
        trainer.accelerator = SimpleNamespace(gather=lambda values: values)
        loss_info = defaultdict(list)
        gamma = torch.tensor([0.1, 0.5, 0.9])
        trainer._append_popd_diagnostics(
            loss_info,
            {"gamma": gamma, "teacher_old_gap_rms": torch.ones(3)},
            timestep_index=2,
        )
        self.assertEqual(loss_info["popd/gamma/t2"][0].shape, (3,))
        self.assertEqual(loss_info["popd/teacher_old_gap_rms/t2"][0].shape, (3,))

        quantiles = trainer._gather_popd_gamma_quantiles(loss_info)
        torch.testing.assert_close(quantiles["popd/gamma_p50"], torch.tensor(0.5))
        torch.testing.assert_close(quantiles["popd/gamma/t2_p90"], torch.tensor(0.82))


if __name__ == "__main__":
    unittest.main()
