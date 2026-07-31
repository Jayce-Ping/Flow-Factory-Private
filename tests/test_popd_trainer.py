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
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from flow_factory.samples import BaseSample
from flow_factory.trainers.xopd.trainer import XOPDTrainer
from flow_factory.utils.trajectory_collector import (
    SCHEDULER_TRAIN_INDICES,
    resolve_scheduler_train_collection_indices,
)


class _AttrDict(dict):
    """Dictionary with attribute access for lightweight TrainingArguments tests."""

    def __getattr__(self, name):
        return self[name]


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

    def test_only_gate_and_joint_kl_are_broken_out_per_timestep(self) -> None:
        """Per-step series are restricted to the two keys that need a trajectory view.

        Every diagnostic expands into four statistics, so breaking all of them out per trained
        transition produced hundreds of series per epoch. The gate and the joint KL earn a
        per-step view because both vary by orders of magnitude along the denoising axis; the rest
        are only logged pooled.
        """
        trainer = self._trainer()
        trainer.accelerator = SimpleNamespace(gather=lambda values: values)
        loss_info = defaultdict(list)
        gamma = torch.tensor([0.1, 0.5, 0.9])
        trainer._append_popd_diagnostics(
            loss_info,
            {
                "gamma": gamma,
                "teacher_old_kl_joint": torch.tensor([1.0, 2.0, 3.0]),
                "ungated_mean_kl": torch.ones(3),
            },
            timestep_index=2,
        )
        self.assertEqual(loss_info["popd/gamma/t2"][0].shape, (3,))
        self.assertEqual(loss_info["popd/teacher_old_kl_joint/t2"][0].shape, (3,))
        self.assertNotIn("popd/ungated_mean_kl/t2", loss_info)
        # Pooled series still carry every diagnostic that was passed in.
        self.assertEqual(loss_info["popd/ungated_mean_kl"][0].shape, (3,))

        quantiles = trainer._gather_popd_gamma_quantiles(loss_info)
        torch.testing.assert_close(quantiles["popd/gamma_p50"], torch.tensor(0.5))
        self.assertEqual(
            sorted(quantiles),
            [
                "popd/gamma_p01",
                "popd/gamma_p10",
                "popd/gamma_p50",
                "popd/gamma_p90",
                "popd/gamma_p99",
            ],
        )


class TestPOPDScheduleAlignment(unittest.TestCase):
    def test_scheduler_train_sentinel_resolves_after_schedule_is_configured(self) -> None:
        trajectory_indices, callback_indices = resolve_scheduler_train_collection_indices(
            SCHEDULER_TRAIN_INDICES,
            scheduler_train_indices=torch.tensor([3, 1]),
            num_inference_steps=5,
        )
        self.assertEqual(trajectory_indices, [1, 2, 3, 4])
        self.assertEqual(callback_indices, [3, 1])

    def test_same_arch_sde_sample_defers_step_selection_to_inference(self) -> None:
        captured = {}

        class _Adapter:
            scheduler = SimpleNamespace(train_timesteps=torch.tensor([35]))

            def rollout(self):
                return None

            def inference(self, **kwargs):
                captured.update(kwargs)
                return []

        trainer = XOPDTrainer.__new__(XOPDTrainer)
        trainer.adapter = _Adapter()
        trainer.training_args = _AttrDict(
            xopd_resample_steps_per_batch=False,
            num_inference_steps=28,
            num_batches_per_epoch=1,
            xopd_train_steps=None,
            num_xopd_steps=None,
        )
        trainer._is_popd = True
        trainer._is_marginal_cfm = False
        trainer._is_ode = False
        trainer._cross_vae = False
        trainer.log_args = SimpleNamespace(verbose=False)
        trainer.accelerator = SimpleNamespace(is_local_main_process=False)
        trainer.epoch = 0
        trainer.autocast = nullcontext
        trainer._make_train_iter = lambda: iter([{}])
        trainer._maybe_offload_samples_to_cpu = lambda samples: None

        self.assertEqual(trainer.sample(), [])
        self.assertEqual(captured["trajectory_indices"], SCHEDULER_TRAIN_INDICES)

        captured.clear()
        trainer._is_popd = False
        self.assertEqual(trainer.sample(), [])
        self.assertEqual(captured["trajectory_indices"], SCHEDULER_TRAIN_INDICES)


class TestPOPDCallbackOffload(unittest.TestCase):
    def test_sample_to_moves_callback_tensors_in_extra_kwargs(self) -> None:
        sample = BaseSample(
            extra_kwargs={
                "next_latents_mean": torch.ones(2),
                "std_dev_t": torch.ones(1),
                "dt": -torch.ones(1),
            }
        )
        sample.to("meta")
        self.assertEqual(sample.extra_kwargs["next_latents_mean"].device.type, "meta")
        self.assertEqual(sample.extra_kwargs["std_dev_t"].device.type, "meta")
        self.assertEqual(sample.extra_kwargs["dt"].device.type, "meta")


if __name__ == "__main__":
    unittest.main()
