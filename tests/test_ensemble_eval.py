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

"""Unit tests for multi-checkpoint ensemble evaluation helpers."""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any, Dict, List
from unittest.mock import MagicMock

import torch

from flow_factory.hparams.training_args import EnsembleEvalTrainingArguments
from flow_factory.trainers.ensemble_eval.common import (
    ensemble_forward_step,
    load_checkpoints,
    normalize_checkpoint_weights,
)
from flow_factory.trainers.registry import get_trainer_class


class TestNormalizeCheckpointWeights(unittest.TestCase):
    def test_uniform_when_none(self) -> None:
        weights = normalize_checkpoint_weights(None, 3)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(sum(weights), 1.0)
        for w in weights:
            self.assertAlmostEqual(w, 1.0 / 3.0)

    def test_normalizes_provided_weights(self) -> None:
        weights = normalize_checkpoint_weights([1.0, 3.0], 2)
        torch.testing.assert_close(torch.tensor(weights), torch.tensor([0.25, 0.75]))

    def test_rejects_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            normalize_checkpoint_weights([1.0], 2)

    def test_rejects_negative_weights(self) -> None:
        with self.assertRaises(ValueError):
            normalize_checkpoint_weights([-1.0, 2.0], 2)

    def test_rejects_zero_sum(self) -> None:
        with self.assertRaises(ValueError):
            normalize_checkpoint_weights([0.0, 0.0], 2)


class TestEnsembleEvalTrainingArgumentsPostInit(unittest.TestCase):
    def test_requires_checkpoint_paths(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleEvalTrainingArguments(
                checkpoint_paths=[],
                unique_sample_num_per_epoch=1,
                group_size=1,
                per_device_batch_size=1,
            )

    def test_rejects_mismatched_weights(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleEvalTrainingArguments(
                checkpoint_paths=["/tmp/a", "/tmp/b"],
                checkpoint_weights=[1.0],
                unique_sample_num_per_epoch=1,
                group_size=1,
                per_device_batch_size=1,
            )


class TestLoadCheckpoints(unittest.TestCase):
    def test_empty_paths_raises(self) -> None:
        adapter = MagicMock()
        with self.assertRaises(ValueError):
            load_checkpoints(adapter, [], "cpu")


class _MockSchedulerOutput:
    def __init__(self, noise_pred: torch.Tensor, next_latents: torch.Tensor) -> None:
        self.noise_pred = noise_pred
        self.next_latents = next_latents
        self.log_prob = None


class _MockAdapter:
    def __init__(self, preds_by_name: Dict[str, torch.Tensor]) -> None:
        self._preds_by_name = preds_by_name
        self.scheduler = MagicMock()
        self.scheduler.step.side_effect = self._scheduler_step
        self._sched_cache = (
            frozenset(
                {
                    "noise_pred",
                    "timestep",
                    "latents",
                    "timestep_next",
                    "next_latents",
                    "compute_log_prob",
                    "log_prob_reduction",
                    "return_dict",
                    "return_kwargs",
                    "noise_level",
                }
            ),
            False,
        )

    def _scheduler_step(self, **kwargs: Any) -> _MockSchedulerOutput:
        latents = kwargs["latents"]
        noise_pred = kwargs["noise_pred"]
        return _MockSchedulerOutput(
            noise_pred=noise_pred,
            next_latents=latents + noise_pred,
        )

    @contextmanager
    def use_named_parameters(self, name: str):
        self._active_name = name
        yield

    def forward(self, **kwargs: Any) -> _MockSchedulerOutput:
        del kwargs
        pred = self._preds_by_name[self._active_name]
        return _MockSchedulerOutput(noise_pred=pred, next_latents=pred)


class TestEnsembleForwardStep(unittest.TestCase):
    def test_weighted_blend_before_scheduler(self) -> None:
        preds = {
            "eval_ckpt_0": torch.tensor([1.0]),
            "eval_ckpt_1": torch.tensor([3.0]),
        }
        adapter = _MockAdapter(preds)
        out = ensemble_forward_step(
            adapter,
            ["eval_ckpt_0", "eval_ckpt_1"],
            [0.25, 0.75],
            {
                "t": torch.tensor(500),
                "t_next": torch.tensor(400),
                "latents": torch.tensor([0.0]),
                "compute_log_prob": False,
                "return_kwargs": ["next_latents", "noise_pred"],
            },
            adapter._sched_cache,
            base_forward=adapter.forward,
        )
        torch.testing.assert_close(out.noise_pred, torch.tensor([2.5]))
        torch.testing.assert_close(out.next_latents, torch.tensor([2.5]))
        adapter.scheduler.step.assert_called_once()
        call_kwargs = adapter.scheduler.step.call_args.kwargs
        torch.testing.assert_close(call_kwargs["noise_pred"], torch.tensor([2.5]))


class TestEnsembleForwardStepWithPatchedForward(unittest.TestCase):
    def test_uses_base_forward_not_patched_forward(self) -> None:
        preds = {
            "eval_ckpt_0": torch.tensor([1.0]),
            "eval_ckpt_1": torch.tensor([3.0]),
        }
        adapter = _MockAdapter(preds)
        real_forward = adapter.forward

        def patched_forward(**kwargs: Any) -> _MockSchedulerOutput:
            raise AssertionError("patched_forward must not be called from ensemble_forward_step")

        adapter.forward = patched_forward  # type: ignore[method-assign]

        out = ensemble_forward_step(
            adapter,
            ["eval_ckpt_0", "eval_ckpt_1"],
            [0.5, 0.5],
            {
                "t": torch.tensor(500),
                "t_next": torch.tensor(400),
                "latents": torch.tensor([0.0]),
                "compute_log_prob": False,
                "return_kwargs": ["noise_pred"],
            },
            adapter._sched_cache,
            base_forward=real_forward,
        )
        torch.testing.assert_close(out.noise_pred, torch.tensor([2.0]))


class TestEnsembleEvalRegistry(unittest.TestCase):
    def test_trainer_registered(self) -> None:
        cls = get_trainer_class("ensemble-eval")
        self.assertEqual(cls.__name__, "EnsembleEvalTrainer")


if __name__ == "__main__":
    unittest.main()
