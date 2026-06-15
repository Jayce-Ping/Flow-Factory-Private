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
    MERGED_SNAPSHOT_NAME,
    _dynamic_kl_weights,
    build_merged_lora_snapshot,
    ensemble_forward_step,
    load_checkpoints,
    normalize_checkpoint_weights,
    pcgrad_blend_noise_preds,
    pcgrad_blend_noise_preds_channelwise,
    pcgrad_blend_noise_preds_normalized,
    ties_blend_deltas,
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
    def test_allows_empty_checkpoint_paths(self) -> None:
        args = EnsembleEvalTrainingArguments(
            checkpoint_paths=[],
            unique_sample_num_per_epoch=1,
            group_size=1,
            per_device_batch_size=1,
        )
        self.assertEqual(args.checkpoint_paths, [])

    def test_rejects_weights_when_checkpoint_paths_empty(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleEvalTrainingArguments(
                checkpoint_paths=[],
                checkpoint_weights=[1.0],
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

    def test_rejects_invalid_blend_mode(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleEvalTrainingArguments(
                checkpoint_paths=["/tmp/a"],
                ensemble_blend_mode="invalid",  # type: ignore[arg-type]
                unique_sample_num_per_epoch=1,
                group_size=1,
                per_device_batch_size=1,
            )

    def test_rejects_non_positive_pcgrad_eps(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleEvalTrainingArguments(
                checkpoint_paths=["/tmp/a"],
                pcgrad_eps=0.0,
                unique_sample_num_per_epoch=1,
                group_size=1,
                per_device_batch_size=1,
            )

    def test_allows_weight_merge_with_uniform(self) -> None:
        args = EnsembleEvalTrainingArguments(
            checkpoint_paths=["/tmp/a", "/tmp/b"],
            ensemble_blend_mode="weight_merge",
            ensemble_blend_weighting="uniform",
            unique_sample_num_per_epoch=1,
            group_size=1,
            per_device_batch_size=1,
        )
        self.assertEqual(args.ensemble_blend_mode, "weight_merge")

    def test_rejects_weight_merge_with_kl_weighting(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleEvalTrainingArguments(
                checkpoint_paths=["/tmp/a", "/tmp/b"],
                ensemble_blend_mode="weight_merge",
                ensemble_blend_weighting="kl",
                unique_sample_num_per_epoch=1,
                group_size=1,
                per_device_batch_size=1,
            )


class TestBuildMergedLoraSnapshot(unittest.TestCase):
    class _MergeAdapter:
        """Minimal adapter stub exercising the named-parameter snapshot API."""

        def __init__(self, snapshots: Dict[str, List[torch.Tensor]]) -> None:
            self._snap: Dict[str, Any] = dict(snapshots)

        def get_named_parameters(self, name: str) -> List[torch.Tensor]:
            return self._snap[name]

        def list_named_parameters(self) -> List[str]:
            return list(self._snap)

        def add_named_parameters(
            self, name: str, device: Any = None, overwrite: bool = True
        ) -> None:
            self._snap[name] = None

        def update_named_parameters(self, name: str, new_parameters: Any = None) -> None:
            self._snap[name] = list(new_parameters)

    def test_uniform_average(self) -> None:
        adapter = self._MergeAdapter(
            {
                "eval_ckpt_0": [torch.tensor([2.0, 0.0]), torch.tensor([[4.0]])],
                "eval_ckpt_1": [torch.tensor([0.0, 4.0]), torch.tensor([[0.0]])],
            }
        )
        name = build_merged_lora_snapshot(
            adapter, ["eval_ckpt_0", "eval_ckpt_1"], [0.5, 0.5]
        )
        self.assertEqual(name, MERGED_SNAPSHOT_NAME)
        merged = adapter.get_named_parameters(MERGED_SNAPSHOT_NAME)
        torch.testing.assert_close(merged[0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(merged[1], torch.tensor([[2.0]]))

    def test_weighted_average(self) -> None:
        adapter = self._MergeAdapter(
            {
                "eval_ckpt_0": [torch.tensor([10.0])],
                "eval_ckpt_1": [torch.tensor([0.0])],
            }
        )
        build_merged_lora_snapshot(adapter, ["eval_ckpt_0", "eval_ckpt_1"], [0.25, 0.75])
        merged = adapter.get_named_parameters(MERGED_SNAPSHOT_NAME)
        torch.testing.assert_close(merged[0], torch.tensor([2.5]))

    def test_preserves_dtype(self) -> None:
        adapter = self._MergeAdapter(
            {
                "eval_ckpt_0": [torch.tensor([2.0, 4.0], dtype=torch.bfloat16)],
                "eval_ckpt_1": [torch.tensor([4.0, 8.0], dtype=torch.bfloat16)],
            }
        )
        build_merged_lora_snapshot(adapter, ["eval_ckpt_0", "eval_ckpt_1"], [0.5, 0.5])
        merged = adapter.get_named_parameters(MERGED_SNAPSHOT_NAME)
        self.assertEqual(merged[0].dtype, torch.bfloat16)
        torch.testing.assert_close(merged[0], torch.tensor([3.0, 6.0], dtype=torch.bfloat16))

    def test_rejects_length_mismatch(self) -> None:
        adapter = self._MergeAdapter({"eval_ckpt_0": [torch.tensor([1.0])]})
        with self.assertRaises(ValueError):
            build_merged_lora_snapshot(adapter, ["eval_ckpt_0"], [0.5, 0.5])

    def test_rejects_empty(self) -> None:
        adapter = self._MergeAdapter({})
        with self.assertRaises(ValueError):
            build_merged_lora_snapshot(adapter, [], [])


class TestPcgradBlendNoisePreds(unittest.TestCase):
    def test_single_checkpoint_returns_input(self) -> None:
        pred = torch.tensor([1.0, 2.0])
        out = pcgrad_blend_noise_preds([pred])
        torch.testing.assert_close(out, pred)

    def test_two_conflict_cancels(self) -> None:
        out = pcgrad_blend_noise_preds(
            [torch.tensor([2.0]), torch.tensor([-2.0])],
        )
        torch.testing.assert_close(out, torch.tensor([0.0]))

    def test_two_non_conflict_sums(self) -> None:
        out = pcgrad_blend_noise_preds(
            [torch.tensor([2.0]), torch.tensor([1.0])],
        )
        torch.testing.assert_close(out, torch.tensor([3.0]))

    def test_batchwise_conflict_per_sample(self) -> None:
        preds = [
            torch.tensor([[2.0], [-1.0]]),
            torch.tensor([[-2.0], [1.0]]),
        ]
        out = pcgrad_blend_noise_preds(preds)
        torch.testing.assert_close(out, torch.tensor([[0.0], [0.0]]))

    def test_rejects_empty_sequence(self) -> None:
        with self.assertRaises(ValueError):
            pcgrad_blend_noise_preds([])

    def test_rejects_non_positive_eps(self) -> None:
        with self.assertRaises(ValueError):
            pcgrad_blend_noise_preds([torch.tensor([1.0])], eps=0.0)


class TestPcgradBlendNoisePredsNormalized(unittest.TestCase):
    def test_single_returns_weighted_input(self) -> None:
        out = pcgrad_blend_noise_preds_normalized([torch.tensor([3.0])], [0.5])
        torch.testing.assert_close(out, torch.tensor([1.5]))

    def test_two_conflict_cancels(self) -> None:
        out = pcgrad_blend_noise_preds_normalized(
            [torch.tensor([2.0]), torch.tensor([-2.0])], [0.5, 0.5]
        )
        torch.testing.assert_close(out, torch.tensor([0.0]))

    def test_two_non_conflict_equals_weighted(self) -> None:
        # Aligned directions: result restores magnitude+weight = weighted blend.
        out = pcgrad_blend_noise_preds_normalized(
            [torch.tensor([2.0]), torch.tensor([1.0])], [0.5, 0.5]
        )
        torch.testing.assert_close(out, torch.tensor([1.5]))

    def test_magnitude_invariant_geometry(self) -> None:
        # Two 2D vectors, one with much larger magnitude but partially conflicting.
        # Normalized projection works on unit directions so the large-norm vector
        # does not dominate the conflict geometry.
        v0 = torch.tensor([[10.0, 0.0]])
        v1 = torch.tensor([[-1.0, 1.0]])
        out = pcgrad_blend_noise_preds_normalized([v0, v1], [0.5, 0.5])
        # u0=[1,0], u1=[-1,1]/sqrt(2). dot(u0,u1)<0 -> conflict.
        # pc0 = u0 - (u0·u1)u1 ; pc1 = u1 - (u1·u0)u0. Recombine w*n*pc.
        cos = -1.0 / (2.0**0.5)
        u0 = torch.tensor([[1.0, 0.0]])
        u1 = torch.tensor([[-1.0, 1.0]]) / (2.0**0.5)
        pc0 = u0 - cos * u1
        pc1 = u1 - cos * u0
        expected = 0.5 * 10.0 * pc0 + 0.5 * (2.0**0.5) * pc1
        torch.testing.assert_close(out, expected)

    def test_rejects_empty_sequence(self) -> None:
        with self.assertRaises(ValueError):
            pcgrad_blend_noise_preds_normalized([], [])

    def test_rejects_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            pcgrad_blend_noise_preds_normalized([torch.tensor([1.0]), torch.tensor([2.0])], [1.0])

    def test_rejects_non_positive_eps(self) -> None:
        with self.assertRaises(ValueError):
            pcgrad_blend_noise_preds_normalized([torch.tensor([1.0])], [1.0], eps=0.0)


class TestTiesBlendDeltas(unittest.TestCase):
    def test_single_returns_input_delta(self) -> None:
        out = ties_blend_deltas([torch.tensor([[2.0, 0.0, -3.0]])], [1.0])
        torch.testing.assert_close(out, torch.tensor([[2.0, 0.0, -3.0]]))

    def test_sign_vote_drops_disagreeing(self) -> None:
        tau0 = torch.tensor([[2.0, 1.0, -1.0]])
        tau1 = torch.tensor([[-1.0, 2.0, 1.0]])
        out = ties_blend_deltas([tau0, tau1], [0.5, 0.5])
        # weighted_sum=[0.5,1.5,0.0] -> gamma=[1,1,0].
        # e0: only tau0 agrees -> 0.5*2/0.5 = 2.0
        # e1: both agree -> (0.5*1+0.5*2)/1.0 = 1.5
        # e2: gamma=0 -> no agreement -> 0.0
        torch.testing.assert_close(out, torch.tensor([[2.0, 1.5, 0.0]]))

    def test_no_disagreement_equals_weighted(self) -> None:
        tau0 = torch.tensor([[2.0, 4.0]])
        tau1 = torch.tensor([[1.0, 2.0]])
        out = ties_blend_deltas([tau0, tau1], [0.5, 0.5])
        torch.testing.assert_close(out, torch.tensor([[1.5, 3.0]]))

    def test_density_trim_keeps_top_magnitude(self) -> None:
        # density=0.5 over 4 elements keeps top-2 by |tau| per sample.
        tau0 = torch.tensor([[4.0, 0.1, 3.0, 0.2]])
        out = ties_blend_deltas([tau0], [1.0], density=0.5)
        # Single delta: keeps the two largest-magnitude entries, zeros the rest.
        torch.testing.assert_close(out, torch.tensor([[4.0, 0.0, 3.0, 0.0]]))

    def test_rejects_empty_sequence(self) -> None:
        with self.assertRaises(ValueError):
            ties_blend_deltas([], [])

    def test_rejects_invalid_density(self) -> None:
        with self.assertRaises(ValueError):
            ties_blend_deltas([torch.tensor([1.0])], [1.0], density=0.0)


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

    def test_pcgrad_blend_before_scheduler(self) -> None:
        preds = {
            "eval_ckpt_0": torch.tensor([2.0]),
            "eval_ckpt_1": torch.tensor([-2.0]),
        }
        adapter = _MockAdapter(preds)
        out = ensemble_forward_step(
            adapter,
            ["eval_ckpt_0", "eval_ckpt_1"],
            [0.5, 0.5],
            {
                "t": torch.tensor(500),
                "t_next": torch.tensor(400),
                "latents": torch.tensor([0.0]),
                "compute_log_prob": False,
                "return_kwargs": ["next_latents", "noise_pred"],
            },
            adapter._sched_cache,
            base_forward=adapter.forward,
            blend_mode="pcgrad",
        )
        torch.testing.assert_close(out.noise_pred, torch.tensor([0.0]))
        call_kwargs = adapter.scheduler.step.call_args.kwargs
        torch.testing.assert_close(call_kwargs["noise_pred"], torch.tensor([0.0]))

    def test_rejects_invalid_blend_mode(self) -> None:
        preds = {
            "eval_ckpt_0": torch.tensor([1.0]),
            "eval_ckpt_1": torch.tensor([3.0]),
        }
        adapter = _MockAdapter(preds)
        with self.assertRaises(ValueError):
            ensemble_forward_step(
                adapter,
                ["eval_ckpt_0", "eval_ckpt_1"],
                [0.5, 0.5],
                {
                    "t": torch.tensor(500),
                    "latents": torch.tensor([0.0]),
                    "compute_log_prob": False,
                },
                adapter._sched_cache,
                base_forward=adapter.forward,
                blend_mode="invalid",  # type: ignore[arg-type]
            )


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


# ---------------------------------------------------------------------------
# PCGrad Channelwise Tests
# ---------------------------------------------------------------------------


class TestPcgradBlendChannelwise(unittest.TestCase):
    """Tests for pcgrad_blend_noise_preds_channelwise."""

    def test_single_checkpoint_returns_input(self) -> None:
        pred = torch.randn(2, 3, 4, 4)
        out = pcgrad_blend_noise_preds_channelwise([pred])
        torch.testing.assert_close(out, pred)

    def test_4d_per_channel_conflict(self) -> None:
        """4D tensor: channel 0 conflicts, channel 1 does not."""
        # Shape (B=1, C=2, H=1, W=1)
        # Channel 0: pred0=+2, pred1=-2 → conflict (dot<0) → project to 0
        # Channel 1: pred0=+1, pred1=+3 → no conflict → sum = 4
        pred0 = torch.tensor([[[[2.0]], [[1.0]]]])  # (1, 2, 1, 1)
        pred1 = torch.tensor([[[[-2.0]], [[3.0]]]])  # (1, 2, 1, 1)
        out = pcgrad_blend_noise_preds_channelwise([pred0, pred1])
        # Channel 0: conflicting → both projected to 0, sum=0
        # Channel 1: non-conflicting → sum = 1 + 3 = 4
        torch.testing.assert_close(out, torch.tensor([[[[0.0]], [[4.0]]]]))

    def test_3d_per_token_conflict(self) -> None:
        """3D tensor: token 0 conflicts, token 1 does not."""
        # Shape (B=1, seq_len=2, feat=2)
        # Token 0: pred0=[2, 1], pred1=[-2, -1] → dot = -4 - 1 = -5 < 0 → conflict
        # Token 1: pred0=[1, 1], pred1=[1, 1] → dot = 1 + 1 = 2 > 0 → no conflict
        pred0 = torch.tensor([[[2.0, 1.0], [1.0, 1.0]]])  # (1, 2, 2)
        pred1 = torch.tensor([[[-2.0, -1.0], [1.0, 1.0]]])  # (1, 2, 2)
        out = pcgrad_blend_noise_preds_channelwise([pred0, pred1])
        # Token 0: full conflict (opposite directions) → both project to 0
        # Token 1: no conflict → sum = [2, 2]
        torch.testing.assert_close(out, torch.tensor([[[0.0, 0.0], [2.0, 2.0]]]))

    def test_non_conflict_equals_sum(self) -> None:
        """When all channels align, result equals simple sum (same as weighted)."""
        pred0 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
        pred1 = torch.tensor([[[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]]])
        out = pcgrad_blend_noise_preds_channelwise([pred0, pred1])
        expected = pred0 + pred1
        torch.testing.assert_close(out, expected)

    def test_rejects_2d_tensor(self) -> None:
        """ndim < 3 should raise ValueError."""
        with self.assertRaises(ValueError):
            pcgrad_blend_noise_preds_channelwise(
                [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]
            )

    def test_rejects_empty_sequence(self) -> None:
        with self.assertRaises(ValueError):
            pcgrad_blend_noise_preds_channelwise([])


# ---------------------------------------------------------------------------
# PCGrad Residual Mode Tests (via ensemble_forward_step)
# ---------------------------------------------------------------------------


class _MockAdapterWithRef(_MockAdapter):
    """Mock adapter that also supports use_ref_parameters for residual mode."""

    def __init__(
        self,
        preds_by_name: Dict[str, torch.Tensor],
        ref_pred: torch.Tensor,
    ) -> None:
        super().__init__(preds_by_name)
        self._ref_pred = ref_pred

    @contextmanager
    def use_ref_parameters(self):
        """Temporarily set active prediction to the reference (pretrained) pred."""
        prev = getattr(self, "_active_name", None)
        self._active_name = "__ref__"
        yield
        self._active_name = prev

    def forward(self, **kwargs: Any) -> _MockSchedulerOutput:
        del kwargs
        if self._active_name == "__ref__":
            pred = self._ref_pred
        else:
            pred = self._preds_by_name[self._active_name]
        return _MockSchedulerOutput(noise_pred=pred, next_latents=pred)


class TestEnsembleForwardStepResidualMode(unittest.TestCase):
    """Tests for blend_mode='pcgrad_residual' via ensemble_forward_step."""

    def test_residual_conflicting_deltas(self) -> None:
        """When checkpoint deltas conflict, PCGrad projects them."""
        # ref_pred = [10.0], ckpt_0 = [12.0], ckpt_1 = [8.0]
        # delta_0 = [2.0], delta_1 = [-2.0] → conflict!
        # After PCGrad on deltas: both project to 0
        # Result = ref + 0 = [10.0]
        ref_pred = torch.tensor([10.0])
        preds = {
            "eval_ckpt_0": torch.tensor([12.0]),
            "eval_ckpt_1": torch.tensor([8.0]),
        }
        adapter = _MockAdapterWithRef(preds, ref_pred)
        out = ensemble_forward_step(
            adapter,
            ["eval_ckpt_0", "eval_ckpt_1"],
            [0.5, 0.5],
            {
                "t": torch.tensor(500),
                "t_next": torch.tensor(400),
                "latents": torch.tensor([0.0]),
                "compute_log_prob": False,
                "return_kwargs": ["next_latents", "noise_pred"],
            },
            adapter._sched_cache,
            base_forward=adapter.forward,
            blend_mode="pcgrad_residual",
        )
        # Weighted deltas: 0.5*[2.0]=[1.0] and 0.5*[-2.0]=[-1.0] → conflict → both → 0
        # Result = ref + 0 = 10.0
        torch.testing.assert_close(out.noise_pred, torch.tensor([10.0]))

    def test_residual_aligned_deltas_equals_weighted(self) -> None:
        """When deltas are aligned, residual mode equals weighted_sum."""
        # ref = [0.0], ckpt_0 = [2.0], ckpt_1 = [4.0]
        # delta_0 = [2.0], delta_1 = [4.0] → aligned (both positive)
        # weighted deltas: 0.5*[2.0] + 0.5*[4.0] = [1.0] + [2.0] = [3.0]
        # result = ref + [3.0] = [3.0]
        # This equals weighted_sum: 0.5*[2.0] + 0.5*[4.0] = [3.0]
        ref_pred = torch.tensor([0.0])
        preds = {
            "eval_ckpt_0": torch.tensor([2.0]),
            "eval_ckpt_1": torch.tensor([4.0]),
        }
        adapter = _MockAdapterWithRef(preds, ref_pred)
        out = ensemble_forward_step(
            adapter,
            ["eval_ckpt_0", "eval_ckpt_1"],
            [0.5, 0.5],
            {
                "t": torch.tensor(500),
                "t_next": torch.tensor(400),
                "latents": torch.tensor([0.0]),
                "compute_log_prob": False,
                "return_kwargs": ["next_latents", "noise_pred"],
            },
            adapter._sched_cache,
            base_forward=adapter.forward,
            blend_mode="pcgrad_residual",
        )
        torch.testing.assert_close(out.noise_pred, torch.tensor([3.0]))

    def test_residual_single_checkpoint(self) -> None:
        """Single checkpoint: result = ref + weight * delta = ckpt_pred * weight + ref * (1 - weight)."""
        ref_pred = torch.tensor([5.0])
        preds = {"eval_ckpt_0": torch.tensor([10.0])}
        adapter = _MockAdapterWithRef(preds, ref_pred)
        out = ensemble_forward_step(
            adapter,
            ["eval_ckpt_0"],
            [1.0],
            {
                "t": torch.tensor(500),
                "t_next": torch.tensor(400),
                "latents": torch.tensor([0.0]),
                "compute_log_prob": False,
                "return_kwargs": ["next_latents", "noise_pred"],
            },
            adapter._sched_cache,
            base_forward=adapter.forward,
            blend_mode="pcgrad_residual",
        )
        # delta = [10.0 - 5.0] = [5.0], single checkpoint → no PCGrad needed
        # result = ref + 1.0 * delta = 5.0 + 5.0 = 10.0
        torch.testing.assert_close(out.noise_pred, torch.tensor([10.0]))


class TestEnsembleForwardStepChannelwiseMode(unittest.TestCase):
    """Tests for blend_mode='pcgrad_channelwise' via ensemble_forward_step."""

    def test_channelwise_4d_per_channel_conflict(self) -> None:
        """4D per-channel conflict detection through ensemble_forward_step."""
        # (B=1, C=2, H=1, W=1): channel 0 conflicts, channel 1 aligned
        preds = {
            "eval_ckpt_0": torch.tensor([[[[4.0]], [[2.0]]]]),
            "eval_ckpt_1": torch.tensor([[[[-4.0]], [[6.0]]]]),
        }
        adapter = _MockAdapter(preds)
        out = ensemble_forward_step(
            adapter,
            ["eval_ckpt_0", "eval_ckpt_1"],
            [0.5, 0.5],
            {
                "t": torch.tensor(500),
                "t_next": torch.tensor(400),
                "latents": torch.zeros(1, 2, 1, 1),
                "compute_log_prob": False,
                "return_kwargs": ["next_latents", "noise_pred"],
            },
            adapter._sched_cache,
            base_forward=adapter.forward,
            blend_mode="pcgrad_channelwise",
        )
        # Scaled: 0.5*[4]=[2], 0.5*[-4]=[-2] in ch0 → conflict → 0
        # Scaled: 0.5*[2]=[1], 0.5*[6]=[3] in ch1 → aligned → 4
        torch.testing.assert_close(out.noise_pred, torch.tensor([[[[0.0]], [[4.0]]]]))


class TestEnsembleForwardStepResidualNormalizedMode(unittest.TestCase):
    """Tests for blend_mode='pcgrad_residual_normalized' via ensemble_forward_step."""

    def _run(self, ref_pred, preds, names, weights, latents):
        adapter = _MockAdapterWithRef(preds, ref_pred)
        return ensemble_forward_step(
            adapter,
            names,
            weights,
            {
                "t": torch.tensor(500),
                "t_next": torch.tensor(400),
                "latents": latents,
                "compute_log_prob": False,
                "return_kwargs": ["next_latents", "noise_pred"],
            },
            adapter._sched_cache,
            base_forward=adapter.forward,
            blend_mode="pcgrad_residual_normalized",
        )

    def test_conflicting_deltas(self) -> None:
        # ref=10, ckpt deltas +2/-2 -> opposite unit directions cancel -> v_b.
        out = self._run(
            torch.tensor([10.0]),
            {"eval_ckpt_0": torch.tensor([12.0]), "eval_ckpt_1": torch.tensor([8.0])},
            ["eval_ckpt_0", "eval_ckpt_1"],
            [0.5, 0.5],
            torch.tensor([0.0]),
        )
        torch.testing.assert_close(out.noise_pred, torch.tensor([10.0]))

    def test_aligned_deltas_equals_weighted(self) -> None:
        # ref=0, deltas +2/+4 aligned -> restores magnitude+weight = weighted.
        out = self._run(
            torch.tensor([0.0]),
            {"eval_ckpt_0": torch.tensor([2.0]), "eval_ckpt_1": torch.tensor([4.0])},
            ["eval_ckpt_0", "eval_ckpt_1"],
            [0.5, 0.5],
            torch.tensor([0.0]),
        )
        torch.testing.assert_close(out.noise_pred, torch.tensor([3.0]))


class TestEnsembleForwardStepResidualChannelwiseMode(unittest.TestCase):
    """Tests for blend_mode='pcgrad_residual_channelwise' via ensemble_forward_step."""

    def test_per_channel_delta_conflict(self) -> None:
        ref_pred = torch.zeros(1, 2, 1, 1)
        preds = {
            "eval_ckpt_0": torch.tensor([[[[2.0]], [[1.0]]]]),
            "eval_ckpt_1": torch.tensor([[[[-2.0]], [[3.0]]]]),
        }
        adapter = _MockAdapterWithRef(preds, ref_pred)
        out = ensemble_forward_step(
            adapter,
            ["eval_ckpt_0", "eval_ckpt_1"],
            [0.5, 0.5],
            {
                "t": torch.tensor(500),
                "t_next": torch.tensor(400),
                "latents": torch.zeros(1, 2, 1, 1),
                "compute_log_prob": False,
                "return_kwargs": ["next_latents", "noise_pred"],
            },
            adapter._sched_cache,
            base_forward=adapter.forward,
            blend_mode="pcgrad_residual_channelwise",
        )
        # ch0: deltas +2/-2 -> conflict -> 0; ch1: deltas +1/+3 aligned -> 0.5+1.5=2.
        torch.testing.assert_close(out.noise_pred, torch.tensor([[[[0.0]], [[2.0]]]]))


class TestEnsembleForwardStepTiesMode(unittest.TestCase):
    """Tests for blend_mode='ties' via ensemble_forward_step."""

    def test_sign_vote(self) -> None:
        ref_pred = torch.zeros(1, 3)
        preds = {
            "eval_ckpt_0": torch.tensor([[2.0, 1.0, -1.0]]),
            "eval_ckpt_1": torch.tensor([[-1.0, 2.0, 1.0]]),
        }
        adapter = _MockAdapterWithRef(preds, ref_pred)
        out = ensemble_forward_step(
            adapter,
            ["eval_ckpt_0", "eval_ckpt_1"],
            [0.5, 0.5],
            {
                "t": torch.tensor(500),
                "t_next": torch.tensor(400),
                "latents": torch.zeros(1, 3),
                "compute_log_prob": False,
                "return_kwargs": ["next_latents", "noise_pred"],
            },
            adapter._sched_cache,
            base_forward=adapter.forward,
            blend_mode="ties",
        )
        torch.testing.assert_close(out.noise_pred, torch.tensor([[2.0, 1.5, 0.0]]))

    def test_no_disagreement_equals_weighted(self) -> None:
        ref_pred = torch.zeros(1, 2)
        preds = {
            "eval_ckpt_0": torch.tensor([[2.0, 4.0]]),
            "eval_ckpt_1": torch.tensor([[1.0, 2.0]]),
        }
        adapter = _MockAdapterWithRef(preds, ref_pred)
        out = ensemble_forward_step(
            adapter,
            ["eval_ckpt_0", "eval_ckpt_1"],
            [0.5, 0.5],
            {
                "t": torch.tensor(500),
                "t_next": torch.tensor(400),
                "latents": torch.zeros(1, 2),
                "compute_log_prob": False,
                "return_kwargs": ["next_latents", "noise_pred"],
            },
            adapter._sched_cache,
            base_forward=adapter.forward,
            blend_mode="ties",
        )
        torch.testing.assert_close(out.noise_pred, torch.tensor([[1.5, 3.0]]))

    def test_single_checkpoint_returns_v0(self) -> None:
        ref_pred = torch.tensor([5.0])
        preds = {"eval_ckpt_0": torch.tensor([10.0])}
        adapter = _MockAdapterWithRef(preds, ref_pred)
        out = ensemble_forward_step(
            adapter,
            ["eval_ckpt_0"],
            [1.0],
            {
                "t": torch.tensor(500),
                "t_next": torch.tensor(400),
                "latents": torch.tensor([0.0]),
                "compute_log_prob": False,
                "return_kwargs": ["next_latents", "noise_pred"],
            },
            adapter._sched_cache,
            base_forward=adapter.forward,
            blend_mode="ties",
        )
        torch.testing.assert_close(out.noise_pred, torch.tensor([10.0]))


# ---------------------------------------------------------------------------
# Training Arguments Validation for blend modes
# ---------------------------------------------------------------------------


class TestBlendModeValidation(unittest.TestCase):
    def test_accepts_all_valid_blend_modes(self) -> None:
        for mode in (
            "weighted",
            "pcgrad",
            "pcgrad_residual",
            "pcgrad_channelwise",
            "pcgrad_normalized",
            "pcgrad_residual_normalized",
            "pcgrad_residual_channelwise",
            "ties",
        ):
            args = EnsembleEvalTrainingArguments(
                checkpoint_paths=["/tmp/a"],
                ensemble_blend_mode=mode,
                unique_sample_num_per_epoch=1,
                group_size=1,
                per_device_batch_size=1,
            )
            self.assertEqual(args.ensemble_blend_mode, mode)

    def test_rejects_invalid_blend_mode(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleEvalTrainingArguments(
                checkpoint_paths=["/tmp/a"],
                ensemble_blend_mode="invalid",  # type: ignore[arg-type]
                unique_sample_num_per_epoch=1,
                group_size=1,
                per_device_batch_size=1,
            )

    def test_accepts_valid_ties_density(self) -> None:
        args = EnsembleEvalTrainingArguments(
            checkpoint_paths=["/tmp/a"],
            ensemble_blend_mode="ties",
            ties_density=0.5,
            unique_sample_num_per_epoch=1,
            group_size=1,
            per_device_batch_size=1,
        )
        self.assertEqual(args.ties_density, 0.5)

    def test_rejects_invalid_ties_density(self) -> None:
        for bad in (0.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                EnsembleEvalTrainingArguments(
                    checkpoint_paths=["/tmp/a"],
                    ties_density=bad,
                    unique_sample_num_per_epoch=1,
                    group_size=1,
                    per_device_batch_size=1,
                )


# ---------------------------------------------------------------------------
# Dynamic KL / inverse-KL teacher weighting
# ---------------------------------------------------------------------------


class TestDynamicKlWeights(unittest.TestCase):
    def test_kl_upweights_larger_norm(self) -> None:
        taus = [torch.tensor([[3.0, 0.0]]), torch.tensor([[0.0, 1.0]])]  # D = 9, 1
        w = _dynamic_kl_weights(taus, [0.5, 0.5], alpha=1.0, eps=1e-8)
        torch.testing.assert_close(w[0], torch.tensor([0.9]))
        torch.testing.assert_close(w[1], torch.tensor([0.1]))

    def test_kl_inv_downweights_larger_norm(self) -> None:
        taus = [torch.tensor([[3.0, 0.0]]), torch.tensor([[0.0, 1.0]])]  # D = 9, 1
        w = _dynamic_kl_weights(taus, [0.5, 0.5], alpha=-1.0, eps=1e-8)
        # raw = 0.5/9, 0.5 -> normalized 0.1, 0.9
        torch.testing.assert_close(w[0], torch.tensor([0.1]))
        torch.testing.assert_close(w[1], torch.tensor([0.9]))

    def test_sums_to_one_per_sample(self) -> None:
        taus = [torch.randn(4, 3, 2, 2), torch.randn(4, 3, 2, 2), torch.randn(4, 3, 2, 2)]
        w = _dynamic_kl_weights(taus, [1 / 3, 1 / 3, 1 / 3], alpha=1.0, eps=1e-8)
        total = w[0] + w[1] + w[2]
        torch.testing.assert_close(total, torch.ones(4))

    def test_per_sample_routing(self) -> None:
        # sample 0 -> teacher 0 deviates; sample 1 -> teacher 1 deviates.
        tau0 = torch.tensor([[3.0, 0.0], [0.0, 0.0]])
        tau1 = torch.tensor([[0.0, 0.0], [0.0, 3.0]])
        w = _dynamic_kl_weights([tau0, tau1], [0.5, 0.5], alpha=1.0, eps=1e-8)
        self.assertGreater(w[0][0].item(), 0.99)
        self.assertLess(w[0][1].item(), 0.01)
        self.assertLess(w[1][0].item(), 0.01)
        self.assertGreater(w[1][1].item(), 0.99)

    def test_static_weights_compose(self) -> None:
        taus = [torch.tensor([[2.0, 0.0]]), torch.tensor([[0.0, 2.0]])]  # equal D = 4
        w = _dynamic_kl_weights(taus, [0.25, 0.75], alpha=1.0, eps=1e-8)
        torch.testing.assert_close(w[0], torch.tensor([0.25]))
        torch.testing.assert_close(w[1], torch.tensor([0.75]))

    def test_rejects_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            _dynamic_kl_weights([torch.tensor([[1.0]])], [0.5, 0.5], alpha=1.0, eps=1e-8)


class TestEnsembleForwardStepKlWeighting(unittest.TestCase):
    """KL / inverse-KL weighting via ensemble_forward_step (pcgrad_residual)."""

    def _run(self, weighting: str):
        ref_pred = torch.zeros(1, 2)
        preds = {
            "eval_ckpt_0": torch.tensor([[3.0, 0.0]]),  # tau0 = [3, 0] -> D = 9
            "eval_ckpt_1": torch.tensor([[0.0, 1.0]]),  # tau1 = [0, 1] -> D = 1
        }
        adapter = _MockAdapterWithRef(preds, ref_pred)
        return ensemble_forward_step(
            adapter,
            ["eval_ckpt_0", "eval_ckpt_1"],
            [0.5, 0.5],
            {
                "t": torch.tensor(500),
                "t_next": torch.tensor(400),
                "latents": torch.zeros(1, 2),
                "compute_log_prob": False,
                "return_kwargs": ["next_latents", "noise_pred"],
            },
            adapter._sched_cache,
            base_forward=adapter.forward,
            blend_mode="pcgrad_residual",
            weighting=weighting,
        )

    def test_kl(self) -> None:
        # No conflict (dot=0); kl weights 0.9/0.1 -> 0.9*[3,0]+0.1*[0,1].
        out = self._run("kl")
        torch.testing.assert_close(out.noise_pred, torch.tensor([[2.7, 0.1]]))

    def test_kl_inv(self) -> None:
        # kl_inv weights 0.1/0.9 -> 0.1*[3,0]+0.9*[0,1].
        out = self._run("kl_inv")
        torch.testing.assert_close(out.noise_pred, torch.tensor([[0.3, 0.9]]))

    def test_ties_kl_weighted_sign_vote(self) -> None:
        ref_pred = torch.zeros(1, 2)
        preds = {
            "eval_ckpt_0": torch.tensor([[2.0, 0.0]]),  # D = 4
            "eval_ckpt_1": torch.tensor([[0.0, 1.0]]),  # D = 1
        }
        adapter = _MockAdapterWithRef(preds, ref_pred)
        out = ensemble_forward_step(
            adapter,
            ["eval_ckpt_0", "eval_ckpt_1"],
            [0.5, 0.5],
            {
                "t": torch.tensor(500),
                "t_next": torch.tensor(400),
                "latents": torch.zeros(1, 2),
                "compute_log_prob": False,
                "return_kwargs": ["next_latents", "noise_pred"],
            },
            adapter._sched_cache,
            base_forward=adapter.forward,
            blend_mode="ties",
            weighting="kl",
        )
        # kl weights 0.8/0.2; disjoint mean per element -> [2.0, 1.0].
        torch.testing.assert_close(out.noise_pred, torch.tensor([[2.0, 1.0]]))

    def test_rejects_kl_on_full_velocity_mode(self) -> None:
        preds = {
            "eval_ckpt_0": torch.tensor([1.0]),
            "eval_ckpt_1": torch.tensor([2.0]),
        }
        adapter = _MockAdapter(preds)
        with self.assertRaises(ValueError):
            ensemble_forward_step(
                adapter,
                ["eval_ckpt_0", "eval_ckpt_1"],
                [0.5, 0.5],
                {
                    "t": torch.tensor(500),
                    "t_next": torch.tensor(400),
                    "latents": torch.tensor([0.0]),
                    "compute_log_prob": False,
                    "return_kwargs": ["next_latents", "noise_pred"],
                },
                adapter._sched_cache,
                base_forward=adapter.forward,
                blend_mode="pcgrad",
                weighting="kl",
            )


class TestBlendWeightingValidation(unittest.TestCase):
    def test_accepts_uniform_with_any_mode(self) -> None:
        args = EnsembleEvalTrainingArguments(
            checkpoint_paths=["/tmp/a"],
            ensemble_blend_mode="weighted",
            ensemble_blend_weighting="uniform",
            unique_sample_num_per_epoch=1,
            group_size=1,
            per_device_batch_size=1,
        )
        self.assertEqual(args.ensemble_blend_weighting, "uniform")

    def test_accepts_kl_with_residual_and_ties(self) -> None:
        for mode in (
            "pcgrad_residual",
            "pcgrad_residual_channelwise",
            "pcgrad_residual_normalized",
            "ties",
        ):
            for weighting in ("kl", "kl_inv"):
                args = EnsembleEvalTrainingArguments(
                    checkpoint_paths=["/tmp/a"],
                    ensemble_blend_mode=mode,
                    ensemble_blend_weighting=weighting,
                    unique_sample_num_per_epoch=1,
                    group_size=1,
                    per_device_batch_size=1,
                )
                self.assertEqual(args.ensemble_blend_weighting, weighting)

    def test_rejects_invalid_weighting(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleEvalTrainingArguments(
                checkpoint_paths=["/tmp/a"],
                ensemble_blend_weighting="bogus",  # type: ignore[arg-type]
                unique_sample_num_per_epoch=1,
                group_size=1,
                per_device_batch_size=1,
            )

    def test_rejects_kl_with_non_residual_mode(self) -> None:
        for mode in ("weighted", "pcgrad", "pcgrad_channelwise", "pcgrad_normalized"):
            with self.assertRaises(ValueError):
                EnsembleEvalTrainingArguments(
                    checkpoint_paths=["/tmp/a"],
                    ensemble_blend_mode=mode,
                    ensemble_blend_weighting="kl",
                    unique_sample_num_per_epoch=1,
                    group_size=1,
                    per_device_batch_size=1,
                )


class TestEvalInferenceContextDDPBypass(unittest.TestCase):
    """The ensemble eval context bypasses the wrapper ONLY under plain DDP, so the
    per-teacher use_named_parameters swap takes effect (otherwise teachers produce
    identical noise_pred and every blend mode collapses). It must NOT bypass under
    DeepSpeed (swaps reflected natively) or sharded backends (ZeRO-3 / FSDP hold
    only param shards -> bypassing would read empty/garbage weights)."""

    @staticmethod
    def _make_trainer(*, is_deepspeed: bool, is_fsdp: bool = False, is_sharded: bool = False):
        from types import SimpleNamespace

        cls = get_trainer_class("ensemble-eval")

        # `unwrapped` stands in for pipeline.transformer (get_component_unwrapped);
        # `wrapped` for the prepared module stored in _components.
        wrapped = torch.nn.Linear(1, 1)
        unwrapped = torch.nn.Linear(1, 1)

        class _FakeAdapter:
            def __init__(self) -> None:
                self._components = {"transformer": wrapped}
                self.forward = lambda **kw: None  # patched/restored by the context

            def get_component(self, name: str):
                return self._components[name]

            def get_component_unwrapped(self, name: str):
                return unwrapped

            def set_component(self, name: str, module) -> None:
                self._components[name] = module

            def _is_deepspeed(self) -> bool:
                return is_deepspeed

            def _is_fsdp(self) -> bool:
                return is_fsdp

            def _is_param_sharded(self) -> bool:
                return is_sharded

        trainer = object.__new__(cls)
        trainer._checkpoint_names = ["eval_ckpt_0", "eval_ckpt_1"]
        trainer._weights = [0.5, 0.5]
        trainer._sched_cache = (frozenset(), False)
        trainer._pcgrad_generator = None
        trainer.adapter = _FakeAdapter()
        trainer.training_args = SimpleNamespace(
            ensemble_blend_mode="pcgrad",
            ensemble_blend_weighting="uniform",
            pcgrad_eps=1e-8,
            ties_density=1.0,
        )
        return trainer, wrapped, unwrapped

    def test_ddp_bypasses_to_unwrapped_transformer(self) -> None:
        trainer, wrapped, unwrapped = self._make_trainer(is_deepspeed=False)
        self.assertIs(trainer.adapter.get_component("transformer"), wrapped)
        with trainer._eval_inference_context():
            # Plain DDP: the active transformer must be the unwrapped
            # pipeline.transformer so per-checkpoint swaps are observed.
            self.assertIs(trainer.adapter.get_component("transformer"), unwrapped)
        # Restored on exit.
        self.assertIs(trainer.adapter.get_component("transformer"), wrapped)

    def test_deepspeed_does_not_bypass(self) -> None:
        trainer, wrapped, _ = self._make_trainer(is_deepspeed=True)
        with trainer._eval_inference_context():
            # DeepSpeed reflects .data.copy_() swaps natively -> keep the canonical
            # wrapped path (matches ff-train's default deepspeed_zero2 behavior).
            self.assertIs(trainer.adapter.get_component("transformer"), wrapped)
        self.assertIs(trainer.adapter.get_component("transformer"), wrapped)

    def test_sharded_backend_does_not_bypass(self) -> None:
        trainer, wrapped, _ = self._make_trainer(is_deepspeed=False, is_sharded=True)
        with trainer._eval_inference_context():
            # ZeRO-3 / FSDP hold only param shards -> bypassing would read garbage.
            self.assertIs(trainer.adapter.get_component("transformer"), wrapped)
        self.assertIs(trainer.adapter.get_component("transformer"), wrapped)


class TestUseRefParametersInPlaceLoRA(unittest.TestCase):
    """Under the DDP bypass the active transformer is the unwrapped
    pipeline.transformer -- an in-place get_peft_model-injected module, *not* a
    PeftModel wrapper. use_ref_parameters must still disable its LoRA (so the
    residual/TIES v_base is the true base) and restore it on exit, instead of
    warning and leaving the adapter active."""

    def test_disables_and_restores_inplace_injected_lora(self) -> None:
        from types import SimpleNamespace

        import torch.nn as nn
        from peft import LoraConfig, get_peft_model

        from flow_factory.models.abc import BaseAdapter

        class _Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                return self.proj(x)

        # Concrete stub so the ABC can be instantiated; __init__ is bypassed and
        # the abstract methods are never called by use_ref_parameters.
        class _StubAdapter(BaseAdapter):
            def decode_latents(self, *a, **k):  # pragma: no cover - stub
                raise NotImplementedError

            def forward(self, *a, **k):  # pragma: no cover - stub
                raise NotImplementedError

            def inference(self, *a, **k):  # pragma: no cover - stub
                raise NotImplementedError

            def load_pipeline(self, *a, **k):  # pragma: no cover - stub
                raise NotImplementedError

        base = _Tiny()
        # Injects LoRA into base.proj in-place; the wrapper is discarded so the
        # active component is the bare module (mirrors the DDP-bypass path).
        peft_model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]))
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                with torch.no_grad():
                    param.add_(torch.randn_like(param))

        adapter = object.__new__(_StubAdapter)
        adapter.model_args = SimpleNamespace(finetune_type="lora")
        adapter.target_module_map = {"transformer": None}
        adapter.get_component = lambda name: base
        adapter._unwrap = lambda module: module

        x = torch.randn(3, 4)
        out_lora = base(x).clone()
        with adapter.use_ref_parameters():
            out_ref = base(x).clone()
        out_after = base(x).clone()

        # LoRA disabled inside the context (output reverts to pure base) ...
        self.assertFalse(torch.allclose(out_lora, out_ref))
        self.assertTrue(torch.allclose(out_ref, base.proj.base_layer(x)))
        # ... and restored on exit.
        self.assertTrue(torch.allclose(out_lora, out_after))


class TestEnsembleEvalRegistry(unittest.TestCase):
    def test_trainer_registered(self) -> None:
        cls = get_trainer_class("ensemble-eval")
        self.assertEqual(cls.__name__, "EnsembleEvalTrainer")


if __name__ == "__main__":
    unittest.main()
