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

"""Focused trainer rollout tests for marginal-mixture conditional flow matching."""

import unittest
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from flow_factory.models.flux.flux2_klein import Flux2KleinSample
from flow_factory.samples import BaseSample
from flow_factory.trainers.xopd import common as xopd_common
from flow_factory.trainers.xopd import trainer as xopd_trainer_module
from flow_factory.trainers.xopd.common import draw_marginal_cfm_branches
from flow_factory.trainers.xopd.trainer import XOPDTrainer
from flow_factory.utils.base import stitch_batch_metadata
from flow_factory.utils.trajectory_collector import compute_trajectory_indices


class _AttrDict(dict):
    """Dictionary with attribute access for lightweight TrainingArguments tests."""

    def __getattr__(self, name):
        return self[name]


class _RecordingAdapter:
    """Record inference routing while returning structurally realistic samples."""

    def __init__(self):
        self.scheduler = SimpleNamespace(dynamics_type="ODE")
        self.active_source = "old"
        self.autocast_depth = 0
        self.calls = []
        self.rollout_calls = 0
        self.teacher_context_entries = 0
        self.drop_last_output = False
        self.invalid_output_index = None

    def rollout(self):
        self.rollout_calls += 1

    @contextmanager
    def use_teacher_transformer(self):
        self.teacher_context_entries += 1
        previous_source = self.active_source
        self.active_source = "teacher"
        try:
            yield
        finally:
            self.active_source = previous_source

    def inference(self, **kwargs):
        prompt_embeds = kwargs["prompt_embeds"]
        batch_size = int(prompt_embeds.shape[0])
        self.calls.append(
            {
                "source": self.active_source,
                "grad_enabled": torch.is_grad_enabled(),
                "autocast_active": self.autocast_depth > 0,
                "kwargs": kwargs,
            }
        )

        samples = []
        for row_index in range(batch_size):
            samples.append(
                Flux2KleinSample(
                    prompt=kwargs["prompt"][row_index],
                    prompt_ids=self._row(kwargs.get("prompt_ids"), row_index),
                    prompt_embeds=prompt_embeds[row_index],
                    text_ids=kwargs["text_ids"][row_index],
                    negative_prompt=kwargs["negative_prompt"][row_index],
                    negative_prompt_ids=self._row(
                        kwargs.get("negative_prompt_ids"),
                        row_index,
                    ),
                    negative_prompt_embeds=self._row(
                        kwargs.get("negative_prompt_embeds"),
                        row_index,
                    ),
                    negative_text_ids=self._row(
                        kwargs.get("negative_text_ids"),
                        row_index,
                    ),
                    teacher_prompt_embeds=kwargs["teacher_prompt_embeds"][row_index],
                    teacher_text_ids=kwargs["teacher_text_ids"][row_index],
                    teacher_negative_prompt_embeds=self._row(
                        kwargs.get("teacher_negative_prompt_embeds"),
                        row_index,
                    ),
                    teacher_negative_text_ids=self._row(
                        kwargs.get("teacher_negative_text_ids"),
                        row_index,
                    ),
                    condition_images=self._row(
                        kwargs.get("condition_images"),
                        row_index,
                    ),
                    image_latents=self._row(kwargs.get("image_latents"), row_index),
                    image_latent_ids=self._row(
                        kwargs.get("image_latent_ids"),
                        row_index,
                    ),
                    extra_kwargs={
                        "rollout_source": self.active_source,
                        "noise_pred": torch.full(
                            (2, 3),
                            1.0 if self.active_source == "teacher" else -1.0,
                        ),
                    },
                )
            )
        if self.drop_last_output:
            return samples[:-1]
        if self.invalid_output_index is not None:
            samples[self.invalid_output_index] = object()
        return samples

    @staticmethod
    def _row(value, row_index):
        if value is None:
            return None
        return value[row_index]


class _AutocastTracker:
    def __init__(self, adapter):
        self.adapter = adapter

    def __enter__(self):
        self.adapter.autocast_depth += 1

    def __exit__(self, exc_type, exc_value, traceback):
        self.adapter.autocast_depth -= 1


class _OptimizationAdapter:
    """Minimal differentiable adapter for marginal-CFM optimizer tests."""

    def __init__(self, events=None):
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.scheduler = SimpleNamespace(
            dynamics_type="ODE",
            noise_level=0.0,
            num_train_timesteps=1000,
        )
        self.trainable_components = (self,)
        self.events = events if events is not None else []
        self.train_calls = 0

    def train(self):
        self.train_calls += 1

    def forward(self, **kwargs):
        self.events.append("forward")
        latents = kwargs["latents"]
        noise_pred = self.weight.expand_as(latents)
        return SimpleNamespace(
            noise_pred=noise_pred,
            next_latents_mean=torch.zeros_like(latents),
            std_dev_t=torch.zeros(latents.shape[0], 1),
            dt=-torch.ones(latents.shape[0], 1),
        )

    def collect_moe_aux_loss(self):
        self.events.append("moe")
        return self.weight * 0.0 + 3.0

    def collect_router_z_loss(self):
        self.events.append("router_z")
        return self.weight * 0.0 + 4.0

    def collect_weight_sum_penalty(self):
        self.events.append("weight_sum")
        return self.weight * 0.0 + 5.0

    def get_trainable_parameters(self):
        return [self.weight]


class _OptimizationAccelerator:
    """CPU accelerator stub with a configurable final synchronization boundary."""

    def __init__(self, events=None, *, sync_on_accumulate_call=None):
        self.device = torch.device("cpu")
        self.events = events if events is not None else []
        self.sync_on_accumulate_call = sync_on_accumulate_call
        self.accumulate_calls = 0
        self.sync_gradients = False
        self.backward_losses = []
        self.is_local_main_process = False

    @contextmanager
    def accumulate(self, *components):
        self.accumulate_calls += 1
        self.sync_gradients = self.accumulate_calls == self.sync_on_accumulate_call
        yield

    def backward(self, loss):
        self.events.append("backward")
        self.backward_losses.append(loss.detach())
        loss.backward()


class _RecordingOptimizer:
    def __init__(self, events):
        self.events = events
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self):
        self.events.append("step")
        self.step_calls += 1

    def zero_grad(self):
        self.events.append("zero_grad")
        self.zero_grad_calls += 1


class TestMarginalCFMSample(unittest.TestCase):
    def _batch(self):
        batch_size = 4
        return {
            "prompt": [f"prompt-{row}" for row in range(batch_size)],
            "prompt_ids": torch.arange(batch_size * 2).reshape(batch_size, 2),
            "prompt_embeds": torch.arange(
                batch_size * 2 * 3,
                dtype=torch.float32,
            ).reshape(batch_size, 2, 3),
            "text_ids": torch.arange(batch_size * 2).reshape(batch_size, 2) + 100,
            "negative_prompt": [f"negative-{row}" for row in range(batch_size)],
            "negative_prompt_ids": torch.arange(batch_size * 2).reshape(batch_size, 2) + 200,
            "negative_prompt_embeds": torch.arange(
                batch_size * 2 * 3,
                dtype=torch.float32,
            ).reshape(batch_size, 2, 3)
            + 300,
            "negative_text_ids": torch.arange(batch_size * 2).reshape(batch_size, 2) + 400,
            "teacher_prompt_embeds": torch.arange(
                batch_size * 2 * 3,
                dtype=torch.float32,
            ).reshape(batch_size, 2, 3)
            + 500,
            "teacher_text_ids": torch.arange(batch_size * 2).reshape(batch_size, 2) + 600,
            "teacher_negative_prompt_embeds": torch.arange(
                batch_size * 2 * 3,
                dtype=torch.float32,
            ).reshape(batch_size, 2, 3)
            + 700,
            "teacher_negative_text_ids": torch.arange(batch_size * 2).reshape(batch_size, 2) + 800,
            "condition_images": [
                [
                    torch.full((3, 2, 2), float(row)),
                    torch.full((3, 3, 2), float(row + 10)),
                ]
                for row in range(batch_size)
            ],
            "image_latents": [torch.full((row + 1, 2), float(row)) for row in range(batch_size)],
            "image_latent_ids": [
                torch.full((row + 1, 3), row, dtype=torch.long) for row in range(batch_size)
            ],
            "metadata": [{"row_tag": f"tag-{row}"} for row in range(batch_size)],
            "__source__": "shared-dataset",
        }

    def _trainer(self, batch, *, alpha=0.5, seed=1, marginal=True, popd=False):
        adapter = _RecordingAdapter()
        trainer = XOPDTrainer.__new__(XOPDTrainer)
        trainer.adapter = adapter
        trainer.training_args = _AttrDict(
            seed=seed,
            marginal_cfm_alpha=alpha,
            guidance_scale=2.0,
            xopd_resample_steps_per_batch=False,
            num_inference_steps=4,
            num_batches_per_epoch=1,
            xopd_train_steps=None,
            num_xopd_steps=None,
        )
        trainer.student_gs = 2.0
        trainer.teacher_gs = 3.0
        trainer._is_marginal_cfm = marginal
        trainer._is_popd = popd
        trainer._is_ode = True
        trainer._cross_vae = False
        trainer.epoch = 2
        trainer.log_args = SimpleNamespace(verbose=False)
        trainer.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            is_local_main_process=False,
        )
        trainer.autocast = lambda: _AutocastTracker(adapter)
        trainer._make_train_iter = lambda: iter([batch])
        trainer.offload_calls = []
        trainer._maybe_offload_samples_to_cpu = lambda samples: trainer.offload_calls.append(
            samples
        )
        return trainer

    def test_mixed_branches_route_once_restore_order_and_student_conditioning(self):
        batch = self._batch()
        trainer = self._trainer(batch)
        expected_branches = draw_marginal_cfm_branches(
            4,
            alpha=0.5,
            seed=1,
            epoch=2,
            batch_index=0,
        )
        self.assertTrue(expected_branches.any())
        self.assertTrue((~expected_branches).any())

        stitch_calls = []

        def record_stitch(stitch_batch, samples):
            stitch_calls.append((stitch_batch, samples))
            stitch_batch_metadata(stitch_batch, samples)

        with patch.object(xopd_trainer_module, "stitch_batch_metadata", record_stitch):
            samples = trainer.sample()

        self.assertEqual(trainer.adapter.rollout_calls, 1)
        self.assertEqual([call["source"] for call in trainer.adapter.calls], ["old", "teacher"])
        self.assertEqual(
            sum(call["kwargs"]["prompt_embeds"].shape[0] for call in trainer.adapter.calls),
            len(batch["prompt"]),
        )
        self.assertEqual([sample.prompt for sample in samples], batch["prompt"])
        self.assertEqual(
            [sample.extra_kwargs["rollout_source"] for sample in samples],
            ["teacher" if branch else "old" for branch in expected_branches.tolist()],
        )

        expected_trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=trainer._train_timestep_indices,
            num_inference_steps=trainer.training_args.num_inference_steps,
        )
        for call in trainer.adapter.calls:
            self.assertFalse(call["grad_enabled"])
            self.assertTrue(call["autocast_active"])
            self.assertEqual(call["kwargs"]["extra_call_back_kwargs"], ["noise_pred"])
            self.assertEqual(call["kwargs"]["trajectory_indices"], expected_trajectory_indices)
            self.assertEqual(call["kwargs"]["__source__"], "shared-dataset")

        old_call, teacher_call = trainer.adapter.calls
        self.assertEqual(old_call["kwargs"]["guidance_scale"], trainer.student_gs)
        self.assertEqual(teacher_call["kwargs"]["guidance_scale"], trainer.teacher_gs)
        self.assertEqual(trainer.adapter.teacher_context_entries, 1)
        self.assertNotIn("prompt_ids", teacher_call["kwargs"])
        self.assertNotIn("negative_prompt_ids", teacher_call["kwargs"])

        teacher_rows = torch.nonzero(expected_branches, as_tuple=False).flatten()
        old_rows = torch.nonzero(~expected_branches, as_tuple=False).flatten()
        torch.testing.assert_close(
            teacher_call["kwargs"]["prompt_embeds"],
            batch["teacher_prompt_embeds"].index_select(0, teacher_rows),
        )
        torch.testing.assert_close(
            teacher_call["kwargs"]["text_ids"],
            batch["teacher_text_ids"].index_select(0, teacher_rows),
        )
        torch.testing.assert_close(
            teacher_call["kwargs"]["negative_prompt_embeds"],
            batch["teacher_negative_prompt_embeds"].index_select(0, teacher_rows),
        )
        torch.testing.assert_close(
            teacher_call["kwargs"]["negative_text_ids"],
            batch["teacher_negative_text_ids"].index_select(0, teacher_rows),
        )
        for subset_row, original_row in zip(
            old_call["kwargs"]["condition_images"],
            old_rows.tolist(),
        ):
            self.assertIs(subset_row, batch["condition_images"][original_row])
        for subset_row, original_row in zip(
            teacher_call["kwargs"]["condition_images"],
            teacher_rows.tolist(),
        ):
            self.assertIs(subset_row, batch["condition_images"][original_row])

        for row, sample in enumerate(samples):
            torch.testing.assert_close(sample.prompt_ids, batch["prompt_ids"][row])
            torch.testing.assert_close(sample.prompt_embeds, batch["prompt_embeds"][row])
            torch.testing.assert_close(sample.text_ids, batch["text_ids"][row])
            torch.testing.assert_close(
                sample.negative_prompt_ids,
                batch["negative_prompt_ids"][row],
            )
            torch.testing.assert_close(
                sample.negative_prompt_embeds,
                batch["negative_prompt_embeds"][row],
            )
            torch.testing.assert_close(
                sample.negative_text_ids,
                batch["negative_text_ids"][row],
            )
            torch.testing.assert_close(
                sample.teacher_prompt_embeds,
                batch["teacher_prompt_embeds"][row],
            )
            self.assertEqual(sample.extra_kwargs["row_tag"], f"tag-{row}")
            branch = sample.extra_kwargs["marginal_cfm_branch"]
            self.assertEqual(branch.dtype, torch.bool)
            self.assertEqual(branch.shape, torch.Size([]))
            self.assertEqual(branch.item(), expected_branches[row].item())

        stacked = BaseSample.stack(samples)
        self.assertEqual(stacked["marginal_cfm_branch"].shape, (4,))
        torch.testing.assert_close(stacked["marginal_cfm_branch"], expected_branches)
        self.assertEqual(len(stitch_calls), 1)
        self.assertIs(stitch_calls[0][0], batch)
        self.assertEqual(len(trainer.offload_calls), 1)
        self.assertIs(trainer.offload_calls[0], stitch_calls[0][1])
        for returned_sample, processed_sample in zip(samples, stitch_calls[0][1]):
            self.assertIs(returned_sample, processed_sample)

    def test_alpha_boundaries_skip_only_the_empty_subset(self):
        for alpha, expected_source in ((0.0, "old"), (1.0, "teacher")):
            with self.subTest(alpha=alpha):
                batch = self._batch()
                trainer = self._trainer(batch, alpha=alpha)

                samples = trainer.sample()

                self.assertEqual(
                    [call["source"] for call in trainer.adapter.calls],
                    [expected_source],
                )
                self.assertEqual(len(samples), len(batch["prompt"]))
                self.assertTrue(
                    all(
                        sample.extra_kwargs["marginal_cfm_branch"].item()
                        == (expected_source == "teacher")
                        for sample in samples
                    )
                )
                self.assertEqual(
                    trainer.adapter.teacher_context_entries,
                    int(expected_source == "teacher"),
                )

    def test_shared_generator_is_preserved_unchanged_for_subset_inference(self):
        batch = self._batch()
        shared_generator = torch.Generator().manual_seed(123)
        batch["generator"] = shared_generator
        trainer = self._trainer(batch, alpha=0.0)

        trainer.sample()

        self.assertIs(trainer.adapter.calls[0]["kwargs"]["generator"], shared_generator)

    def test_teacher_rollout_requires_positive_and_cfg_negative_conditioning(self):
        missing_cases = (
            ("teacher_prompt_embeds", "teacher_prompt_embeds"),
            ("teacher_text_ids", "teacher_text_ids"),
            ("teacher_negative_prompt_embeds", "teacher_negative_prompt_embeds"),
            ("teacher_negative_text_ids", "teacher_negative_text_ids"),
        )
        for removed_key, expected_key in missing_cases:
            with self.subTest(removed_key=removed_key):
                batch = self._batch()
                batch.pop(removed_key)
                trainer = self._trainer(batch, alpha=1.0)

                with self.assertRaises(ValueError) as context:
                    trainer.sample()

                self.assertIn(expected_key, str(context.exception))
                self.assertEqual(trainer.adapter.calls, [])

    def test_non_cfg_teacher_removes_active_negatives_then_restores_student_fields(self):
        batch = self._batch()
        trainer = self._trainer(batch, alpha=1.0)
        trainer.teacher_gs = 1.0

        samples = trainer.sample()

        teacher_kwargs = trainer.adapter.calls[0]["kwargs"]
        for key in (
            "negative_prompt_ids",
            "negative_prompt_embeds",
            "negative_text_ids",
        ):
            self.assertNotIn(key, teacher_kwargs)
        for row, sample in enumerate(samples):
            torch.testing.assert_close(
                sample.negative_prompt_ids,
                batch["negative_prompt_ids"][row],
            )
            torch.testing.assert_close(
                sample.negative_prompt_embeds,
                batch["negative_prompt_embeds"][row],
            )
            torch.testing.assert_close(
                sample.negative_text_ids,
                batch["negative_text_ids"][row],
            )

    def test_rejects_mismatched_or_unsupported_batch_values_before_inference(self):
        invalid_values = (
            ("prompt", ["prompt-0", "prompt-1", "prompt-2"]),
            ("prompt_ids", torch.zeros(3, 2)),
            ("unsupported", object()),
        )
        for key, invalid_value in invalid_values:
            with self.subTest(key=key):
                batch = self._batch()
                batch[key] = invalid_value
                trainer = self._trainer(batch)

                with self.assertRaises((TypeError, ValueError)) as context:
                    trainer.sample()

                self.assertIn(key, str(context.exception))
                self.assertEqual(trainer.adapter.calls, [])

    def test_rejects_subset_output_count_mismatch(self):
        batch = self._batch()
        trainer = self._trainer(batch, alpha=0.0)
        trainer.adapter.drop_last_output = True

        with self.assertRaisesRegex(RuntimeError, "expected=4.*received=3"):
            trainer.sample()

    def test_rejects_invalid_teacher_output_before_conditioning_restoration(self):
        batch = self._batch()
        trainer = self._trainer(batch, alpha=1.0)
        trainer.adapter.invalid_output_index = 0

        with self.assertRaisesRegex(
            TypeError,
            "expected BaseSample.*row=0.*use_teacher=True.*type=object",
        ):
            trainer.sample()

    def test_a4_callbacks_maps_and_branch_label_follow_sample_offload(self):
        sample = Flux2KleinSample(
            all_latents=torch.zeros(2, 3),
            latent_index_map=torch.tensor([0, 1]),
            extra_kwargs={
                "noise_pred": torch.ones(1, 3),
                "callback_index_map": torch.tensor([0]),
                "marginal_cfm_branch": torch.tensor(True),
            },
        )

        sample.to("meta")

        self.assertEqual(sample.all_latents.device.type, "meta")
        self.assertEqual(sample.latent_index_map.device.type, "meta")
        for key in ("noise_pred", "callback_index_map", "marginal_cfm_branch"):
            self.assertEqual(sample.extra_kwargs[key].device.type, "meta")


class TestExistingXOPDSampleModes(unittest.TestCase):
    def _run_mode(self, *, popd):
        helper = TestMarginalCFMSample()
        batch = helper._batch()
        trainer = helper._trainer(batch, marginal=False, popd=popd)
        samples = trainer.sample()
        return batch, trainer, samples

    def test_direct_rollout_remains_one_full_student_batch_without_callbacks(self):
        batch, trainer, samples = self._run_mode(popd=False)

        self.assertEqual(len(samples), len(batch["prompt"]))
        self.assertEqual([call["source"] for call in trainer.adapter.calls], ["old"])
        self.assertNotIn("extra_call_back_kwargs", trainer.adapter.calls[0]["kwargs"])
        self.assertTrue(all("marginal_cfm_branch" not in sample.extra_kwargs for sample in samples))

    def test_popd_rollout_preserves_behavior_transition_callbacks(self):
        _, trainer, samples = self._run_mode(popd=True)

        self.assertEqual(
            trainer.adapter.calls[0]["kwargs"]["extra_call_back_kwargs"],
            ["next_latents_mean", "std_dev_t", "dt"],
        )
        self.assertTrue(all("marginal_cfm_branch" not in sample.extra_kwargs for sample in samples))


class TestMarginalCFMOptimization(unittest.TestCase):
    def _trainer(
        self,
        *,
        marginal=True,
        popd=False,
        sync_on_accumulate_call=None,
        events=None,
    ):
        events = events if events is not None else []
        adapter = _OptimizationAdapter(events)
        accelerator = _OptimizationAccelerator(
            events,
            sync_on_accumulate_call=sync_on_accumulate_call,
        )
        trainer = XOPDTrainer.__new__(XOPDTrainer)
        trainer.adapter = adapter
        trainer.accelerator = accelerator
        trainer.optimizer = _RecordingOptimizer(events)
        trainer.training_args = _AttrDict(
            per_device_batch_size=2,
            seed=7,
            xopd_resample_steps_per_batch=False,
            num_inference_steps=1,
            xopd_train_steps=None,
            num_xopd_steps=None,
            kl_beta=0.0,
            kl_type="x-based",
            moe_load_balance_coeff=0.0,
            router_z_loss_coeff=0.0,
            mof_weight_sum_penalty_coeff=0.0,
            max_grad_norm=1.0,
        )
        trainer._is_marginal_cfm = marginal
        trainer._is_popd = popd
        trainer._is_ode = True
        trainer._pixel_loss = False
        trainer._is_hsct = False
        trainer.pathwise_coef = 1.0
        trainer.normalize_d_k = False
        trainer.xopd_dk_space = "v"
        trainer.student_gs = 1.0
        trainer.epoch = 0
        trainer.step = 0
        trainer.log_args = SimpleNamespace(verbose=False)
        trainer.autocast = nullcontext
        trainer._forward_param_names = frozenset()
        trainer._forward_accepts_var_kwargs = True
        trainer._clip_grad_norm_ep_aware = lambda parameters, max_norm: events.append(
            "clip"
        ) or torch.tensor(0.25)
        trainer.logged = []
        trainer.log_data = lambda data, step: trainer.logged.append((data, step))
        return trainer

    @staticmethod
    def _batch(
        *,
        num_timesteps=1,
        branches=None,
        targets=None,
        callback_index_map=None,
    ):
        batch_size = 2
        if branches is None:
            branches = torch.tensor([False, True])
        if targets is None:
            targets = torch.zeros(batch_size, num_timesteps, 2)
        if callback_index_map is None:
            callback_index_map = torch.arange(num_timesteps)
        return {
            "timesteps": torch.arange(
                num_timesteps,
                0,
                -1,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .expand(batch_size, -1),
            "all_latents": torch.zeros(batch_size, num_timesteps + 1, 2),
            "noise_pred": targets,
            "marginal_cfm_branch": branches,
            "callback_index_map": callback_index_map,
        }

    @staticmethod
    def _samples():
        callback_map = torch.tensor([0], dtype=torch.long)
        return [
            BaseSample(
                timesteps=torch.tensor([1.0]),
                all_latents=torch.zeros(2, 2),
                latent_index_map=torch.tensor([0, 1]),
                extra_kwargs={
                    "noise_pred": torch.zeros(1, 2),
                    "callback_index_map": callback_map.clone(),
                    "marginal_cfm_branch": torch.tensor(branch),
                },
            )
            for branch in (False, True)
        ]

    def test_student_return_keys_include_velocity_for_a4_and_preserve_kl_behavior(self):
        trainer = self._trainer()
        self.assertEqual(trainer._student_return_kwargs_for_train().count("noise_pred"), 1)

        trainer._is_marginal_cfm = False
        self.assertNotIn("noise_pred", trainer._student_return_kwargs_for_train())
        trainer.training_args["kl_beta"] = 1.0
        trainer.training_args["kl_type"] = "v-based"
        self.assertEqual(trainer._student_return_kwargs_for_train().count("noise_pred"), 1)
        trainer.training_args["kl_type"] = "x-based"
        self.assertNotIn("noise_pred", trainer._student_return_kwargs_for_train())

    def test_optimize_normalizes_callback_map_and_skips_teacher_prepasses(self):
        trainer = self._trainer()
        captured = {}
        trainer._precompute_teacher_means = lambda **kwargs: self.fail(
            "A4 must not precompute teacher means"
        )
        trainer._precompute_popd_step_caches = lambda **kwargs: self.fail(
            "A4 must not precompute P-OPD caches"
        )

        def capture_train_pass(**kwargs):
            captured.update(kwargs)
            return kwargs["loss_info"]

        trainer._optimize_train_pass = capture_train_pass
        trainer.optimize(self._samples())

        torch.testing.assert_close(captured["callback_index_map"], torch.tensor([0]))
        self.assertIsNone(captured["mu_teacher_list"])
        self.assertIsNone(captured["popd_cache_list"])
        self.assertEqual(trainer.adapter.train_calls, 1)

    def test_train_pass_strictly_separates_a4_and_existing_targets(self):
        batch = self._batch()
        latent_index_map = torch.tensor([0, 1])

        a4 = self._trainer()
        invalid_a4 = (
            {"callback_index_map": None},
            {
                "callback_index_map": torch.tensor([0]),
                "mu_teacher_list": [torch.zeros(2, 2)],
            },
            {
                "callback_index_map": torch.tensor([0]),
                "popd_cache_list": [],
            },
        )
        for overrides in invalid_a4:
            kwargs = {
                "batch": batch,
                "latents_index_map": latent_index_map,
                "num_timesteps": 1,
                "mu_teacher_list": None,
                "popd_cache_list": None,
                "callback_index_map": torch.tensor([0]),
                "loss_info": defaultdict(list),
            }
            kwargs.update(overrides)
            with self.subTest(mode="a4", overrides=overrides), self.assertRaises(ValueError):
                a4._optimize_train_pass(**kwargs)

        direct = self._trainer(marginal=False)
        with self.assertRaisesRegex(ValueError, "callback"):
            direct._optimize_train_pass(
                batch=batch,
                latents_index_map=latent_index_map,
                num_timesteps=1,
                mu_teacher_list=[torch.zeros(2, 2)],
                popd_cache_list=None,
                callback_index_map=torch.tensor([0]),
                loss_info=defaultdict(list),
            )
        with self.assertRaisesRegex(ValueError, "teacher"):
            direct._optimize_train_pass(
                batch=batch,
                latents_index_map=latent_index_map,
                num_timesteps=1,
                mu_teacher_list=None,
                popd_cache_list=None,
                callback_index_map=None,
                loss_info=defaultdict(list),
            )

    def test_callback_resolution_and_finite_target_errors_fail_fast(self):
        trainer = self._trainer()
        latent_index_map = torch.tensor([0, 1])
        invalid = (
            (torch.tensor([-1]), torch.zeros(2, 1, 2), "compact_index=-1"),
            (torch.tensor([1]), torch.zeros(2, 1, 2), "callback_count=1"),
            (
                torch.tensor([0]),
                torch.tensor([[[float("nan"), 0.0]], [[0.0, 0.0]]]),
                "finite target_noise_pred",
            ),
        )
        for callback_map, targets, expected in invalid:
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError) as context:
                    trainer._optimize_train_pass(
                        batch=self._batch(
                            targets=targets,
                            callback_index_map=callback_map,
                        ),
                        latents_index_map=latent_index_map,
                        num_timesteps=1,
                        mu_teacher_list=None,
                        popd_cache_list=None,
                        callback_index_map=callback_map,
                        loss_info=defaultdict(list),
                    )
                self.assertIn(expected, str(context.exception))

    def test_invalid_latent_index_is_rejected_before_student_forward(self):
        trainer = self._trainer()
        with self.assertRaisesRegex(ValueError, "compact_index=-1.*latent_count=2"):
            trainer._optimize_train_pass(
                batch=self._batch(),
                latents_index_map=torch.tensor([0, -1]),
                num_timesteps=1,
                mu_teacher_list=None,
                popd_cache_list=None,
                callback_index_map=torch.tensor([0]),
                loss_info=defaultdict(list),
            )
        self.assertEqual(trainer.adapter.events, [])

    def test_target_is_detached_student_gets_grad_and_branch_logs_omit_empty_branch(self):
        latent_index_map = torch.tensor([0, 1])
        for branches, present_key, absent_key in (
            (
                torch.tensor([False, False]),
                "marginal_cfm/loss_old",
                "marginal_cfm/loss_teacher",
            ),
            (
                torch.tensor([True, True]),
                "marginal_cfm/loss_teacher",
                "marginal_cfm/loss_old",
            ),
        ):
            with self.subTest(branches=branches):
                trainer = self._trainer()
                targets = torch.zeros(2, 1, 2, requires_grad=True)
                loss_info = trainer._optimize_train_pass(
                    batch=self._batch(branches=branches, targets=targets),
                    latents_index_map=latent_index_map,
                    num_timesteps=1,
                    mu_teacher_list=None,
                    popd_cache_list=None,
                    callback_index_map=torch.tensor([0]),
                    loss_info=defaultdict(list),
                )
                self.assertIsNotNone(trainer.adapter.weight.grad)
                self.assertIsNone(targets.grad)
                self.assertIn(present_key, loss_info)
                self.assertNotIn(absent_key, loss_info)
                for key in (
                    "marginal_cfm/teacher_branch_fraction",
                    "marginal_cfm/target_velocity_rms",
                    "marginal_cfm/target_velocity_l2",
                    "marginal_cfm/student_target_gap_rms",
                    "d_k",
                    "loss",
                    "d_k/0",
                    "loss/0",
                ):
                    self.assertIn(key, loss_info)

    def test_loss_order_and_sync_only_optimizer_step_preserve_shared_shell(self):
        events = []
        trainer = self._trainer(sync_on_accumulate_call=2, events=events)
        trainer.training_args["num_inference_steps"] = 2
        trainer.pathwise_coef = 2.0
        trainer.training_args.update(
            kl_beta=1.0,
            kl_type="v-based",
            moe_load_balance_coeff=0.5,
            router_z_loss_coeff=0.25,
            mof_weight_sum_penalty_coeff=0.1,
        )

        def compute_kl_anchor(student_out, forward_kwargs):
            events.append("kl")
            return torch.tensor(0.7), torch.tensor(0.7)

        trainer._compute_kl_anchor = compute_kl_anchor

        def reduce_loss(accelerator, loss_info):
            events.append("reduce")
            return {
                key: torch.cat(values).mean() if values[0].ndim > 0 else torch.stack(values).mean()
                for key, values in loss_info.items()
            }

        original_log_data = trainer.log_data

        def record_log(data, step):
            events.append("log")
            original_log_data(data, step)

        trainer.log_data = record_log
        batch = self._batch(
            num_timesteps=2,
            callback_index_map=torch.tensor([0, 1]),
        )
        with patch.object(xopd_trainer_module, "reduce_loss_info", reduce_loss):
            remaining = trainer._optimize_train_pass(
                batch=batch,
                latents_index_map=torch.tensor([0, 1, 2]),
                num_timesteps=2,
                mu_teacher_list=None,
                popd_cache_list=None,
                callback_index_map=torch.tensor([0, 1]),
                loss_info=defaultdict(list),
            )

        self.assertEqual(events.count("backward"), 2)
        self.assertEqual(trainer.optimizer.step_calls, 1)
        self.assertEqual(trainer.optimizer.zero_grad_calls, 1)
        self.assertEqual(trainer.step, 1)
        self.assertEqual(dict(remaining), {})
        self.assertEqual(
            events[-11:],
            [
                "forward",
                "moe",
                "router_z",
                "weight_sum",
                "kl",
                "backward",
                "clip",
                "step",
                "zero_grad",
                "reduce",
                "log",
            ],
        )
        logged, logged_step = trainer.logged[0]
        self.assertEqual(logged_step, 0)
        self.assertAlmostEqual(logged["train/loss"].item(), 5.7, places=5)
        for key in (
            "train/d_k",
            "train/loss",
            "train/d_k/0",
            "train/d_k/1",
            "train/marginal_cfm/teacher_branch_fraction",
            "train/marginal_cfm/callback_count",
        ):
            self.assertIn(key, logged)
        self.assertEqual(logged["train/marginal_cfm/callback_count"].item(), 2.0)
        self.assertFalse(any("popd/gamma" in key for key in logged))


class TestMarginalCFMLatentIndex(unittest.TestCase):
    def test_resolves_valid_index_and_rejects_invalid_map_bounds(self):
        self.assertTrue(
            hasattr(xopd_common, "resolve_marginal_cfm_latent_index"),
            "marginal CFM latent index resolver is missing",
        )
        resolve_marginal_cfm_latent_index = xopd_common.resolve_marginal_cfm_latent_index
        latent_index_map = torch.tensor([1, 0], dtype=torch.int64)
        self.assertEqual(
            resolve_marginal_cfm_latent_index(
                latent_index_map,
                timestep_index=0,
                latent_count=2,
            ),
            1,
        )

        invalid = (
            (
                torch.tensor([0, -1]),
                1,
                2,
                ("timestep_index=1", "compact_index=-1", "latent_count=2"),
            ),
            (
                torch.tensor([0, 1]),
                2,
                2,
                ("timestep_index=2", "map_length=2", "valid_timestep_bounds=[0, 2)"),
            ),
            (
                torch.tensor([0, 2]),
                1,
                2,
                ("timestep_index=1", "compact_index=2", "valid_compact_bounds=[0, 2)"),
            ),
        )
        for callback_map, timestep_index, latent_count, expected_parts in invalid:
            with self.subTest(
                callback_map=callback_map,
                timestep_index=timestep_index,
                latent_count=latent_count,
            ):
                with self.assertRaises(ValueError) as context:
                    resolve_marginal_cfm_latent_index(
                        callback_map,
                        timestep_index=timestep_index,
                        latent_count=latent_count,
                    )
                message = str(context.exception)
                for expected_part in expected_parts:
                    self.assertIn(expected_part, message)


class TestMarginalCFMEvaluation(unittest.TestCase):
    def test_a4_reward_eval_requests_final_image_only_and_does_not_cache_trajectory(self):
        for marginal in (True, False):
            with self.subTest(marginal=marginal):
                inference_calls = []
                rewarded_samples = []

                class _EvalAdapter:
                    def inference(self, **kwargs):
                        inference_calls.append(kwargs)
                        return [BaseSample(prompt=kwargs["prompt"][0])]

                trainer = XOPDTrainer.__new__(XOPDTrainer)
                trainer._is_marginal_cfm = marginal
                trainer.adapter = _EvalAdapter()
                trainer.training_args = SimpleNamespace(num_inference_steps=3)
                trainer.test_dataloaders = {"validation": [{"prompt": ["eval prompt"]}]}
                trainer.log_args = SimpleNamespace(verbose=False)
                trainer.accelerator = SimpleNamespace(is_local_main_process=False)
                trainer.eval_reward_buffer = SimpleNamespace(
                    add_samples=lambda samples: rewarded_samples.extend(samples)
                )
                trainer._eval_rollout_cache = {}
                merged_eval = _AttrDict(guidance_scale=1.0, num_inference_steps=3)

                samples = trainer._run_eval_inference_batches(
                    "validation",
                    merged_eval,
                    eval_seed=11,
                )

                self.assertEqual(rewarded_samples, samples)
                if marginal:
                    self.assertIsNone(inference_calls[0]["trajectory_indices"])
                    self.assertEqual(trainer._eval_rollout_cache, {})
                else:
                    self.assertEqual(
                        inference_calls[0]["trajectory_indices"],
                        [0, 1, 2, 3],
                    )
                    self.assertIs(trainer._eval_rollout_cache["validation"], samples)

    def test_a4_skips_validation_d_k_but_keeps_student_and_teacher_baseline_eval(self):
        for marginal, expected_validation_calls in ((True, 0), (False, 1)):
            with self.subTest(marginal=marginal):
                trainer = XOPDTrainer.__new__(XOPDTrainer)
                trainer._is_marginal_cfm = marginal
                trainer._teacher_baseline_scalars = {"eval/teacher/reward": 2.0}
                trainer.accelerator = SimpleNamespace(is_main_process=True)
                trainer.step = 3
                logged = []
                trainer.log_data = lambda data, step: logged.append((data, step))

                with (
                    patch.object(xopd_trainer_module.BaseTrainer, "evaluate") as student_eval,
                    patch.object(
                        XOPDTrainer,
                        "_evaluate_validation_d_k",
                    ) as validation_d_k,
                ):
                    trainer.evaluate()

                student_eval.assert_called_once_with()
                self.assertEqual(validation_d_k.call_count, expected_validation_calls)
                self.assertEqual(logged, [({"eval/teacher/reward": 2.0}, 3)])


if __name__ == "__main__":
    unittest.main()
