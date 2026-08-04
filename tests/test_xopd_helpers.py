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

"""Unit tests for Cross-OPD (XOPD) pure helpers and training-args validation."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from flow_factory.hparams.training_args import XOPDTrainingArguments
from flow_factory.models.flux.flux2_klein import Flux2KleinAdapter, Flux2KleinSample
from flow_factory.samples import BaseSample
from flow_factory.scheduler import SDESchedulerOutput
from flow_factory.trainers.xopd import common as xopd_common
from flow_factory.trainers.xopd.common import (
    align_l0_inner_steps,
    compute_forward_risk_velocity_loss,
    compute_per_step_kl,
    compute_xopd_detail_mask,
    compute_popd_diagnostics,
    compute_popd_gaussian_mean_kl,
    compute_popd_quantiles,
    compute_popd_responsibility,
    compute_transition_variance,
    compute_xopd_pdm_loss,
    extract_i2i_condition_kwargs,
    extract_popd_behavior_transition,
    interleaved_source_iter,
    l0_loss_weight,
    masked_per_sample_mean,
    validate_l1_one_step_per_epoch,
    validate_popd_configuration,
    validate_source_ratio,
    validate_xopd_target_configuration,
)


class _BranchRecordingTransformer:
    def __init__(self) -> None:
        self.calls = []

    def cache_context(self, name):
        return nullcontext()

    def __call__(self, *, hidden_states, encoder_hidden_states, **kwargs):
        self.calls.append(encoder_hidden_states)
        value = encoder_hidden_states[:, : hidden_states.shape[1], : hidden_states.shape[2]]
        return (value,)


class _BranchRecordingScheduler:
    noise_level = 0.0

    def step(self, *, noise_pred, return_kwargs, **kwargs):
        available = {
            "noise_pred": noise_pred,
            "next_latents_mean": noise_pred,
            "std_dev_t": torch.zeros(noise_pred.shape[0], 1),
            "dt": -torch.ones(noise_pred.shape[0], 1),
        }
        return SDESchedulerOutput.from_dict(
            {key: available[key] for key in return_kwargs if key in available}
        )


class TestFlux2KleinCFGBranchOutput(unittest.TestCase):
    def _adapter(self):
        adapter = Flux2KleinAdapter.__new__(Flux2KleinAdapter)
        adapter._components = {}
        adapter.transformer = _BranchRecordingTransformer()
        adapter.pipeline = SimpleNamespace(scheduler=_BranchRecordingScheduler())
        adapter._unwrap = lambda module: module
        return adapter

    @staticmethod
    def _kwargs():
        return {
            "t": torch.tensor([500.0]),
            "t_next": torch.tensor([0.0]),
            "latents": torch.zeros(1, 2, 3),
            "latent_ids": torch.zeros(1, 2, 4),
            "prompt_embeds": torch.ones(1, 2, 3),
            "text_ids": torch.zeros(1, 2, dtype=torch.long),
            "negative_prompt_embeds": torch.zeros(1, 2, 3),
            "negative_text_ids": torch.zeros(1, 2, dtype=torch.long),
            "guidance_scale": 1.0,
            "compute_log_prob": False,
        }

    def test_composed_gs_one_uses_only_positive_forward(self) -> None:
        adapter = self._adapter()
        output = adapter._forward(**self._kwargs(), return_kwargs=["noise_pred"])
        self.assertEqual(len(adapter.transformer.calls), 1)
        torch.testing.assert_close(output.noise_pred, torch.ones(1, 2, 3))

    def test_requested_branches_force_negative_forward_at_gs_one(self) -> None:
        adapter = self._adapter()
        output = adapter._forward(
            **self._kwargs(),
            return_kwargs=["positive_noise_pred", "negative_noise_pred"],
        )
        self.assertEqual(len(adapter.transformer.calls), 2)
        torch.testing.assert_close(output.positive_noise_pred, torch.ones(1, 2, 3))
        torch.testing.assert_close(output.negative_noise_pred, torch.zeros(1, 2, 3))


class TestXOPDPDMLoss(unittest.TestCase):
    def test_pdm_rejects_composed_error_cancellation(self) -> None:
        teacher_positive = torch.zeros(2, 3)
        teacher_negative = torch.zeros(2, 3)
        student_positive = torch.full((2, 3), -1.0)
        student_negative = torch.full((2, 3), -2.0)
        result = compute_xopd_pdm_loss(
            student_positive=student_positive,
            student_negative=student_negative,
            teacher_positive=teacher_positive,
            teacher_negative=teacher_negative,
            pdm_lambda=1.0,
            teacher_guidance_scale=2.0,
            student_guidance_scale=2.0,
        )
        torch.testing.assert_close(
            result.composed_mse_at_teacher_gs,
            torch.zeros(2),
        )
        torch.testing.assert_close(result.loss, torch.full((2,), 2.0))

    def test_pdm_zero_loss_requires_matching_branches_and_backpropagates(self) -> None:
        teacher_positive = torch.randn(2, 3)
        teacher_negative = torch.randn(2, 3)
        student_positive = teacher_positive.clone().requires_grad_(True)
        student_negative = teacher_negative.clone().requires_grad_(True)
        result = compute_xopd_pdm_loss(
            student_positive=student_positive,
            student_negative=student_negative,
            teacher_positive=teacher_positive,
            teacher_negative=teacher_negative,
            pdm_lambda=0.5,
            teacher_guidance_scale=4.0,
            student_guidance_scale=1.0,
        )
        torch.testing.assert_close(result.loss, torch.zeros(2))
        result.loss.mean().backward()
        torch.testing.assert_close(student_positive.grad, torch.zeros_like(student_positive))
        torch.testing.assert_close(student_negative.grad, torch.zeros_like(student_negative))


class TestForwardRiskVelocityLoss(unittest.TestCase):
    def test_equal_predictive_risk_returns_prior_gate(self) -> None:
        student = torch.zeros(2, 4, requires_grad=True)
        result = compute_forward_risk_velocity_loss(
            student_prediction=student,
            teacher_prediction=torch.zeros(2, 4),
            flow_target=torch.ones(2, 4),
            alpha=0.25,
            temperature=0.5,
            max_delta_rms=None,
        )
        torch.testing.assert_close(
            result.teacher_responsibility,
            torch.full((2,), 0.25),
        )
        torch.testing.assert_close(result.teacher_delta_rms, torch.zeros(2))

    def test_gate_prefers_predictor_with_lower_forward_risk(self) -> None:
        result = compute_forward_risk_velocity_loss(
            student_prediction=torch.zeros(2, 2, requires_grad=True),
            teacher_prediction=torch.ones(2, 2),
            flow_target=torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
            alpha=0.5,
            temperature=0.1,
            max_delta_rms=None,
        )
        self.assertGreater(result.teacher_responsibility[0].item(), 0.5)
        self.assertLess(result.teacher_responsibility[1].item(), 0.5)

    def test_event_mean_gate_is_dimension_stable(self) -> None:
        kwargs = {
            "alpha": 0.5,
            "temperature": 0.25,
            "max_delta_rms": None,
        }
        small = compute_forward_risk_velocity_loss(
            student_prediction=torch.zeros(1, 2, requires_grad=True),
            teacher_prediction=torch.ones(1, 2),
            flow_target=torch.ones(1, 2),
            **kwargs,
        )
        large = compute_forward_risk_velocity_loss(
            student_prediction=torch.zeros(1, 20, requires_grad=True),
            teacher_prediction=torch.ones(1, 20),
            flow_target=torch.ones(1, 20),
            **kwargs,
        )
        torch.testing.assert_close(
            small.teacher_responsibility,
            large.teacher_responsibility,
        )

    def test_radius_bounds_target_and_gradient_only_updates_student(self) -> None:
        student = torch.zeros(1, 4, requires_grad=True)
        teacher = torch.full((1, 4), 10.0, requires_grad=True)
        target = torch.full((1, 4), 10.0, requires_grad=True)
        result = compute_forward_risk_velocity_loss(
            student_prediction=student,
            teacher_prediction=teacher,
            flow_target=target,
            alpha=0.5,
            temperature=0.1,
            max_delta_rms=0.2,
        )
        self.assertLessEqual(result.target_delta_rms.item(), 0.2 + 1.0e-6)
        self.assertEqual(result.projection_active.item(), 1.0)
        result.loss.mean().backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(target.grad)
        self.assertFalse(result.teacher_responsibility.requires_grad)

    def test_rejects_invalid_shapes_and_hyperparameters(self) -> None:
        base = {
            "student_prediction": torch.zeros(1, 2),
            "teacher_prediction": torch.ones(1, 2),
            "flow_target": torch.ones(1, 2),
            "alpha": 0.5,
            "temperature": 1.0,
            "max_delta_rms": None,
        }
        for overrides in (
            {"teacher_prediction": torch.ones(1, 3)},
            {"alpha": 0.0},
            {"temperature": 0.0},
            {"max_delta_rms": 0.0},
        ):
            kwargs = dict(base)
            kwargs.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(
                (TypeError, ValueError)
            ):
                compute_forward_risk_velocity_loss(**kwargs)


class TestXOPDPerStepKL(unittest.TestCase):
    def test_unnormalized_is_mean_sq(self) -> None:
        mu_s = torch.zeros(2, 3)
        mu_t = torch.full((2, 3), 2.0)
        std_dev_t = torch.tensor([1.0, 1.0])
        dt = torch.tensor([-0.25, -0.25])
        d_k = compute_per_step_kl(mu_s, mu_t, std_dev_t, dt, normalize=False)
        torch.testing.assert_close(d_k, torch.tensor([4.0, 4.0]))

    def test_normalized_divides_by_sigma_bar(self) -> None:
        mu_s = torch.zeros(2, 3)
        mu_t = torch.full((2, 3), 2.0)
        std_dev_t = torch.tensor([1.0, 1.0])
        dt = torch.tensor([-0.25, -0.25])
        # sigma_bar^2 = std^2 * (-dt) = 0.25 -> divide by 2 * 0.25 = 0.5 -> 4 / 0.5 = 8.
        d_k = compute_per_step_kl(mu_s, mu_t, std_dev_t, dt, normalize=True)
        torch.testing.assert_close(d_k, torch.tensor([8.0, 8.0]))

    def test_ode_zero_std_falls_back_to_mse(self) -> None:
        mu_s = torch.zeros(2, 3)
        mu_t = torch.full((2, 3), 2.0)
        std_dev_t = torch.zeros(2)
        dt = torch.tensor([-0.25, -0.25])
        d_k = compute_per_step_kl(mu_s, mu_t, std_dev_t, dt, normalize=True)
        torch.testing.assert_close(d_k, torch.tensor([4.0, 4.0]))


class TestXOPDDetailMask(unittest.TestCase):
    def test_masks_large_anti_aligned_teacher_correction(self) -> None:
        base = torch.tensor(
            [
                [[0.0], [1.0], [0.0], [1.0]],
                [[0.0], [1.0], [0.0], [1.0]],
            ]
        )
        delta_x0 = torch.stack((-base[0], base[1]))
        mu_student = base.clone()
        mu_teacher = base + 0.5 * delta_x0
        result = compute_xopd_detail_mask(
            mu_student=mu_student,
            mu_teacher=mu_teacher,
            latents=base,
            sigma=torch.tensor([0.5, 0.5]),
            dt=torch.tensor([-0.25, -0.25]),
            threshold=1.0,
        )
        torch.testing.assert_close(
            result.gradient_alignment,
            torch.tensor([-1.0, 1.0]),
        )
        torch.testing.assert_close(
            result.velocity_gap_rms,
            torch.tensor([2.0**0.5, 2.0**0.5]),
        )
        torch.testing.assert_close(
            result.harmful_score,
            torch.tensor([2.0**0.5, 0.0]),
        )
        self.assertEqual(result.keep_mask.tolist(), [False, True])

    def test_masked_mean_excludes_masked_samples(self) -> None:
        values = torch.tensor([2.0, 8.0], requires_grad=True)
        loss = masked_per_sample_mean(values, torch.tensor([True, False]))
        torch.testing.assert_close(loss, torch.tensor(2.0))
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([1.0, 0.0]))

    def test_all_masked_mean_is_differentiable_zero(self) -> None:
        values = torch.tensor([2.0, 8.0], requires_grad=True)
        loss = masked_per_sample_mean(values, torch.tensor([False, False]))
        torch.testing.assert_close(loss, torch.tensor(0.0))
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([0.0, 0.0]))


class TestXOPDL0LossWeight(unittest.TestCase):
    def test_uniform(self) -> None:
        sigma = torch.tensor([0.1, 0.5, 0.9])
        w = l0_loss_weight(sigma, scheme="uniform")
        torch.testing.assert_close(w, torch.ones(3))

    def test_snr_ratio(self) -> None:
        sigma = torch.tensor([0.25, 0.5])
        w = l0_loss_weight(sigma, scheme="snr")
        torch.testing.assert_close(w, (1.0 - sigma) / sigma)

    def test_min_snr_clamps(self) -> None:
        sigma = torch.tensor([0.01, 0.5])  # ratio = 99, 1
        w = l0_loss_weight(sigma, scheme="min_snr", gamma=5.0)
        self.assertAlmostEqual(w[0].item(), 5.0, places=4)
        self.assertAlmostEqual(w[1].item(), 1.0, places=4)

    def test_no_blowup_at_zero(self) -> None:
        sigma = torch.tensor([0.0])
        w = l0_loss_weight(sigma, scheme="min_snr", gamma=5.0)
        self.assertTrue(torch.isfinite(w).all())
        self.assertLessEqual(w.item(), 5.0 + 1e-6)


class TestXOPDAlignL0InnerSteps(unittest.TestCase):
    def test_already_aligned_unchanged(self) -> None:
        # GAS = num_batches * T = 6 * 1 -> d = 1, any value already aligned.
        self.assertEqual(align_l0_inner_steps(6, 6, 4), 4)

    def test_round_up_to_multiple_of_t(self) -> None:
        # num_batches=6, GAS=12 -> T=2 (d = 12 // gcd(6,12)=12//6=2); 3 -> 4.
        self.assertEqual(align_l0_inner_steps(6, 12, 3), 4)
        self.assertEqual(align_l0_inner_steps(6, 12, 4), 4)

    def test_gcd_divisor(self) -> None:
        # num_batches=4, GAS=8 -> d = 8 // gcd(4,8)=8//4=2; 3 -> 4.
        self.assertEqual(align_l0_inner_steps(4, 8, 3), 4)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            align_l0_inner_steps(0, 6, 4)
        with self.assertRaises(ValueError):
            align_l0_inner_steps(6, 6, 0)


class TestXOPDValidateL1OneStep(unittest.TestCase):
    def test_valid_passes(self) -> None:
        validate_l1_one_step_per_epoch(
            num_batches_per_epoch=6,
            num_train_timesteps=1,
            gradient_accumulation_steps=6,
            num_inner_epochs=1,
        )

    def test_gas_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_l1_one_step_per_epoch(
                num_batches_per_epoch=6,
                num_train_timesteps=1,
                gradient_accumulation_steps=12,
                num_inner_epochs=1,
            )

    def test_inner_epochs_not_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_l1_one_step_per_epoch(
                num_batches_per_epoch=6,
                num_train_timesteps=1,
                gradient_accumulation_steps=6,
                num_inner_epochs=2,
            )


class TestXOPDTrainingArguments(unittest.TestCase):
    def test_requires_teacher_path(self) -> None:
        with self.assertRaises(ValueError):
            XOPDTrainingArguments(teacher_model_name_or_path="")

    def test_rejects_unshared_vae_te(self) -> None:
        with self.assertRaises(ValueError):
            XOPDTrainingArguments(
                teacher_model_name_or_path="/tmp/teacher",
                assume_shared_vae_text_encoder=False,
            )

    def test_student_guidance_synced_and_preprocess_max(self) -> None:
        args = XOPDTrainingArguments(
            teacher_model_name_or_path="/tmp/teacher",
            teacher_guidance_scale=4.0,
            student_guidance_scale=1.0,
        )
        # Base guidance_scale is driven by the student knob.
        self.assertEqual(args.guidance_scale, 1.0)
        # Preprocess must encode negatives if EITHER side uses CFG.
        self.assertEqual(args.get_preprocess_guidance_scale(), 4.0)

    def test_pdm_forces_negative_preprocessing_at_rollout_gs_one(self) -> None:
        args = XOPDTrainingArguments(
            teacher_model_name_or_path="/tmp/teacher",
            trainer_type="xopd",
            xopd_cfg_objective="pdm",
            xopd_pdm_lambda=1.0,
            xopd_dk_space="v",
            normalize_d_k=False,
            teacher_guidance_scale=1.0,
            student_guidance_scale=1.0,
        )
        self.assertEqual(args.guidance_scale, 1.0)
        self.assertEqual(args.get_preprocess_guidance_scale(), 2.0)

    def test_pdm_rejects_popd_and_non_velocity_loss(self) -> None:
        with self.assertRaises(ValueError):
            XOPDTrainingArguments(
                teacher_model_name_or_path="/tmp/teacher",
                trainer_type="xopd",
                xopd_cfg_objective="pdm",
                xopd_target_mode="p_opd",
                xopd_dk_space="v",
                normalize_d_k=False,
            )
        with self.assertRaises(ValueError):
            XOPDTrainingArguments(
                teacher_model_name_or_path="/tmp/teacher",
                trainer_type="xopd",
                xopd_cfg_objective="pdm",
                xopd_target_mode="direct",
                xopd_dk_space="xt",
                normalize_d_k=False,
            )

    def test_popd_defaults_preserve_direct_xopd(self) -> None:
        args = XOPDTrainingArguments(teacher_model_name_or_path="/tmp/teacher")
        self.assertEqual(args.xopd_target_mode, "direct")
        self.assertEqual(args.marginal_cfm_alpha, 0.5)
        self.assertEqual(args.popd_alpha, 0.5)
        self.assertEqual(args.popd_temperature, 1.0)
        self.assertEqual(args.forward_risk_alpha, 0.5)
        self.assertEqual(args.forward_risk_temperature, 1.0)
        self.assertIsNone(args.forward_risk_max_delta_rms)

    def test_forward_risk_accepts_valid_same_vae_ode_configuration(self) -> None:
        args = XOPDTrainingArguments(
            trainer_type="xopd",
            teacher_model_name_or_path="/tmp/teacher",
            xopd_target_mode="forward_risk",
            xopd_dk_space="v",
            normalize_d_k=False,
            forward_risk_alpha=0.4,
            forward_risk_temperature=0.02,
            forward_risk_max_delta_rms=0.1,
        )
        self.assertEqual(args.forward_risk_alpha, 0.4)
        self.assertEqual(args.forward_risk_temperature, 0.02)
        self.assertEqual(args.forward_risk_max_delta_rms, 0.1)

    def test_forward_risk_rejects_invalid_gate_and_incompatible_objectives(self) -> None:
        base = {
            "trainer_type": "xopd",
            "teacher_model_name_or_path": "/tmp/teacher",
            "xopd_target_mode": "forward_risk",
            "xopd_dk_space": "v",
            "normalize_d_k": False,
        }
        for overrides in (
            {"forward_risk_alpha": 0.0},
            {"forward_risk_temperature": 0.0},
            {"forward_risk_max_delta_rms": 0.0},
            {"xopd_cfg_objective": "pdm"},
            {"kl_beta": 0.1},
        ):
            kwargs = dict(base)
            kwargs.update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(
                (TypeError, ValueError)
            ):
                XOPDTrainingArguments(**kwargs)

    def test_detail_mask_requires_x0_norm(self) -> None:
        with self.assertRaises(ValueError):
            XOPDTrainingArguments(
                teacher_model_name_or_path="/tmp/teacher",
                xopd_detail_mask_enabled=True,
                xopd_dk_space="xt",
            )

    def test_detail_mask_validates_per_step_threshold_count(self) -> None:
        with self.assertRaises(ValueError):
            XOPDTrainingArguments(
                teacher_model_name_or_path="/tmp/teacher",
                xopd_detail_mask_enabled=True,
                xopd_dk_space="x0_norm",
                num_inference_steps=28,
                xopd_detail_mask_step_thresholds=[0.1, 0.2],
            )

    def test_detail_mask_accepts_x0_norm_thresholds(self) -> None:
        args = XOPDTrainingArguments(
            teacher_model_name_or_path="/tmp/teacher",
            xopd_detail_mask_enabled=True,
            xopd_dk_space="x0_norm",
            num_inference_steps=2,
            xopd_detail_mask_step_thresholds=[0.1, 0.2],
        )
        self.assertTrue(args.xopd_detail_mask_enabled)

    def test_direct_xopd_ignores_unused_popd_values(self) -> None:
        args = XOPDTrainingArguments(
            teacher_model_name_or_path="/tmp/teacher",
            xopd_target_mode="direct",
            popd_alpha=0.0,
            popd_temperature=0.0,
            marginal_cfm_alpha=float("nan"),
        )
        self.assertEqual(args.xopd_target_mode, "direct")

    def test_popd_rejects_invalid_alpha_and_temperature(self) -> None:
        for alpha in (0.0, 1.0, float("nan")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                XOPDTrainingArguments(
                    teacher_model_name_or_path="/tmp/teacher",
                    xopd_target_mode="p_opd",
                    popd_alpha=alpha,
                )
        for temperature in (0.0, -1.0, float("nan")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                XOPDTrainingArguments(
                    teacher_model_name_or_path="/tmp/teacher",
                    xopd_target_mode="p_opd",
                    popd_temperature=temperature,
                )

    def test_marginal_cfm_accepts_alpha_endpoints(self) -> None:
        for alpha in (0.0, 0.5, 1.0):
            with self.subTest(alpha=alpha):
                args = XOPDTrainingArguments(
                    trainer_type="xopd",
                    teacher_model_name_or_path="/tmp/teacher",
                    xopd_target_mode="marginal_cfm",
                    marginal_cfm_alpha=alpha,
                )
                self.assertEqual(args.marginal_cfm_alpha, alpha)

    def test_marginal_cfm_rejects_invalid_alpha(self) -> None:
        invalid = (True, "0.5", -0.1, 1.1, float("nan"), float("inf"), float("-inf"))
        for alpha in invalid:
            with self.subTest(alpha=alpha), self.assertRaises((TypeError, ValueError)):
                XOPDTrainingArguments(
                    trainer_type="xopd",
                    teacher_model_name_or_path="/tmp/teacher",
                    xopd_target_mode="marginal_cfm",
                    marginal_cfm_alpha=alpha,
                )

    def test_marginal_cfm_rejects_non_xopd_trainers(self) -> None:
        for trainer_type in ("xpdm", "xdmd"):
            with (
                self.subTest(trainer_type=trainer_type),
                self.assertRaisesRegex(
                    ValueError,
                    "trainer_type",
                ),
            ):
                XOPDTrainingArguments(
                    trainer_type=trainer_type,
                    teacher_model_name_or_path="/tmp/teacher",
                    xopd_target_mode="marginal_cfm",
                )

    def test_invalid_target_mode_lists_every_supported_mode(self) -> None:
        with self.assertRaises(ValueError) as context:
            XOPDTrainingArguments(
                trainer_type="xopd",
                teacher_model_name_or_path="/tmp/teacher",
                xopd_target_mode="unsupported",
            )
        message = str(context.exception)
        for mode in ("direct", "p_opd", "marginal_cfm", "forward_risk"):
            self.assertIn(mode, message)

    def test_popd_remains_independent_from_marginal_cfm_alpha(self) -> None:
        args = XOPDTrainingArguments(
            trainer_type="xopd",
            teacher_model_name_or_path="/tmp/teacher",
            xopd_target_mode="p_opd",
            popd_alpha=0.25,
            marginal_cfm_alpha=float("nan"),
        )
        self.assertEqual(args.popd_alpha, 0.25)


class TestPOPDConfiguration(unittest.TestCase):
    def test_accepts_supported_same_vae_sde_configuration(self) -> None:
        validate_popd_configuration(
            target_mode="p_opd",
            dynamics_type="Flow-SDE",
            noise_level=0.7,
            xopd_dk_space="xt",
            normalize_d_k=True,
            is_cross_vae=False,
            pixel_loss=False,
        )

    def test_direct_mode_is_unrestricted(self) -> None:
        validate_popd_configuration(
            target_mode="direct",
            dynamics_type="ODE",
            noise_level=0.0,
            xopd_dk_space="x0_norm",
            normalize_d_k=False,
            is_cross_vae=True,
            pixel_loss=True,
        )

    def test_rejects_unsupported_popd_combinations(self) -> None:
        invalid = (
            {"dynamics_type": "ODE"},
            {"noise_level": 0.0},
            {"xopd_dk_space": "v"},
            {"normalize_d_k": False},
            {"is_cross_vae": True},
            {"pixel_loss": True},
        )
        base = {
            "target_mode": "p_opd",
            "dynamics_type": "Flow-SDE",
            "noise_level": 0.7,
            "xopd_dk_space": "xt",
            "normalize_d_k": True,
            "is_cross_vae": False,
            "pixel_loss": False,
        }
        for override in invalid:
            kwargs = {**base, **override}
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_popd_configuration(**kwargs)

    def test_rejects_other_target_modes_with_legacy_error_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "'direct' or 'p_opd'"):
            validate_popd_configuration(
                target_mode="marginal_cfm",
                dynamics_type="ODE",
                noise_level=0.0,
                xopd_dk_space="v",
                normalize_d_k=False,
                is_cross_vae=False,
                pixel_loss=False,
            )


class TestXOPDTargetConfiguration(unittest.TestCase):
    def _validate(self, **overrides) -> None:
        configuration = {
            "target_mode": "marginal_cfm",
            "dynamics_type": "ODE",
            "noise_level": 0.0,
            "xopd_dk_space": "v",
            "normalize_d_k": False,
            "is_cross_vae": False,
            "pixel_loss": False,
            "vae_transport": "identity",
        }
        configuration.update(overrides)
        validate_xopd_target_configuration(**configuration)

    def test_accepts_marginal_cfm_same_vae_ode_configuration(self) -> None:
        self._validate()

    def test_accepts_forward_risk_same_vae_ode_configuration(self) -> None:
        self._validate(target_mode="forward_risk")

    def test_forward_risk_rejects_incompatible_geometry(self) -> None:
        invalid = (
            {"dynamics_type": "Flow-SDE"},
            {"noise_level": 0.1},
            {"xopd_dk_space": "xt"},
            {"normalize_d_k": True},
            {"is_cross_vae": True},
            {"pixel_loss": True},
            {"vae_transport": "flow"},
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises((TypeError, ValueError)):
                self._validate(target_mode="forward_risk", **override)

    def test_rejects_invalid_marginal_cfm_configuration_matrix(self) -> None:
        invalid = (
            ({"dynamics_type": "Flow-SDE"}, "dynamics_type='ODE'", "dynamics_type='Flow-SDE'"),
            ({"noise_level": 0.1}, "noise_level == 0", "noise_level=0.1"),
            ({"noise_level": float("nan")}, "finite", "noise_level=nan"),
            ({"xopd_dk_space": "xt"}, "xopd_dk_space='v'", "xopd_dk_space='xt'"),
            ({"normalize_d_k": True}, "normalize_d_k=False", "normalize_d_k=True"),
            ({"is_cross_vae": True}, "is_cross_vae=False", "is_cross_vae=True"),
            ({"pixel_loss": True}, "pixel_loss=False", "pixel_loss=True"),
        )
        for override, expected, received in invalid:
            with (
                self.subTest(override=override),
                self.assertRaises((TypeError, ValueError)) as context,
            ):
                self._validate(**override)
            message = str(context.exception)
            self.assertIn(expected, message)
            self.assertIn(received, message)

    def test_marginal_cfm_requires_identity_transport(self) -> None:
        for vae_transport in ("linear", "pixel", "hsct", "flow", "unknown"):
            with self.subTest(vae_transport=vae_transport):
                with self.assertRaisesRegex(
                    ValueError,
                    "expected vae_transport='identity'",
                ) as context:
                    self._validate(vae_transport=vae_transport)
                self.assertIn(f"got vae_transport={vae_transport!r}", str(context.exception))

    def test_dispatcher_preserves_direct_and_popd_validation(self) -> None:
        self._validate(
            target_mode="direct",
            dynamics_type="Flow-SDE",
            noise_level=float("nan"),
            xopd_dk_space="x0_norm",
            normalize_d_k=True,
            is_cross_vae=True,
            pixel_loss=True,
            vae_transport="flow",
        )
        self._validate(
            target_mode="p_opd",
            dynamics_type="Flow-SDE",
            noise_level=0.7,
            xopd_dk_space="xt",
            normalize_d_k=True,
            is_cross_vae=False,
            pixel_loss=False,
            vae_transport="flow",
        )


class TestMarginalCFMBranches(unittest.TestCase):
    def test_draw_is_deterministic_cpu_bool_and_tracks_alpha(self) -> None:
        first = xopd_common.draw_marginal_cfm_branches(
            100_000,
            alpha=0.37,
            seed=123,
            epoch=4,
            batch_index=5,
        )
        second = xopd_common.draw_marginal_cfm_branches(
            100_000,
            alpha=0.37,
            seed=123,
            epoch=4,
            batch_index=5,
        )
        different_batch = xopd_common.draw_marginal_cfm_branches(
            100_000,
            alpha=0.37,
            seed=123,
            epoch=4,
            batch_index=6,
        )

        self.assertEqual(first.shape, (100_000,))
        self.assertEqual(first.dtype, torch.bool)
        self.assertEqual(first.device.type, "cpu")
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different_batch))
        self.assertAlmostEqual(first.float().mean().item(), 0.37, delta=0.005)

    def test_draw_honors_exact_alpha_boundaries(self) -> None:
        old = xopd_common.draw_marginal_cfm_branches(
            8,
            alpha=0,
            seed=1,
            epoch=2,
            batch_index=3,
        )
        teacher = xopd_common.draw_marginal_cfm_branches(
            8,
            alpha=1,
            seed=1,
            epoch=2,
            batch_index=3,
        )
        self.assertFalse(old.any())
        self.assertTrue(teacher.all())

    def test_draw_rejects_invalid_batch_size_alpha_and_keys(self) -> None:
        for batch_size in (True, 0, -1, 1.5):
            with (
                self.subTest(batch_size=batch_size),
                self.assertRaises((TypeError, ValueError)) as context,
            ):
                xopd_common.draw_marginal_cfm_branches(
                    batch_size,
                    alpha=0.5,
                    seed=1,
                    epoch=2,
                    batch_index=3,
                )
            self.assertIn("batch_size", str(context.exception))

        for alpha in (True, "0.5", -0.1, 1.1, float("nan"), float("inf")):
            with (
                self.subTest(alpha=alpha),
                self.assertRaises((TypeError, ValueError)) as context,
            ):
                xopd_common.draw_marginal_cfm_branches(
                    4,
                    alpha=alpha,
                    seed=1,
                    epoch=2,
                    batch_index=3,
                )
            self.assertIn("alpha", str(context.exception))

        for name in ("seed", "epoch", "batch_index"):
            for invalid in (True, 1.5, "1"):
                kwargs = {"seed": 1, "epoch": 2, "batch_index": 3}
                kwargs[name] = invalid
                with (
                    self.subTest(name=name, invalid=invalid),
                    self.assertRaises(TypeError) as context,
                ):
                    xopd_common.draw_marginal_cfm_branches(4, alpha=0.5, **kwargs)
                self.assertIn(name, str(context.exception))


class TestMarginalCFMCallbackMap(unittest.TestCase):
    def test_normalizes_vector_and_identical_stacked_rows_with_sentinel(self) -> None:
        callback_map = torch.tensor([2, -1, 0, 1], dtype=torch.int64)
        normalized = xopd_common.normalize_callback_index_map(callback_map)
        stacked = xopd_common.normalize_callback_index_map(
            torch.stack((callback_map, callback_map.clone()))
        )
        torch.testing.assert_close(normalized, callback_map)
        torch.testing.assert_close(stacked, callback_map)

    def test_rejects_row_mismatch_dtype_shape_and_empty_maps(self) -> None:
        with self.assertRaisesRegex(ValueError, "same callback_index_map"):
            xopd_common.normalize_callback_index_map(
                torch.tensor([[0, -1], [-1, 0]], dtype=torch.int64)
            )

        invalid = (
            [0, 1],
            torch.tensor([0.0, 1.0]),
            torch.tensor([True, False]),
            torch.tensor(1),
            torch.zeros(1, 1, 1, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, 2, dtype=torch.int64),
            torch.empty(2, 0, dtype=torch.int64),
        )
        for callback_map in invalid:
            with (
                self.subTest(callback_map=callback_map),
                self.assertRaises((TypeError, ValueError)),
            ):
                xopd_common.normalize_callback_index_map(callback_map)

    def test_resolves_valid_index_and_rejects_sentinel_or_out_of_range(self) -> None:
        callback_map = torch.tensor([[2, -1, 0], [2, -1, 0]], dtype=torch.int32)
        self.assertEqual(
            xopd_common.resolve_marginal_cfm_callback_index(
                callback_map,
                timestep_index=0,
                callback_count=3,
            ),
            2,
        )

        for timestep_index, callback_count, expected in (
            (1, 3, "compact_index=-1"),
            (0, 2, "callback_count=2"),
        ):
            with (
                self.subTest(
                    timestep_index=timestep_index,
                    callback_count=callback_count,
                ),
                self.assertRaises(ValueError) as context,
            ):
                xopd_common.resolve_marginal_cfm_callback_index(
                    callback_map,
                    timestep_index=timestep_index,
                    callback_count=callback_count,
                )
            message = str(context.exception)
            self.assertIn(f"timestep_index={timestep_index}", message)
            self.assertIn(expected, message)

    def test_resolver_validates_timestep_and_callback_count(self) -> None:
        callback_map = torch.tensor([0, -1], dtype=torch.int64)
        for timestep_index in (True, -1, 2, 1.5):
            with (
                self.subTest(timestep_index=timestep_index),
                self.assertRaises((TypeError, ValueError)),
            ):
                xopd_common.resolve_marginal_cfm_callback_index(
                    callback_map,
                    timestep_index=timestep_index,
                    callback_count=1,
                )
        for callback_count in (True, -1, 1.5):
            with (
                self.subTest(callback_count=callback_count),
                self.assertRaises((TypeError, ValueError)),
            ):
                xopd_common.resolve_marginal_cfm_callback_index(
                    callback_map,
                    timestep_index=0,
                    callback_count=callback_count,
                )

    def test_real_flux2_klein_sample_stack_preserves_map_and_branch_shapes(self) -> None:
        callback_map = torch.tensor([-1, 0, -1, 1], dtype=torch.int64)
        samples = [
            Flux2KleinSample(
                extra_kwargs={
                    "callback_index_map": callback_map,
                    "marginal_cfm_branch": torch.tensor(False),
                }
            ),
            Flux2KleinSample(
                extra_kwargs={
                    "callback_index_map": callback_map.clone(),
                    "marginal_cfm_branch": torch.tensor(True),
                }
            ),
        ]
        batch = BaseSample.stack(samples)

        self.assertEqual(batch["callback_index_map"].shape, (2, 4))
        self.assertEqual(batch["marginal_cfm_branch"].shape, (2,))
        torch.testing.assert_close(
            xopd_common.normalize_callback_index_map(batch["callback_index_map"]),
            callback_map,
        )
        torch.testing.assert_close(
            batch["marginal_cfm_branch"],
            torch.tensor([False, True]),
        )


class TestMarginalCFMVelocityLoss(unittest.TestCase):
    def test_loss_is_fp32_event_mean_and_only_student_receives_gradients(self) -> None:
        student = torch.tensor(
            [[1.0, 3.0], [2.0, 6.0]],
            dtype=torch.float16,
            requires_grad=True,
        )
        target = torch.tensor(
            [[0.0, 1.0], [4.0, 2.0]],
            dtype=torch.float16,
            requires_grad=True,
        )
        loss = xopd_common.compute_marginal_cfm_velocity_loss(student, target)

        self.assertEqual(loss.shape, (2,))
        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, torch.tensor([2.5, 10.0]))
        loss.sum().backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(target.grad)

    def test_loss_rejects_type_shape_empty_event_and_nonfinite_inputs(self) -> None:
        invalid_pairs = (
            ([1.0], torch.ones(1, 1)),
            (torch.ones(1, 1), [1.0]),
            (torch.ones(2, 2), torch.ones(2, 3)),
            (torch.ones(2), torch.ones(2)),
            (torch.empty(0, 2), torch.empty(0, 2)),
            (torch.empty(2, 0), torch.empty(2, 0)),
            (torch.tensor([[float("inf")]]), torch.zeros(1, 1)),
            (torch.zeros(1, 1), torch.tensor([[float("nan")]])),
        )
        for student, target in invalid_pairs:
            with (
                self.subTest(student=student, target=target),
                self.assertRaises((TypeError, ValueError)),
            ):
                xopd_common.compute_marginal_cfm_velocity_loss(student, target)

    def test_diagnostics_match_hand_computed_scales_and_are_detached(self) -> None:
        student = torch.tensor(
            [[1.0, 3.0], [2.0, 6.0]],
            requires_grad=True,
        )
        target = torch.tensor(
            [[0.0, 1.0], [4.0, 2.0]],
            requires_grad=True,
        )
        teacher_branch_mask = torch.tensor([False, True])
        metrics = xopd_common.compute_marginal_cfm_diagnostics(
            student_noise_pred=student,
            target_noise_pred=target,
            teacher_branch_mask=teacher_branch_mask,
        )

        self.assertEqual(
            set(metrics),
            {
                "teacher_branch_fraction",
                "loss",
                "target_velocity_rms",
                "target_velocity_l2",
                "student_target_gap_rms",
            },
        )
        torch.testing.assert_close(
            metrics["teacher_branch_fraction"],
            torch.tensor([0.0, 1.0]),
        )
        torch.testing.assert_close(metrics["loss"], torch.tensor([2.5, 10.0]))
        torch.testing.assert_close(
            metrics["target_velocity_rms"],
            torch.tensor([0.5**0.5, 10.0**0.5]),
        )
        torch.testing.assert_close(
            metrics["target_velocity_l2"],
            torch.tensor([1.0, 20.0**0.5]),
        )
        torch.testing.assert_close(
            metrics["student_target_gap_rms"],
            torch.tensor([2.5**0.5, 10.0**0.5]),
        )
        self.assertTrue(all(value.shape == (2,) for value in metrics.values()))
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))
        torch.testing.assert_close(
            metrics["loss"][~teacher_branch_mask],
            torch.tensor([2.5]),
        )
        torch.testing.assert_close(
            metrics["loss"][teacher_branch_mask],
            torch.tensor([10.0]),
        )

        old_only = xopd_common.compute_marginal_cfm_diagnostics(
            student_noise_pred=student,
            target_noise_pred=target,
            teacher_branch_mask=torch.zeros(2, dtype=torch.bool),
        )
        self.assertEqual(old_only["loss"][torch.ones(2, dtype=torch.bool)].shape, (2,))
        self.assertEqual(old_only["loss"][torch.zeros(2, dtype=torch.bool)].shape, (0,))

    def test_diagnostics_validates_branch_mask(self) -> None:
        student = torch.zeros(2, 3)
        target = torch.ones(2, 3)
        invalid_masks = (
            [False, True],
            torch.tensor([0, 1]),
            torch.tensor([[False, True]]),
            torch.tensor([True]),
        )
        for teacher_branch_mask in invalid_masks:
            with (
                self.subTest(teacher_branch_mask=teacher_branch_mask),
                self.assertRaises((TypeError, ValueError)),
            ):
                xopd_common.compute_marginal_cfm_diagnostics(
                    student_noise_pred=student,
                    target_noise_pred=target,
                    teacher_branch_mask=teacher_branch_mask,
                )


class TestPOPDTransitionVariance(unittest.TestCase):
    def test_flow_and_dance_sde_include_negative_dt(self) -> None:
        std = torch.tensor([2.0, 3.0])
        dt = torch.tensor([-0.25, -0.5])
        expected = torch.tensor([1.0, 4.5])
        for dynamics_type in ("Flow-SDE", "Dance-SDE"):
            with self.subTest(dynamics_type=dynamics_type):
                actual = compute_transition_variance(std, dt, dynamics_type)
                torch.testing.assert_close(actual, expected)

    def test_cps_uses_step_standard_deviation_directly(self) -> None:
        actual = compute_transition_variance(
            torch.tensor([2.0, 3.0]),
            torch.tensor([-0.25, -0.5]),
            "CPS",
        )
        torch.testing.assert_close(actual, torch.tensor([4.0, 9.0]))

    def test_rejects_ode_and_invalid_variance(self) -> None:
        with self.assertRaises(ValueError):
            compute_transition_variance(torch.ones(1), torch.tensor([-0.1]), "ODE")
        for std in (torch.zeros(1), torch.tensor([float("nan")])):
            with self.subTest(std=std), self.assertRaises(ValueError):
                compute_transition_variance(std, torch.tensor([-0.1]), "Flow-SDE")


class TestPOPDResponsibility(unittest.TestCase):
    def test_identical_components_return_alpha(self) -> None:
        mu = torch.zeros(2, 3)
        result = compute_popd_responsibility(
            next_latents=torch.randn(2, 3),
            mu_old=mu,
            mu_teacher=mu,
            transition_variance=torch.ones(2),
            alpha=0.25,
            temperature=1.0,
        )
        torch.testing.assert_close(result.log_ratio_sum, torch.zeros(2))
        torch.testing.assert_close(result.teacher_responsibility, torch.full((2,), 0.25))
        torch.testing.assert_close(result.teacher_old_kl_joint, torch.zeros(2))
        self.assertEqual(result.event_dim, 3)

    def test_gate_tracks_which_component_better_explains_transition(self) -> None:
        mu_old = torch.zeros(2, 2)
        mu_teacher = torch.ones(2, 2)
        next_latents = torch.stack((torch.ones(2), torch.zeros(2)))
        result = compute_popd_responsibility(
            next_latents=next_latents,
            mu_old=mu_old,
            mu_teacher=mu_teacher,
            transition_variance=torch.ones(2),
            alpha=0.5,
            temperature=1.0,
        )
        self.assertGreater(result.teacher_responsibility[0].item(), 0.5)
        self.assertLess(result.teacher_responsibility[1].item(), 0.5)

    def test_temperature_event_dim_matches_latent_mean(self) -> None:
        mu_old = torch.zeros(1, 4)
        mu_teacher = torch.ones(1, 4)
        result = compute_popd_responsibility(
            next_latents=torch.zeros(1, 4),
            mu_old=mu_old,
            mu_teacher=mu_teacher,
            transition_variance=torch.ones(1),
            alpha=0.5,
            temperature=4.0,
        )
        torch.testing.assert_close(
            result.tempered_log_ratio,
            result.log_ratio_sum / result.event_dim,
        )

    def test_responsibility_is_detached(self) -> None:
        mu_old = torch.zeros(1, 2, requires_grad=True)
        mu_teacher = torch.ones(1, 2, requires_grad=True)
        result = compute_popd_responsibility(
            next_latents=torch.zeros(1, 2),
            mu_old=mu_old,
            mu_teacher=mu_teacher,
            transition_variance=torch.ones(1),
            alpha=0.5,
            temperature=1.0,
        )
        self.assertFalse(result.teacher_responsibility.requires_grad)

    def test_gaussian_mean_kl_only_updates_student(self) -> None:
        mu_student = torch.zeros(1, 2, requires_grad=True)
        mu_teacher = torch.ones(1, 2, requires_grad=True)
        loss = compute_popd_gaussian_mean_kl(
            mu_student,
            mu_teacher,
            torch.ones(1),
        ).mean()
        loss.backward()
        self.assertIsNotNone(mu_student.grad)
        self.assertIsNone(mu_teacher.grad)

    def test_diagnostics_report_joint_and_per_dimension_scales(self) -> None:
        mu_old = torch.zeros(1, 2)
        mu_teacher = torch.tensor([[2.0, 0.0]])
        next_latents = torch.tensor([[2.0, -2.0]])
        variance = torch.tensor([4.0])
        result = compute_popd_responsibility(
            next_latents=next_latents,
            mu_old=mu_old,
            mu_teacher=mu_teacher,
            transition_variance=variance,
            alpha=0.5,
            temperature=1.0,
        )
        metrics = compute_popd_diagnostics(
            next_latents=next_latents,
            mu_old=mu_old,
            mu_teacher=mu_teacher,
            mu_student=mu_old,
            transition_variance=variance,
            dt=torch.tensor([-0.25]),
            responsibility=result,
        )
        torch.testing.assert_close(metrics["old_innovation_rms"], torch.ones(1))
        torch.testing.assert_close(metrics["teacher_old_kl_joint"], torch.tensor([0.5]))
        torch.testing.assert_close(metrics["behavior_drift_rms"], torch.zeros(1))
        # K = (D/2) w^2 is the identity the temperature is calibrated from, so the whitened
        # per-dimension gap has to be reported and has to be consistent with the joint KL.
        torch.testing.assert_close(
            metrics["teacher_old_gap_whitened_rms"],
            torch.tensor([(2.0 / 4.0) ** 0.5]),
        )
        event_dim = next_latents.shape[1]
        torch.testing.assert_close(
            metrics["teacher_old_kl_joint"],
            0.5 * event_dim * metrics["teacher_old_gap_whitened_rms"].square(),
        )
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))

    def test_default_diagnostics_are_the_essential_ten_and_verbose_restores_the_rest(self) -> None:
        mu_old = torch.zeros(1, 2)
        mu_teacher = torch.tensor([[2.0, 0.0]])
        next_latents = torch.tensor([[2.0, -2.0]])
        variance = torch.tensor([4.0])
        common = dict(
            next_latents=next_latents,
            mu_old=mu_old,
            mu_teacher=mu_teacher,
            mu_student=mu_old,
            transition_variance=variance,
            dt=torch.tensor([-0.25]),
            responsibility=compute_popd_responsibility(
                next_latents=next_latents,
                mu_old=mu_old,
                mu_teacher=mu_teacher,
                transition_variance=variance,
                alpha=0.5,
                temperature=1.0,
            ),
        )
        essential = compute_popd_diagnostics(**common)
        self.assertEqual(
            sorted(essential),
            [
                "behavior_drift_rms",
                "event_dim",
                "gamma",
                "gamma_gt_099",
                "gamma_lt_001",
                "gated_mean_kl",
                "log_rho_sum",
                "old_innovation_rms",
                "teacher_old_gap_whitened_rms",
                "teacher_old_kl_joint",
                "ungated_mean_kl",
            ],
        )
        verbose = compute_popd_diagnostics(**common, verbose=True)
        self.assertLess(len(essential), len(verbose))
        self.assertTrue(set(essential).issubset(verbose))
        for dropped in (
            "alpha",
            "temperature",
            "teacher_old_kl_per_dim",
            "gate_logit",
            "student_teacher_gap_whitened_rms",
        ):
            self.assertIn(dropped, verbose)
            self.assertNotIn(dropped, essential)


class TestPOPDBehaviorTransition(unittest.TestCase):
    def test_extracts_callback_values_for_trajectory_timestep(self) -> None:
        mu_old = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        std = torch.tensor([[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]]])
        dt = -torch.ones(2, 3, 1)
        callback_map = torch.tensor([[2, 0, 1, -1], [2, 0, 1, -1]])
        result = extract_popd_behavior_transition(
            {
                "next_latents_mean": mu_old,
                "std_dev_t": std,
                "dt": dt,
                "callback_index_map": callback_map,
            },
            timestep_index=1,
        )
        torch.testing.assert_close(result.mu_old, mu_old[:, 0])
        torch.testing.assert_close(result.std_dev_t, std[:, 0])
        torch.testing.assert_close(result.dt, dt[:, 0])

    def test_rejects_missing_or_inconsistent_callback_data(self) -> None:
        valid = {
            "next_latents_mean": torch.zeros(2, 1, 3),
            "std_dev_t": torch.ones(2, 1, 1),
            "dt": -torch.ones(2, 1, 1),
            "callback_index_map": torch.zeros(2, 2, dtype=torch.long),
        }
        for missing in valid:
            batch = {key: value for key, value in valid.items() if key != missing}
            with self.subTest(missing=missing), self.assertRaises(KeyError):
                extract_popd_behavior_transition(batch, timestep_index=0)

        inconsistent = dict(valid)
        inconsistent["callback_index_map"] = torch.tensor([[0, -1], [-1, 0]])
        with self.assertRaises(ValueError):
            extract_popd_behavior_transition(inconsistent, timestep_index=0)

    def test_quantiles_use_all_per_sample_values(self) -> None:
        quantiles = compute_popd_quantiles(torch.arange(100, dtype=torch.float32))
        expected = torch.quantile(
            torch.arange(100, dtype=torch.float32),
            torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99]),
        )
        for key, value in zip(("p01", "p10", "p50", "p90", "p99"), expected):
            torch.testing.assert_close(quantiles[key], value)


def _fake_source_dataloaders():
    """Per-source 'dataloaders' as plain lists of batch dicts (re-iterable).

    ``interleaved_source_iter`` only needs ``iter(dl)`` to restart on
    exhaustion, so a list stands in for a real DataLoader. Each batch carries a
    ``metadata`` list so the ``__source__`` per-row injection path is exercised.
    """
    return {
        "geneval": [
            {"prompt": ["g0"], "metadata": [{}]},
            {"prompt": ["g1"], "metadata": [{}]},
        ],
        "ocr": [
            {"prompt": ["o0"], "metadata": [{}]},
        ],
    }


class TestXOPDInterleavedSourceIter(unittest.TestCase):
    def test_equal_round_robin_order_and_tag(self) -> None:
        dls = _fake_source_dataloaders()
        it = interleaved_source_iter(dls, source_ratio=None)
        # Sorted source order: geneval, ocr, geneval, ocr, ...
        seen = [next(it) for _ in range(4)]
        self.assertEqual([b["__source__"] for b in seen], ["geneval", "ocr", "geneval", "ocr"])
        # Source is also injected into per-row metadata.
        self.assertEqual(seen[0]["metadata"][0]["__source__"], "geneval")
        self.assertEqual(seen[1]["metadata"][0]["__source__"], "ocr")

    def test_exhausted_source_restarts(self) -> None:
        dls = _fake_source_dataloaders()  # ocr has a single batch
        it = interleaved_source_iter(dls, source_ratio=None)
        ocr_batches = [b for b in (next(it) for _ in range(6)) if b["__source__"] == "ocr"]
        # ocr re-cycles its lone batch every round (never raises StopIteration).
        self.assertEqual(len(ocr_batches), 3)
        self.assertTrue(all(b["prompt"] == ["o0"] for b in ocr_batches))

    def test_source_ratio_block_cycle(self) -> None:
        dls = _fake_source_dataloaders()
        it = interleaved_source_iter(dls, source_ratio={"geneval": 2, "ocr": 1})
        # Pattern (sorted): G G O repeating.
        order = [next(it)["__source__"] for _ in range(6)]
        self.assertEqual(order, ["geneval", "geneval", "ocr", "geneval", "geneval", "ocr"])

    def test_unknown_source_raises(self) -> None:
        dls = _fake_source_dataloaders()
        with self.assertRaises(ValueError):
            next(interleaved_source_iter(dls, source_ratio={"geneval": 1, "ocr": 1, "nope": 1}))

    def test_missing_source_raises(self) -> None:
        dls = _fake_source_dataloaders()
        with self.assertRaises(ValueError):
            next(interleaved_source_iter(dls, source_ratio={"geneval": 1}))

    def test_non_integer_weight_raises(self) -> None:
        dls = _fake_source_dataloaders()
        with self.assertRaises(ValueError):
            next(interleaved_source_iter(dls, source_ratio={"geneval": 1.5, "ocr": 1}))


class TestXOPDValidateSourceRatio(unittest.TestCase):
    def test_none_is_noop(self) -> None:
        validate_source_ratio(None, num_batches_per_epoch=6, train_dataloaders_by_source={"a": []})

    def test_empty_sources_is_noop(self) -> None:
        # Single-source mode (no per-source dataloaders) skips validation.
        validate_source_ratio({"a": 2}, num_batches_per_epoch=5, train_dataloaders_by_source={})

    def test_divisible_passes(self) -> None:
        validate_source_ratio(
            {"a": 2, "b": 1},
            num_batches_per_epoch=6,
            train_dataloaders_by_source={"a": [], "b": []},
        )

    def test_indivisible_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_source_ratio(
                {"a": 2, "b": 1},  # period 3
                num_batches_per_epoch=7,
                train_dataloaders_by_source={"a": [], "b": []},
            )

    def test_zero_sum_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_source_ratio(
                {"a": 0, "b": 0},
                num_batches_per_epoch=6,
                train_dataloaders_by_source={"a": [], "b": []},
            )


class TestExtractI2IConditionKwargs(unittest.TestCase):
    def test_t2i_batch_returns_empty(self) -> None:
        out = extract_i2i_condition_kwargs({"prompt": ["a cat"]}, prefer_latents=True)
        self.assertEqual(out, {})

    def test_prefer_latents_keeps_cached_latents(self) -> None:
        latents = torch.zeros(1, 4, 8)
        ids = torch.zeros(1, 4, 3)
        cond = [[torch.zeros(3, 8, 8)]]
        out = extract_i2i_condition_kwargs(
            {
                "image_latents": latents,
                "image_latent_ids": ids,
                "condition_images": cond,
                "images": "should_not_appear_when_prefer_latents",
            },
            prefer_latents=True,
        )
        self.assertIs(out["image_latents"], latents)
        self.assertIs(out["image_latent_ids"], ids)
        self.assertIs(out["condition_images"], cond)
        self.assertNotIn("images", out)

    def test_prefer_pixels_omits_student_latents(self) -> None:
        latents = torch.zeros(1, 4, 8)
        cond = [[torch.zeros(3, 8, 8)]]
        out = extract_i2i_condition_kwargs(
            {
                "image_latents": latents,
                "image_latent_ids": torch.zeros(1, 4, 3),
                "condition_images": cond,
            },
            prefer_latents=False,
        )
        self.assertNotIn("image_latents", out)
        self.assertNotIn("image_latent_ids", out)
        self.assertIs(out["images"], cond)

    def test_prefer_pixels_uses_raw_images_when_present(self) -> None:
        raw = [["raw.png"]]
        out = extract_i2i_condition_kwargs(
            {"images": raw, "condition_images": [[torch.zeros(3, 2, 2)]]},
            prefer_latents=False,
        )
        self.assertEqual(out["images"], raw)

    def test_non_dict_raises(self) -> None:
        with self.assertRaises(TypeError):
            extract_i2i_condition_kwargs(["not", "a", "dict"])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
