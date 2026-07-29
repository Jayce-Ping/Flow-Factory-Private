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

import torch

from flow_factory.hparams.training_args import XOPDTrainingArguments
from flow_factory.trainers.xopd.common import (
    align_l0_inner_steps,
    compute_per_step_kl,
    compute_popd_diagnostics,
    compute_popd_gaussian_mean_kl,
    compute_popd_quantiles,
    compute_popd_responsibility,
    compute_transition_variance,
    extract_i2i_condition_kwargs,
    extract_popd_behavior_transition,
    interleaved_source_iter,
    l0_loss_weight,
    validate_l1_one_step_per_epoch,
    validate_popd_configuration,
    validate_source_ratio,
)


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

    def test_popd_defaults_preserve_direct_xopd(self) -> None:
        args = XOPDTrainingArguments(teacher_model_name_or_path="/tmp/teacher")
        self.assertEqual(args.xopd_target_mode, "direct")
        self.assertEqual(args.popd_alpha, 0.5)
        self.assertEqual(args.popd_temperature, 1.0)

    def test_direct_xopd_ignores_unused_popd_values(self) -> None:
        args = XOPDTrainingArguments(
            teacher_model_name_or_path="/tmp/teacher",
            xopd_target_mode="direct",
            popd_alpha=0.0,
            popd_temperature=0.0,
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
        torch.testing.assert_close(metrics["teacher_old_gap_l2"], torch.tensor([2.0]))
        torch.testing.assert_close(
            metrics["teacher_old_gap_rms"],
            torch.tensor([2.0**0.5]),
        )
        torch.testing.assert_close(metrics["teacher_old_kl_joint"], torch.tensor([0.5]))
        torch.testing.assert_close(metrics["teacher_old_kl_per_dim"], torch.tensor([0.25]))
        torch.testing.assert_close(metrics["behavior_drift_rms"], torch.zeros(1))
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))


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
