# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Unit tests for trajectory-DM helpers and Approach-B / TDM config wiring."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from flow_factory.hparams import (
    Arguments,
    XOPDDMTrainingArguments,
    XTDMTrainingArguments,
    get_training_args_class,
)
from flow_factory.trainers.registry import get_trainer_class
from flow_factory.trainers.xopd.traj_dm import (
    add_noise_rf,
    dm_stopgrad_loss,
    ode_euler_next,
    pseudo_huber_c_default,
    pseudo_huber_from_residual,
    self_normalized_dm_grad,
    x0_from_velocity,
)


class TestTrajDMHelpers(unittest.TestCase):
    def test_dm_stopgrad_identity_grad(self) -> None:
        x = torch.randn(2, 3, 4, requires_grad=True)
        grad = torch.randn_like(x)
        loss = dm_stopgrad_loss(x, grad)
        (gx,) = torch.autograd.grad(loss, x)
        # ∂/∂x of 0.5 * mean_{b,s} (x - sg(x-grad))^2 equals grad / (B * S)
        spatial = int(x[0].numel())
        expected = grad / (x.shape[0] * spatial)
        torch.testing.assert_close(gx, expected)
        cos = torch.nn.functional.cosine_similarity(gx.flatten(), grad.flatten(), dim=0)
        self.assertGreater(float(cos), 0.99)

    def test_self_normalized_dm_grad_shape(self) -> None:
        p_r = torch.ones(2, 3)
        p_f = torch.zeros(2, 3)
        g = self_normalized_dm_grad(p_r, p_f)
        self.assertEqual(g.shape, p_r.shape)
        # mean|p_real|=1 -> grad = 1
        torch.testing.assert_close(g, torch.ones_like(p_r))

    def test_x0_and_add_noise_roundtrip_sigma(self) -> None:
        x0 = torch.randn(2, 4)
        t = torch.tensor([500.0, 500.0])
        xt, eps = add_noise_rf(x0, t)
        # velocity target for RF: eps - x0; recover x0 from v at xt
        v = eps - x0
        x0_hat = x0_from_velocity(xt, v, t)
        torch.testing.assert_close(x0_hat, x0, atol=1e-5, rtol=1e-5)

    def test_ode_euler_next_decreases_sigma(self) -> None:
        z = torch.randn(2, 3)
        v = -z  # toward zero
        t = torch.tensor([1000.0, 1000.0])
        t_next = torch.tensor([500.0, 500.0])
        z_next = ode_euler_next(z, v, t, t_next)
        # dt_sigma = σ_next - σ = -0.5; z_next = z + v*(-0.5) = z + (-z)*(-0.5) = 1.5 z
        torch.testing.assert_close(z_next, 1.5 * z)

    def test_pseudo_huber_positive(self) -> None:
        r = torch.randn(2, 8)
        c = pseudo_huber_c_default(8)
        loss = pseudo_huber_from_residual(r, c)
        self.assertGreater(float(loss), 0.0)


class TestTrajDMRegistryAndConfigs(unittest.TestCase):
    def test_training_args_registry(self) -> None:
        self.assertIs(get_training_args_class("xopd_dm"), XOPDDMTrainingArguments)
        self.assertIs(get_training_args_class("xtdm"), XTDMTrainingArguments)

    def test_trainer_registry(self) -> None:
        from flow_factory.trainers.xopd.traj_dm_trainer import XOPDDMTrainer, XTDMTrainer

        self.assertIs(get_trainer_class("xopd_dm"), XOPDDMTrainer)
        self.assertIs(get_trainer_class("xtdm"), XTDMTrainer)

    def test_xtdm_defaults_pseudo_huber(self) -> None:
        # Minimal construct via dataclass defaults (teacher path required by XOPD __post_init__).
        args = XTDMTrainingArguments(
            trainer_type="xtdm",
            teacher_model_name_or_path="dummy/teacher",
            assume_shared_vae_text_encoder=True,
        )
        self.assertEqual(args.tdm_loss_metric, "pseudo_huber")
        self.assertEqual(args.tdm_sim_steps, 4)

    def test_opddm_defaults_mse(self) -> None:
        args = XOPDDMTrainingArguments(
            trainer_type="xopd_dm",
            teacher_model_name_or_path="dummy/teacher",
            assume_shared_vae_text_encoder=True,
        )
        self.assertEqual(args.tdm_loss_metric, "mse")

    def test_smoke_yamls_parse(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "xopd_configs" / "ode_pathwise"
        paths = [
            config_dir / "flux2_klein_32b_to_4b_opddm_smoke.yaml",
            config_dir / "flux2_klein_32b_to_4b_tdm_smoke.yaml",
        ]
        with (
            patch.dict(os.environ, {"WORLD_SIZE": "8"}),
            patch("os.makedirs"),
        ):
            for path in paths:
                args = Arguments.load_from_yaml(str(path))
                self.assertIn(args.training_args.trainer_type, ("xopd_dm", "xtdm"))
                self.assertEqual(args.training_args.tdm_fake_ratio, 6)
                self.assertEqual(args.training_args.num_batches_per_epoch, 6)
                self.assertEqual(args.training_args.gradient_accumulation_steps, 1)
                self.assertEqual(args.scheduler_args.dynamics_type, "ODE")


if __name__ == "__main__":
    unittest.main()
