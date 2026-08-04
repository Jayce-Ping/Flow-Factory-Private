# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Focused trainer integration tests for Arm-A2 forward-risk XOPD."""

import unittest
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from flow_factory.trainers.xopd import trainer as xopd_trainer_module
from flow_factory.trainers.xopd.trainer import XOPDTrainer
from flow_factory.utils.noise_schedule import flow_match_sigma


class _AttrDict(dict):
    def __getattr__(self, name):
        return self[name]


class _ForwardRiskAdapter:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.scheduler = SimpleNamespace(
            dynamics_type="ODE",
            noise_level=0.0,
            num_train_timesteps=1000,
        )
        self.trainable_components = (self,)
        self.source = "student"
        self.calls = []

    @contextmanager
    def use_teacher_transformer(self):
        previous = self.source
        self.source = "teacher"
        try:
            yield
        finally:
            self.source = previous

    def predict_velocity(self, *, latents, **kwargs):
        self.calls.append(
            {
                "source": self.source,
                "latents": latents.detach().clone(),
                "grad_enabled": torch.is_grad_enabled(),
                "kwargs": kwargs,
            }
        )
        if self.source == "teacher":
            return torch.ones_like(latents)
        return self.weight.expand_as(latents)

    def get_trainable_parameters(self):
        return [self.weight]

    def collect_moe_aux_loss(self):
        return None

    def collect_router_z_loss(self):
        return None

    def collect_weight_sum_penalty(self):
        return None


class _Accelerator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.process_index = 0
        self.num_processes = 1
        self.is_local_main_process = False
        self.sync_gradients = False
        self.accumulate_calls = 0
        self.backward_calls = 0

    @contextmanager
    def accumulate(self, *components):
        self.accumulate_calls += 1
        self.sync_gradients = True
        yield

    def backward(self, loss):
        self.backward_calls += 1
        loss.backward()

    def gather(self, value):
        return value


class TestForwardRiskTrainer(unittest.TestCase):
    def _trainer(self) -> XOPDTrainer:
        trainer = XOPDTrainer.__new__(XOPDTrainer)
        trainer.adapter = _ForwardRiskAdapter()
        trainer.accelerator = _Accelerator()
        trainer.optimizer = torch.optim.SGD([trainer.adapter.weight], lr=0.1)
        trainer.training_args = _AttrDict(
            seed=17,
            kl_beta=0.0,
            kl_type="x-based",
            moe_load_balance_coeff=0.0,
            router_z_loss_coeff=0.0,
            mof_weight_sum_penalty_coeff=0.0,
            max_grad_norm=1.0,
        )
        trainer.epoch = 3
        trainer.step = 0
        trainer.log_args = SimpleNamespace(verbose=False)
        trainer.pathwise_coef = 1.0
        trainer.student_gs = 1.0
        trainer.teacher_gs = 1.0
        trainer.forward_risk_alpha = 0.5
        trainer.forward_risk_temperature = 0.1
        trainer.forward_risk_max_delta_rms = 0.25
        trainer._is_forward_risk = True
        trainer._is_marginal_cfm = False
        trainer._is_popd = False
        trainer._is_pdm = False
        trainer._is_ode = True
        trainer._cross_vae = False
        trainer._pixel_loss = False
        trainer._is_hsct = False
        trainer.detail_mask_enabled = False
        trainer._velocity_param_names = frozenset(
            {
                "t",
                "latents",
                "latent_ids",
                "prompt_embeds",
                "text_ids",
                "guidance_scale",
            }
        )
        trainer._velocity_accepts_var_kwargs = False
        trainer.autocast = nullcontext
        trainer._clip_grad_norm_ep_aware = lambda parameters, max_norm: torch.tensor(0.5)
        trainer.log_data = lambda *args, **kwargs: None
        return trainer

    @staticmethod
    def _batch():
        return {
            "timesteps": torch.tensor([[500.0], [500.0]]),
            "all_latents": torch.tensor(
                [
                    [[9.0, 9.0, 9.0], [1.0, 2.0, 3.0]],
                    [[8.0, 8.0, 8.0], [4.0, 5.0, 6.0]],
                ]
            ),
            "prompt_embeds": torch.zeros(2, 2, 3),
            "text_ids": torch.zeros(2, 2, dtype=torch.long),
            "latent_ids": torch.zeros(2, 3, 4),
            "teacher_prompt_embeds": torch.ones(2, 2, 3),
            "teacher_text_ids": torch.ones(2, 2, dtype=torch.long),
        }

    def test_probe_is_deterministic_and_uses_clean_endpoint(self) -> None:
        trainer = self._trainer()
        batch = self._batch()
        t = batch["timesteps"][:, 0]
        x_s_first, target_first = trainer._build_forward_risk_probe(
            batch=batch,
            t=t,
            batch_index=2,
            timestep_index=0,
        )
        x_s_second, target_second = trainer._build_forward_risk_probe(
            batch=batch,
            t=t,
            batch_index=2,
            timestep_index=0,
        )
        torch.testing.assert_close(x_s_first, x_s_second)
        torch.testing.assert_close(target_first, target_second)

        x_data = batch["all_latents"][:, -1]
        epsilon = target_first + x_data
        sigma = flow_match_sigma(t).view(2, 1)
        torch.testing.assert_close(
            x_s_first,
            (1.0 - sigma) * x_data + sigma * epsilon,
        )

    def test_teacher_and_student_query_identical_probe_state(self) -> None:
        trainer = self._trainer()
        batch = self._batch()
        t = batch["timesteps"][:, 0]
        x_s, _ = trainer._build_forward_risk_probe(
            batch=batch,
            t=t,
            batch_index=0,
            timestep_index=0,
        )
        student, teacher = trainer._forward_risk_velocity_predictions(
            batch=batch,
            t=t,
            x_s=x_s,
            teacher_text_cond=trainer._build_teacher_text_cond(batch),
        )
        self.assertTrue(student.requires_grad)
        self.assertFalse(teacher.requires_grad)
        self.assertEqual([call["source"] for call in trainer.adapter.calls], ["teacher", "student"])
        torch.testing.assert_close(
            trainer.adapter.calls[0]["latents"],
            trainer.adapter.calls[1]["latents"],
        )
        self.assertFalse(trainer.adapter.calls[0]["grad_enabled"])
        self.assertTrue(trainer.adapter.calls[1]["grad_enabled"])

    def test_optimize_pass_updates_once_and_logs_calibration_metrics(self) -> None:
        trainer = self._trainer()
        batch = self._batch()
        logged = []
        trainer.log_data = lambda data, step: logged.append((data, step))

        def reduce_loss(_accelerator, values):
            return {
                key: torch.cat(items).float().mean()
                if items[0].ndim > 0
                else torch.stack(items).float().mean()
                for key, items in values.items()
            }

        initial = trainer.adapter.weight.detach().clone()
        with patch.object(xopd_trainer_module, "reduce_loss_info", reduce_loss):
            remaining = trainer._optimize_train_pass(
                batch=batch,
                latents_index_map=torch.tensor([0, 1]),
                num_timesteps=1,
                loss_info=defaultdict(list),
                mu_teacher_list=None,
                pdm_teacher_list=None,
                popd_cache_list=None,
                timestep_indices=[0],
                callback_index_map=None,
                batch_index=0,
            )

        self.assertEqual(remaining, {})
        self.assertEqual(trainer.accelerator.backward_calls, 1)
        self.assertEqual(trainer.step, 1)
        self.assertFalse(torch.equal(initial, trainer.adapter.weight.detach()))
        self.assertEqual(len(logged), 1)
        logged_metrics, logged_step = logged[0]
        self.assertEqual(logged_step, 0)
        self.assertIn("train/forward_risk/advantage", logged_metrics)
        self.assertIn("train/forward_risk/advantage_p90", logged_metrics)
        self.assertIn("train/forward_risk/teacher_delta_rms_p90", logged_metrics)
        self.assertIn(
            "train/forward_risk/teacher_preferred_fraction",
            logged_metrics,
        )
        self.assertIn(
            "train/forward_risk/student_preferred_fraction",
            logged_metrics,
        )
        timestep_metrics = [
            key
            for key in logged_metrics
            if any(
                segment.startswith("t")
                and segment[1:].split("_", maxsplit=1)[0].isdigit()
                for segment in key.split("/")
            )
        ]
        self.assertEqual(timestep_metrics, [])
