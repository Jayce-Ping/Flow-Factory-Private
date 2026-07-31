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

"""Structural and lightweight wiring tests for FlowDirectOPDTrainer."""

from contextlib import contextmanager, nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch

from flow_factory.trainers.abc import BaseTrainer
from flow_factory.trainers.fdopd.common import validate_fdopd_runtime
from flow_factory.trainers.fdopd.trainer import FlowDirectOPDTrainer
from flow_factory.trainers.registry import get_trainer_class


def test_fdopd_trainer_is_registered_and_direct_base_subclass() -> None:
    assert get_trainer_class("flow-direct-opd") is FlowDirectOPDTrainer
    assert FlowDirectOPDTrainer.__bases__ == (BaseTrainer,)


def test_fdopd_builds_cached_donor_cfg_conditioning() -> None:
    trainer = object.__new__(FlowDirectOPDTrainer)
    trainer.donor_guidance_scale = 4.0
    batch = {
        "teacher_prompt_embeds": torch.ones(2, 3, 4),
        "teacher_text_ids": torch.ones(2, 3, 4),
        "teacher_negative_prompt_embeds": torch.zeros(2, 3, 4),
        "teacher_negative_text_ids": torch.zeros(2, 3, 4),
    }

    conditioning = trainer._build_donor_text_conditioning(batch)

    assert set(conditioning) == {
        "prompt_embeds",
        "text_ids",
        "negative_prompt_embeds",
        "negative_text_ids",
    }


def test_runtime_rejects_velocity_loss_under_sde() -> None:
    with pytest.raises(ValueError, match="fdopd_loss_space='xt'"):
        validate_fdopd_runtime(
            dynamics_type="Flow-SDE",
            loss_space="v",
            normalize_d_k=False,
            trust_kl_per_dim=None,
            max_relative_delta_rms=0.1,
        )


def test_runtime_rejects_kl_trust_under_ode() -> None:
    with pytest.raises(ValueError, match="deterministic ODE"):
        validate_fdopd_runtime(
            dynamics_type="ODE",
            loss_space="v",
            normalize_d_k=False,
            trust_kl_per_dim=0.01,
            max_relative_delta_rms=None,
        )


def test_trainer_rejects_loaded_lora_from_wrong_donor_base() -> None:
    component = SimpleNamespace(
        peft_config={
            "default": SimpleNamespace(
                base_model_name_or_path="stabilityai/stable-diffusion-3.5-medium"
            )
        }
    )
    donor = SimpleNamespace(get_component_unwrapped=lambda name: component)

    with pytest.raises(ValueError, match="base_model_name_or_path"):
        FlowDirectOPDTrainer._validate_loaded_donor_base(
            donor,
            expected_base="black-forest-labs/FLUX.2-klein-base-9B",
        )


def test_prepass_composes_recipient_base_plus_donor_rl_shift() -> None:
    def output(velocity):
        value = torch.tensor([velocity], dtype=torch.float32)
        return SimpleNamespace(
            noise_pred=value,
            next_latents_mean=value * 0.1,
            std_dev_t=torch.zeros(1),
            dt=torch.full((1,), -0.1),
        )

    class Recipient:
        scheduler = SimpleNamespace(dynamics_type="ODE")

        @contextmanager
        def use_ref_parameters(self):
            yield

        def forward(self, **kwargs):
            return output([10.0, 10.0])

    class Donor:
        def __init__(self):
            self.base = False

        @contextmanager
        def use_ref_parameters(self):
            self.base = True
            try:
                yield
            finally:
                self.base = False

        def forward(self, **kwargs):
            return output([2.0, 2.0] if self.base else [4.0, 1.0])

    trainer = object.__new__(FlowDirectOPDTrainer)
    trainer.adapter = Recipient()
    trainer.donor_adapter = Donor()
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        is_local_main_process=False,
    )
    trainer.log_args = SimpleNamespace(verbose=False)
    trainer.autocast = nullcontext
    trainer.epoch = 0
    trainer.loss_space = "v"
    trainer.donor_guidance_scale = 1.0
    trainer._donor_forward_param_names = frozenset()
    trainer._donor_forward_accepts_var_kwargs = True
    trainer.training_args = SimpleNamespace(
        fdopd_lambda=0.5,
        fdopd_compute_delta_fp32=True,
        fdopd_max_relative_delta_rms=None,
        fdopd_trust_kl_per_dim=None,
    )
    trainer._sync_donor_scheduler = MethodType(lambda self, latents: None, trainer)
    trainer._step_inputs = MethodType(
        lambda self, **kwargs: (
            torch.tensor([500.0]),
            torch.zeros(1, 2),
            {"latents": torch.zeros(1, 2)},
        ),
        trainer,
    )

    caches = trainer._precompute_step_caches(
        batch={
            "teacher_prompt_embeds": torch.ones(1, 2, 3),
            "teacher_text_ids": torch.ones(1, 2, 4),
        },
        latents_index_map=torch.tensor([0, 1]),
        num_timesteps=1,
        timestep_indices=[0],
    )

    assert len(caches) == 1
    torch.testing.assert_close(
        caches[0].target.target,
        torch.tensor([[11.0, 9.5]]),
    )


def test_xt_loss_uses_cached_cps_variance_directly() -> None:
    loss = FlowDirectOPDTrainer._compute_xt_loss(
        actor_mean=torch.tensor([[2.0, 0.0]]),
        target_mean=torch.zeros(1, 2),
        transition_variance=torch.tensor([4.0]),
        normalize=True,
    )

    torch.testing.assert_close(loss, torch.tensor([0.25]))


def test_trainer_rejects_requested_steps_larger_than_rollout_pool() -> None:
    trainer = object.__new__(FlowDirectOPDTrainer)
    trainer._is_ode = True
    trainer.epoch = 0
    trainer._fdopd_timestep_cache_epoch = None
    trainer.training_args = SimpleNamespace(
        num_inference_steps=2,
        fdopd_train_steps=None,
        num_fdopd_steps=4,
        fdopd_step_sampling="uniform",
        seed=42,
    )

    with pytest.raises(ValueError, match="cannot exceed"):
        _ = trainer._train_timestep_indices


@pytest.mark.parametrize(
    ("model_type", "finetune_type", "cpu_efficient"),
    [
        ("flux2", "full", False),
        ("flux2-klein", "lora", False),
        ("flux2", "lora", True),
    ],
)
def test_trainer_rejects_unsupported_recipient_configuration(
    model_type: str,
    finetune_type: str,
    cpu_efficient: bool,
) -> None:
    model_args = SimpleNamespace(
        model_type=model_type,
        finetune_type=finetune_type,
    )
    adapter = SimpleNamespace(
        _is_fsdp_cpu_efficient_loading=lambda: cpu_efficient,
    )

    with pytest.raises(ValueError, match="recipient"):
        FlowDirectOPDTrainer._validate_recipient_configuration(model_args, adapter)
