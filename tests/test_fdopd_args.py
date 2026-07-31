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

"""Tests for Flow Direct-OPD training arguments."""

import math

import pytest

from flow_factory.hparams.training_args import (
    FlowDirectOPDTrainingArguments,
    get_training_args_class,
)

DONOR_BASE = "black-forest-labs/FLUX.2-klein-base-9B"
DONOR_RL = "Tencent-Hunyuan-Multimodal-RL/" "FLUX2-klein-base-9b-GenEval2-Multi-Reward"


def _args(**overrides) -> FlowDirectOPDTrainingArguments:
    values = {
        "donor_base_model_name_or_path": DONOR_BASE,
        "donor_rl_lora_path": DONOR_RL,
    }
    values.update(overrides)
    return FlowDirectOPDTrainingArguments(**values)


def test_fdopd_arguments_are_registered() -> None:
    assert get_training_args_class("flow-direct-opd") is FlowDirectOPDTrainingArguments


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("donor_base_model_name_or_path", ""),
        ("donor_rl_lora_path", ""),
    ],
)
def test_fdopd_requires_both_donor_paths(field_name: str, field_value: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        _args(**{field_name: field_value})


@pytest.mark.parametrize("transfer_strength", [-1.0, math.nan, math.inf])
def test_fdopd_rejects_invalid_transfer_strength(transfer_strength: float) -> None:
    with pytest.raises(ValueError, match="fdopd_lambda"):
        _args(fdopd_lambda=transfer_strength)


def test_fdopd_rejects_two_trust_budgets() -> None:
    with pytest.raises(ValueError, match="at most one trust"):
        _args(
            fdopd_max_relative_delta_rms=0.1,
            fdopd_trust_kl_per_dim=0.01,
        )


def test_fdopd_guidance_and_timestep_count() -> None:
    args = _args(
        guidance_scale=1.0,
        donor_guidance_scale=4.0,
        num_inference_steps=28,
        num_fdopd_steps=4,
    )

    assert args.get_preprocess_guidance_scale() == 4.0
    assert args.get_num_train_timesteps(None) == 4


def test_fdopd_rejects_normalized_velocity_loss() -> None:
    with pytest.raises(ValueError, match="normalize_d_k"):
        _args(fdopd_loss_space="v", normalize_d_k=True)


def test_fdopd_rejects_ema_weight_swaps() -> None:
    with pytest.raises(ValueError, match="ema_decay"):
        _args(ema_decay=0.9)


def test_fdopd_defaults_to_shared_vae_klein_donor() -> None:
    args = _args()

    assert args.donor_model_type == "flux2-klein"
    assert args.donor_vae_name_or_path == "black-forest-labs/FLUX.2-klein-base-4B"
    assert args.assume_shared_vae is True
    assert args.fdopd_offload_donor_during_rollout is True
