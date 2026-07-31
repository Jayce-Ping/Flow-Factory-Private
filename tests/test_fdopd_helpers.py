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

"""Unit tests for Flow Direct-OPD target and validation helpers."""

import pytest
import torch

from flow_factory.scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    set_scheduler_timesteps,
)
from flow_factory.trainers.fdopd.common import (
    compose_fdopd_target,
    fdopd_target_diagnostics,
    select_fdopd_steps,
    synchronize_fdopd_scheduler_state,
    validate_fdopd_transition_stats,
)


def test_zero_donor_delta_recovers_recipient_base() -> None:
    recipient_base = torch.tensor([[1.0, 2.0]])
    donor_base = torch.tensor([[3.0, 4.0]])

    result = compose_fdopd_target(
        recipient_base=recipient_base,
        donor_base=donor_base,
        donor_rl=donor_base.clone(),
        transfer_strength=1.0,
    )

    torch.testing.assert_close(result.target, recipient_base)
    torch.testing.assert_close(result.lambda_eff, torch.ones(1))


def test_target_uses_rl_minus_base_sign_in_fp32() -> None:
    recipient_base = torch.tensor([[10.0, 20.0]], dtype=torch.bfloat16)
    donor_base = torch.tensor([[4.0, 8.0]], dtype=torch.bfloat16)
    donor_rl = torch.tensor([[6.0, 5.0]], dtype=torch.bfloat16)

    result = compose_fdopd_target(
        recipient_base=recipient_base,
        donor_base=donor_base,
        donor_rl=donor_rl,
        transfer_strength=0.5,
        compute_delta_fp32=True,
    )

    assert result.target.dtype == torch.float32
    torch.testing.assert_close(result.donor_delta, torch.tensor([[2.0, -3.0]]))
    torch.testing.assert_close(result.target, torch.tensor([[11.0, 18.5]]))
    assert result.target.requires_grad is False


def test_relative_rms_trust_clips_each_sample() -> None:
    result = compose_fdopd_target(
        recipient_base=torch.tensor([[2.0, 2.0]]),
        donor_base=torch.zeros(1, 2),
        donor_rl=torch.tensor([[4.0, 0.0]]),
        transfer_strength=1.0,
        max_relative_delta_rms=0.5,
    )

    shift_rms = (result.target - torch.tensor([[2.0, 2.0]])).square().mean().sqrt()
    base_rms = torch.tensor([[2.0, 2.0]]).square().mean().sqrt()
    torch.testing.assert_close(shift_rms / base_rms, torch.tensor(0.5))
    assert bool(result.clipped.item()) is True


def test_kl_per_dim_trust_uses_transition_variance() -> None:
    result = compose_fdopd_target(
        recipient_base=torch.zeros(1, 2),
        donor_base=torch.zeros(1, 2),
        donor_rl=torch.tensor([[2.0, 0.0]]),
        transfer_strength=1.0,
        trust_kl_per_dim=0.25,
        transition_variance=torch.ones(1),
    )

    torch.testing.assert_close(result.unit_kl_per_dim, torch.ones(1))
    torch.testing.assert_close(result.lambda_eff, torch.full((1,), 0.5))
    torch.testing.assert_close(result.target, torch.tensor([[1.0, 0.0]]))


def test_transition_stats_reject_covariance_mismatch() -> None:
    with pytest.raises(ValueError, match="transition std mismatch"):
        validate_fdopd_transition_stats(
            recipient_std=torch.tensor([0.2]),
            donor_std=torch.tensor([0.3]),
            recipient_dt=torch.tensor([-0.1]),
            donor_dt=torch.tensor([-0.1]),
            context="step=3",
        )


def test_stratified_step_selection_covers_every_segment() -> None:
    selected = select_fdopd_steps(
        pool=list(range(28)),
        num_steps=4,
        strategy="stratified",
        seed=123,
    )

    assert len(selected) == 4
    assert 0 <= selected[0] < 7
    assert 7 <= selected[1] < 14
    assert 14 <= selected[2] < 21
    assert 21 <= selected[3] < 28


def test_diagnostics_are_detached_per_sample_scalars() -> None:
    result = compose_fdopd_target(
        recipient_base=torch.tensor([[1.0, 1.0]], requires_grad=True),
        donor_base=torch.zeros(1, 2, requires_grad=True),
        donor_rl=torch.ones(1, 2, requires_grad=True),
        transfer_strength=0.25,
    )

    diagnostics = fdopd_target_diagnostics(result, recipient_base=torch.ones(1, 2))

    assert diagnostics["delta_rms"].shape == (1,)
    assert diagnostics["lambda_eff"].shape == (1,)
    torch.testing.assert_close(
        diagnostics["relative_target_shift_rms"],
        torch.tensor([0.25]),
    )
    assert all(value.requires_grad is False for value in diagnostics.values())


def test_scheduler_sync_copies_exact_recipient_grid() -> None:
    recipient = FlowMatchEulerDiscreteSDEScheduler(
        noise_level=0.7,
        sde_steps=[1, 4],
        num_sde_steps=2,
        seed=123,
    )
    donor = FlowMatchEulerDiscreteSDEScheduler(
        noise_level=0.7,
        sde_steps=[0],
        num_sde_steps=1,
        seed=999,
    )
    set_scheduler_timesteps(recipient, 6, seq_len=256, device="cpu")
    set_scheduler_timesteps(donor, 3, seq_len=256, device="cpu")

    synchronize_fdopd_scheduler_state(recipient, donor)

    torch.testing.assert_close(donor.timesteps, recipient.timesteps)
    torch.testing.assert_close(donor.sigmas, recipient.sigmas)
    torch.testing.assert_close(donor.train_timesteps, recipient.train_timesteps)
    assert donor.num_inference_steps == 6
    assert donor._step_index is None
    assert donor._begin_index is None
