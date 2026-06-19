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

# src/flow_factory/trainers/ppo/gae.py
"""Generalized Advantage Estimation (GAE) for critic-based PPO.

The denoising trajectory is treated as an ordered RL episode over the SDE steps:
``values[:, t]`` is the value of the state at ordered SDE step ``t``, the reward is
received only at the terminal step, and the terminal state's value bootstrap is 0.
This matches a terminal-only reward (``gamma`` close to 1). The example config uses
a full, contiguous SDE trajectory so the per-step bootstrap ``values[:, t+1]`` is the
value of the actual next latent.
"""
from __future__ import annotations

from typing import Tuple

import torch
from accelerate import Accelerator

from ...utils.dist import global_tensor_stats


def compute_gae(
    values: torch.Tensor,
    terminal_rewards: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-step advantages and returns for a terminal-only reward.

    Args:
        values: ``(B, S)`` old value estimates over the ``S`` ordered SDE steps.
        terminal_rewards: ``(B,)`` scalar reward per sample (received at step ``S-1``).
        gamma: Discount factor.
        lam: GAE lambda.

    Returns:
        Tuple ``(advantages, returns)`` each of shape ``(B, S)``.

    Raises:
        ValueError: If shapes are inconsistent.
    """
    if values.ndim != 2:
        raise ValueError(f"`values` must be (B, S), got shape {tuple(values.shape)}.")
    batch_size, num_steps = values.shape
    if terminal_rewards.shape != (batch_size,):
        raise ValueError(
            f"`terminal_rewards` must be ({batch_size},), got " f"{tuple(terminal_rewards.shape)}."
        )

    values = values.float()
    terminal_rewards = terminal_rewards.float()
    advantages = torch.zeros_like(values)
    last_adv = torch.zeros(batch_size, dtype=values.dtype, device=values.device)
    for step in reversed(range(num_steps)):
        if step == num_steps - 1:
            reward = terminal_rewards
            next_value = torch.zeros_like(last_adv)
        else:
            reward = torch.zeros_like(last_adv)
            next_value = values[:, step + 1]
        delta = reward + gamma * next_value - values[:, step]
        last_adv = delta + gamma * lam * last_adv
        advantages[:, step] = last_adv

    returns = advantages + values
    return advantages, returns


def whiten(
    advantages: torch.Tensor,
    accelerator: Accelerator,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Globally whiten advantages across all ``(sample, step)`` pairs and ranks.

    Uses a single all-reduce of ``(count, sum, sum_sq)`` to obtain the global
    mean/std, then standardizes. All ranks must call this together.

    Args:
        advantages: ``(B, S)`` per-step advantages on this rank.
        accelerator: Accelerator for the cross-rank reduction.
        eps: Numerical floor added to the std.

    Returns:
        Whitened advantages, same shape and dtype as ``advantages``.
    """
    stats = global_tensor_stats(accelerator, advantages.reshape(-1))
    return (advantages - stats["mean"]) / (stats["std"] + eps)
