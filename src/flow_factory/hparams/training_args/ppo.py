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

"""Training arguments for critic-based PPO."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from ._base import TrainingArguments, _standardize_clip_range


@dataclass
class PPOTrainingArguments(TrainingArguments):
    r"""Training arguments for critic-based PPO (actor-critic with a value net + GAE).

    PPO is a **coupled** algorithm: it reuses the SDE rollout / per-step log-prob
    machinery (like GRPO) but replaces the single broadcast advantage with a value
    critic that provides a per-step baseline and GAE per-step advantages.
    """

    # --- Policy clipping (per-step PPO ratio; flow log-probs are large so the default is tiny) ---
    clip_range: tuple[float, float] = field(
        default=(-1e-4, 1e-4),
        metadata={"help": "Clipping range for the per-step PPO policy ratio."},
    )
    adv_clip_range: tuple[float, float] = field(
        default=(-5.0, 5.0),
        metadata={"help": "Clipping range for per-step advantages."},
    )
    normalize_advantage: bool = field(
        default=True,
        metadata={
            "help": "Whiten per-step advantages globally (across ranks) before the policy loss."
        },
    )

    # --- Critic / value loss ---
    critic_learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for the value critic (uses its own optimizer)."},
    )
    vf_coef: float = field(
        default=0.5,
        metadata={"help": "Weight of the value (critic) loss in the combined objective."},
    )
    value_clip_range: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "PPO value-clipping epsilon around old values. None disables value clipping."
        },
    )
    critic_warmup_steps: int = field(
        default=30,
        metadata={
            "help": "Number of initial optimizer steps that update only the critic (policy frozen)."
        },
    )

    # --- Critic architecture ---
    critic_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden width of the value critic."},
    )
    critic_num_layers: int = field(
        default=3,
        metadata={"help": "Number of residual MLP blocks in the critic's per-token encoder."},
    )
    critic_time_embed_dim: int = field(
        default=256,
        metadata={"help": "Sinusoidal timestep embedding dim fed to the critic head."},
    )
    critic_attn_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads in the critic's attention-pooling layer."},
    )
    critic_num_query_tokens: int = field(
        default=1,
        metadata={"help": "Number of learnable query tokens for the critic's attention pooling."},
    )

    # --- GAE ---
    gae_gamma: float = field(
        default=1.0,
        metadata={"help": "GAE discount factor per SDE step. 1.0 fits a terminal-only reward."},
    )
    gae_lambda: float = field(
        default=0.95,
        metadata={"help": "GAE lambda (bias/variance trade-off)."},
    )

    # --- Optional KL-to-reference (DPOK-style) ---
    kl_type: Literal["v-based", "x-based"] = field(
        default="x-based",
        metadata={
            "help": "KL divergence space vs the reference policy. 'v-based': velocity, 'x-based': latent."
        },
    )
    kl_beta: float = field(
        default=0.0,
        metadata={"help": "KL-vs-reference penalty beta. 0 to disable (and save memory)."},
    )
    ref_param_device: Literal["cpu", "cuda"] = field(
        default="cuda",
        metadata={"help": "Device to store reference parameters (full finetune + kl_beta>0 only)."},
    )

    def __post_init__(self):
        super().__post_init__()
        # Explicit float() casts guard against scientific-notation strings from CLI/YAML overrides.
        self.critic_learning_rate = float(self.critic_learning_rate)
        self.vf_coef = float(self.vf_coef)
        self.gae_gamma = float(self.gae_gamma)
        self.gae_lambda = float(self.gae_lambda)
        self.kl_beta = float(self.kl_beta)
        self.clip_range = _standardize_clip_range(self.clip_range, "clip_range")
        self.adv_clip_range = _standardize_clip_range(self.adv_clip_range, "adv_clip_range")
        if self.value_clip_range is not None:
            self.value_clip_range = float(self.value_clip_range)
            if self.value_clip_range <= 0:
                raise ValueError(
                    f"`value_clip_range` must be > 0 or None to disable, got {self.value_clip_range}."
                )
        if self.kl_type not in ["v-based", "x-based"]:
            raise ValueError(
                f"Invalid kl_type: {self.kl_type}. Valid options: ['v-based', 'x-based']."
            )
        if self.critic_num_query_tokens < 1:
            raise ValueError(
                f"`critic_num_query_tokens` must be >= 1, got {self.critic_num_query_tokens}."
            )
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"`gae_lambda` must be in [0, 1], got {self.gae_lambda}.")

    def get_num_train_timesteps(self, args: Any) -> int:
        """PPO accumulates one backward per SDE step, so the GAS multiplier is num_sde_steps."""
        return args.scheduler_args.num_sde_steps
