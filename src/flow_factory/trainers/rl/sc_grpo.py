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

# src/flow_factory/trainers/rl/sc_grpo.py
"""
Score-Centered GRPO (SC-GRPO) Trainer.

SC-GRPO keeps GRPO's group advantages and the optional KL-vs-reference penalty,
but optimizes the score-centered policy gradient instead of the PPO clipped ratio.
Each SDE transition is Gaussian with a policy-independent variance, so the
rollout-expected log-likelihood has the closed form
``E_q[log p_theta] = -KL(q || p_theta) + const`` and the per-step loss is

    L = -A * (log p_theta(x') + KL(q || p_theta)),

where ``q`` is the rollout transition (stored ``next_latents_mean``) and ``x'`` the
stored next latent. The gradient ``-A * (score - E_q[score])`` has zero mean under
``q`` for a constant advantage, which cancels the drift toward the sampler caused
by training-inference mismatch.
"""

from collections import defaultdict
from functools import partial
from typing import List

import torch
import tqdm as tqdm_

from ...hparams import SCGRPOTrainingArguments
from ...samples import BaseSample, LatentState, MultiModalStepOutput, ReplayStep
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import compute_trajectory_indices
from ..abc import BaseTrainer
from ..coupled import CoupledReplayRuntimeMixin
from ..registry import register_trainer

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)
logger = setup_logger(__name__)


# ============================ SC-GRPO Trainer ============================
@register_trainer("sc-grpo")
class SCGRPOTrainer(CoupledReplayRuntimeMixin, BaseTrainer):
    """SC-GRPO Trainer: score-centered policy gradient on SDE transitions.

    References:
    [1] Score Centering Stabilizes Off-policy Reinforcement Learning
        - https://arxiv.org/abs/2609.20807
    [2] Flow-GRPO: Training Flow Matching Models via Online RL
        - https://arxiv.org/abs/2505.05470
    """

    # Coupled paradigm: the loss differentiates the stored SDE transition, so ODE
    # dynamics and lossy rollout acceleration are rejected (constraints.md #7).
    paradigm = "coupled"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: SCGRPOTrainingArguments
        self.num_train_timesteps = self.adapter.scheduler.num_sde_steps

    @property
    def enable_kl_loss(self) -> bool:
        """Check if the KL-vs-reference penalty is enabled."""
        return self.training_args.kl_beta > 0.0

    def _log_prob_quadratic_coef(
        self, component: str, std_dev_t: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """Return ``c`` in the scheduler convention ``log_prob = mean(-c * (x - mean)^2) + const``.

        Flow-SDE / Dance-SDE use the normalized Gaussian ``c = 1 / (2 * scale^2)``;
        CPS uses the unnormalized ``c = 1``. The scale is validated either way so a
        deterministic (zero-scale) or ODE transition fails fast.

        Args:
            component: Trajectory component name owning the scheduler.
            std_dev_t: Per-step diffusion std from the policy forward.
            dt: Per-step time delta from the policy forward (negative).

        Returns:
            Coefficient tensor broadcastable to the component latent shape.
        """
        scale = self._effective_transition_std(
            component, std_dev_t, dt, context="SC-GRPO sampler KL"
        )
        if self.adapter.scheduler_group[component].dynamics_type == "CPS":
            return torch.ones_like(scale, dtype=torch.float32)
        return 1.0 / (2.0 * scale.float() ** 2)

    def _sampler_kl(
        self,
        output: MultiModalStepOutput,
        replay: ReplayStep,
        sampler_mean: LatentState,
    ) -> torch.Tensor:
        """Return per-sample ``KL(q || p_theta)`` in the scheduler log-prob convention.

        Equals ``-E_q[log p_theta] + const`` exactly, so adding it to the policy
        log-likelihood centers the score. Raw per-element values are reduced once
        globally, with the same element weighting as the joint ``log_prob``.

        Args:
            output: Current-policy step output carrying ``next_state_mean``.
            replay: Stored rollout transition being replayed.
            sampler_mean: Stored rollout transition mean ``mu_q``.

        Returns:
            KL tensor of shape ``(B,)``; differentiable through ``mu_theta``.
        """
        expected_names = self.adapter.trajectory_component_order
        if sampler_mean.component_names != expected_names:
            raise ValueError(
                f"expected stored rollout next_latents_mean in component order "
                f"{expected_names}, received {sampler_mean.component_names}"
            )
        policy_mean = self._require_output_state(output, "next_state_mean", "policy")
        std_dev_t = self._require_component_mapping(output.std_dev_t, "std_dev_t", "policy output")
        dt = self._require_component_mapping(output.dt, "dt", "policy output")
        kl_elements = {}
        for name in expected_names:
            rollout_mean = sampler_mean.components[name]
            current_mean = policy_mean.components[name]
            if rollout_mean.shape != current_mean.shape:
                raise ValueError(
                    f"expected stored rollout next_latents_mean for component {name!r} to "
                    f"match the policy next_state_mean shape {tuple(current_mean.shape)}, "
                    f"received {tuple(rollout_mean.shape)}"
                )
            coef = self._log_prob_quadratic_coef(name, std_dev_t[name], dt[name])
            kl_elements[name] = coef * (rollout_mean.float() - current_mean.float()) ** 2
        return self.adapter.reduce_latent_values(kl_elements, state=replay.state)

    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts and store the rollout transition mean ``mu_q``."""
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.get_train_step_indices(),
            num_inference_steps=self.training_args.num_inference_steps,
        )
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=True,
            trajectory_indices=trajectory_indices,
            extra_call_back_kwargs=["next_latents_mean"],
        )

    # =========================== Optimization Loop ============================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): score-centered loss and optional KL-vs-ref."""
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        score_centering = self.training_args.score_centering
        kl_guidance_scale = self.training_args.kl_guidance_scale
        adv_clip_range = self.training_args.adv_clip_range
        train_step_indices = self.adapter.get_train_step_indices()
        # The sampler KL needs the policy transition mean and its variance statistics.
        requested_fields = {"log_prob", "next_latents_mean", "std_dev_t", "dt"}
        if self.enable_kl_loss:
            _, ref_return_field = self._kl_space_fields(self.training_args.kl_type, "kl_type")
            requested_fields.add(ref_return_field)
        else:
            ref_return_field = None
        return_fields = self._canonical_return_fields(requested_fields)
        for inner_epoch in range(self.training_args.num_inner_epochs):
            shuffled_samples = self._order_samples_for_optimize(samples, inner_epoch)

            self.adapter.train()
            loss_info = defaultdict(list)

            for batch in tqdm(
                self._iter_prefetched_batches(shuffled_samples, per_device_batch_size),
                total=num_batches,
                desc=f"Epoch {self.epoch} Training",
                position=0,
                disable=not self.show_progress_bar,
            ):
                for timestep_index in tqdm(
                    train_step_indices,
                    desc=f"Epoch {self.epoch} Timestep",
                    position=1,
                    leave=False,
                    disable=not self.show_progress_bar,
                ):
                    step_index = int(timestep_index)
                    with self.accumulate_gradients():
                        # 1. Prepare inputs
                        replay = self.adapter.get_replay_step(batch, step_index)
                        old_log_prob = self._require_replay_log_prob(replay, step_index)
                        sampler_mean = self.adapter.get_replay_callback(
                            batch, step_index, "next_latents_mean"
                        )
                        # 2. Forward pass
                        with self.autocast():
                            output = self._replay_forward(batch, replay, return_fields)

                        # 3. Compute loss: -A * (log p_theta(x') + KL(q || p_theta))
                        adv = batch["advantage"]
                        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                        new_log_prob = self._require_policy_log_prob(
                            output, step_index, self._replay_batch_size(replay)
                        )
                        if score_centering:
                            sampler_kl = self._sampler_kl(output, replay, sampler_mean)
                            policy_objective = new_log_prob + sampler_kl
                        else:
                            with torch.no_grad():
                                sampler_kl = self._sampler_kl(output, replay, sampler_mean)
                            policy_objective = new_log_prob
                        policy_loss = torch.mean(-adv * policy_objective)

                        loss = policy_loss

                        # 4. Optional KL-vs-reference penalty (run at kl_guidance_scale CFG).
                        if self.enable_kl_loss:
                            ref_overrides = (
                                {}
                                if kl_guidance_scale is None
                                else {"guidance_scale": kl_guidance_scale}
                            )
                            with self.autocast():
                                ref_output = self._reference_forward(
                                    batch, replay, (ref_return_field,), **ref_overrides
                                )
                                # kl_div must be computed outside `torch.no_grad()` for correct gradients.
                                kl_div = self._reference_kl_divergence(output, ref_output, replay)
                                kl_loss = self.training_args.kl_beta * kl_div
                                loss += kl_loss
                                loss_info["kl_div"].append(kl_div.detach())
                                loss_info["kl_loss"].append(kl_loss.detach())

                        # 5. Log per-timestep info
                        centered_log_prob = (new_log_prob + sampler_kl).detach()
                        loss_info["sampler_kl"].append(sampler_kl.detach())
                        loss_info["centered_log_ratio"].append(centered_log_prob - old_log_prob)
                        loss_info["plain_log_ratio"].append(new_log_prob.detach() - old_log_prob)
                        loss_info["policy_loss"].append(policy_loss.detach())
                        loss_info["loss"].append(loss.detach())

                        # 6. Backward and optimizer step
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            loss_info = self._apply_optimizer_step(loss_info)
