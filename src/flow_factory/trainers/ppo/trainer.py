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

# src/flow_factory/trainers/ppo/trainer.py
"""Critic-based PPO trainer for flow-matching models.

Actor-critic PPO: reuses the GRPO SDE rollout / per-step log-prob machinery, but
replaces the single broadcast advantage with a value critic that provides a
per-step baseline and GAE per-step advantages, plus a clipped value loss. Policy
and critic share one backward and step their own optimizers.
"""
import os
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional

import torch
import tqdm as tqdm_

from ...hparams import PPOTrainingArguments
from ...samples import BaseSample
from ...utils.base import filter_kwargs
from ...utils.dist import reduce_loss_info
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import compute_trajectory_indices
from ..abc import BaseTrainer
from .critic import ValueCritic
from .gae import compute_gae, whiten

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)
logger = setup_logger(__name__)


class PPOTrainer(BaseTrainer):
    """Critic-based PPO (value net + GAE) trainer.

    References:
        [1] Proximal Policy Optimization Algorithms - https://arxiv.org/abs/1707.06347
        [2] High-Dimensional Continuous Control Using GAE - https://arxiv.org/abs/1506.02438
        [3] Flow-GRPO (shared SDE rollout machinery) - https://arxiv.org/abs/2505.05470
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: PPOTrainingArguments
        self.num_train_timesteps = self.adapter.scheduler.num_sde_steps

    # =========================== Initialization ===========================
    def _initialization(self):
        """Prepare policy (base), then eagerly build + prepare the value critic.

        The critic input channel count ``C`` is resolved up front from
        ``adapter.compute_actual_latent_shape`` so the critic is fully materialized
        before ``accelerator.prepare`` (clean DDP wrapping, no lazy first layer).
        """
        super()._initialization()

        ta = self.training_args
        num_frames = getattr(ta, "num_frames", None)
        latent_shape = self.adapter.compute_actual_latent_shape(
            height=ta.height, width=ta.width, num_frames=num_frames
        )
        probe = torch.empty((1, *latent_shape))
        axes = self.adapter.resolve_latent_axes(probe)
        self.latent_channel_dim = axes.channel
        in_channels = probe.shape[axes.channel]

        self.critic = ValueCritic(
            in_channels=in_channels,
            hidden_dim=ta.critic_hidden_dim,
            num_layers=ta.critic_num_layers,
            time_embed_dim=ta.critic_time_embed_dim,
            num_heads=ta.critic_attn_heads,
            num_query_tokens=ta.critic_num_query_tokens,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=ta.critic_learning_rate,
            betas=ta.adam_betas,
            weight_decay=ta.adam_weight_decay,
            eps=ta.adam_epsilon,
        )
        self.critic, self.critic_optimizer = self.accelerator.prepare(
            self.critic, self.critic_optimizer
        )
        if self.accelerator.is_local_main_process:
            num_params = sum(p.numel() for p in self.critic.parameters())
            logger.info(
                f"Initialized PPO value critic: in_channels={in_channels}, "
                f"hidden_dim={ta.critic_hidden_dim}, params={num_params/1e6:.2f}M"
            )

    @property
    def enable_kl_loss(self) -> bool:
        """Whether the optional KL-to-reference penalty is enabled."""
        return self.training_args.kl_beta > 0.0

    # =========================== Main Loop ===========================
    def start(self):
        """Main training loop (Stages 2-6 per epoch)."""
        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)

            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    "checkpoints",
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            if self.eval_args.eval_freq > 0 and self.epoch % self.eval_args.eval_freq == 0:
                self.evaluate()

            samples = self.sample()
            self.prepare_feedback(samples)
            self.optimize(samples)

            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    # =========================== Sampling (Stages 2-3) ===========================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts (stores full trajectory + per-step log-probs)."""
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.scheduler.train_timesteps,
            num_inference_steps=self.training_args.num_inference_steps,
        )
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=True,
            trajectory_indices=trajectory_indices,
        )

    # =========================== Feedback (Stages 4-5) ===========================
    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Finalize rewards, run the old critic + GAE, and store per-step targets.

        Stores ``step_advantages``/``step_returns``/``step_old_values`` (each of
        shape ``(S,)`` aligned with the sorted SDE steps) on every sample's
        ``extra_kwargs``; these are stacked to ``(B, S)`` and indexed per step in
        :meth:`optimize`.
        """
        rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
        terminal_rewards = self._compute_terminal_rewards(samples, rewards)  # (N,)
        old_values = self._compute_old_values(samples)  # (N, S)
        advantages, returns = compute_gae(
            old_values,
            terminal_rewards,
            gamma=self.training_args.gae_gamma,
            lam=self.training_args.gae_lambda,
        )
        if self.training_args.normalize_advantage:
            advantages = whiten(advantages.to(self.accelerator.device), self.accelerator).cpu()

        for i, sample in enumerate(samples):
            sample.extra_kwargs["step_advantages"] = advantages[i].clone()
            sample.extra_kwargs["step_returns"] = returns[i].clone()
            sample.extra_kwargs["step_old_values"] = old_values[i].clone()

        self._log_feedback_metrics(terminal_rewards, old_values, advantages, returns)

    def _sorted_train_timesteps(self) -> List[int]:
        """Ordered (denoising-direction) SDE step indices for the current epoch."""
        return sorted(int(t) for t in self.adapter.scheduler.train_timesteps.tolist())

    def _compute_terminal_rewards(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Weighted-sum aggregation of per-reward scores into a per-sample scalar.

        Uses the configured reward weights (source-aware where a sample carries a
        ``source``); non-applicable ``NaN`` scores contribute zero. The critic is
        the variance baseline, so (unlike GRPO) no group normalization is applied.
        """
        num_samples = len(samples)
        terminal = torch.zeros(num_samples, dtype=torch.float32)
        reward_weights = self.advantage_processor.reward_weights
        for name, scores in rewards.items():
            scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
            scores = torch.nan_to_num(scores, nan=0.0)
            per_dataset = reward_weights.get(name, {})
            default_weight = next(iter(per_dataset.values())) if per_dataset else 1.0
            weights = torch.tensor(
                [
                    (
                        per_dataset.get(s.source, default_weight)
                        if s.source is not None
                        else default_weight
                    )
                    for s in samples
                ],
                dtype=torch.float32,
            )
            terminal = terminal + weights * scores
        return terminal

    def _compute_old_values(self, samples: List[BaseSample]) -> torch.Tensor:
        """Old-critic value estimates ``(N, S)`` over the sorted SDE steps (no grad)."""
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        sorted_ts = self._sorted_train_timesteps()
        offload = self.training_args.offload_samples_to_cpu

        self.critic.eval()
        chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(samples), per_device_batch_size):
                batch_samples = [
                    s.to(device) for s in samples[start : start + per_device_batch_size]
                ]
                batch = BaseSample.stack(batch_samples)
                latent_index_map = batch["latent_index_map"]
                all_latents = batch["all_latents"]
                timesteps = batch["timesteps"]
                step_values = []
                for timestep_index in sorted_ts:
                    latents = all_latents[:, latent_index_map[timestep_index]]
                    t = timesteps[:, timestep_index]
                    with self.autocast():
                        value = self.critic(latents, t, self.latent_channel_dim)
                    step_values.append(value.float())
                chunks.append(torch.stack(step_values, dim=1).cpu())  # (b, S)
                if offload:
                    for s in batch_samples:
                        s.to("cpu")
        return torch.cat(chunks, dim=0)  # (N, S)

    def _log_feedback_metrics(
        self,
        terminal_rewards: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> None:
        """Log global reward + local value/return/advantage diagnostics."""
        gathered = self.accelerator.gather(terminal_rewards.to(self.accelerator.device))
        self.log_data(
            {
                "train/reward_mean": gathered.mean().item(),
                "train/reward_std": gathered.std().item(),
                "train/value_mean": old_values.mean().item(),
                "train/return_mean": returns.mean().item(),
                "train/advantage_mean": advantages.mean().item(),
                "train/advantage_std": advantages.std().item(),
            },
            step=self.step,
        )

    # =========================== Optimization (Stage 6) ===========================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Per-step clipped value loss + (post-warmup) PPO-clipped policy loss; dual optimizers.

        During the first ``critic_warmup_steps`` optimizer steps only the critic is
        trained (the policy is frozen and its forward is skipped entirely).
        """
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        clip_range = self.training_args.clip_range
        adv_clip_range = self.training_args.adv_clip_range
        value_clip_range = self.training_args.value_clip_range
        vf_coef = self.training_args.vf_coef
        sorted_ts = self._sorted_train_timesteps()

        for inner_epoch in range(self.training_args.num_inner_epochs):
            shuffled_samples = self._order_samples_for_optimize(samples, inner_epoch)

            self.adapter.train()
            self.critic.train()
            loss_info = defaultdict(list)

            for batch_idx in tqdm(
                range(num_batches),
                total=num_batches,
                desc=f"Epoch {self.epoch} Training",
                position=0,
                disable=not self.show_progress_bar,
            ):
                start = batch_idx * per_device_batch_size
                batch_samples = [
                    sample.to(device)
                    for sample in shuffled_samples[start : start + per_device_batch_size]
                ]
                batch = BaseSample.stack(batch_samples)
                # GAE targets live in extra_kwargs, which BaseSample.to() does NOT move;
                # bring the (B, S) per-step tensors onto the device once per micro-batch.
                for key in ("step_advantages", "step_returns", "step_old_values"):
                    batch[key] = batch[key].to(device)
                latents_index_map = batch["latent_index_map"]  # (T+1,)
                log_probs_index_map = batch["log_prob_index_map"]  # (T+1,)
                num_timesteps = batch["timesteps"].shape[1]

                for step_idx, timestep_index in enumerate(
                    tqdm(
                        sorted_ts,
                        desc=f"Epoch {self.epoch} Timestep",
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    )
                ):
                    in_warmup = self.step < self.training_args.critic_warmup_steps
                    with self.accumulate_gradients(self.critic):
                        # 1. Critic value loss (trained every step, including warmup)
                        t = batch["timesteps"][:, timestep_index]
                        latents = batch["all_latents"][:, latents_index_map[timestep_index]]
                        returns_t = batch["step_returns"][:, step_idx]
                        old_values = batch["step_old_values"][:, step_idx]

                        with self.autocast():
                            value = self.critic(latents, t, self.latent_channel_dim)
                        value = value.float()
                        if value_clip_range is not None:
                            value_clipped = old_values + torch.clamp(
                                value - old_values, -value_clip_range, value_clip_range
                            )
                            value_loss = 0.5 * torch.mean(
                                torch.maximum(
                                    (value - returns_t) ** 2, (value_clipped - returns_t) ** 2
                                )
                            )
                        else:
                            value_loss = 0.5 * torch.mean((value - returns_t) ** 2)
                        loss = vf_coef * value_loss

                        # 2. Policy update (PPO-clip [+ KL]) -- skipped during critic warmup,
                        #    so the policy stays frozen and no policy forward is wasted.
                        policy_loss = None
                        ratio = None
                        if not in_warmup:
                            old_log_prob = batch["log_probs"][
                                :, log_probs_index_map[timestep_index]
                            ]
                            t_next = (
                                batch["timesteps"][:, timestep_index + 1]
                                if timestep_index + 1 < num_timesteps
                                else torch.tensor(0, device=device)
                            )
                            next_latents = batch["all_latents"][
                                :, latents_index_map[timestep_index + 1]
                            ]
                            adv = torch.clamp(
                                batch["step_advantages"][:, step_idx],
                                adv_clip_range[0],
                                adv_clip_range[1],
                            )
                            forward_inputs = {
                                **self.training_args,
                                "t": t,
                                "t_next": t_next,
                                "latents": latents,
                                "next_latents": next_latents,
                                "compute_log_prob": True,
                                "noise_level": self.adapter.scheduler.noise_level,
                                **batch,
                            }
                            forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)
                            if self.enable_kl_loss:
                                if self.training_args.kl_type == "v-based":
                                    forward_inputs["return_kwargs"] = [
                                        "log_prob",
                                        "noise_pred",
                                        "dt",
                                    ]
                                else:
                                    forward_inputs["return_kwargs"] = [
                                        "log_prob",
                                        "next_latents_mean",
                                        "dt",
                                    ]
                            else:
                                forward_inputs["return_kwargs"] = ["log_prob", "dt"]

                            with self.autocast():
                                output = self.adapter.forward(**forward_inputs)

                            ratio = torch.exp(output.log_prob - old_log_prob)
                            clipped_ratio = torch.clamp(
                                ratio, 1.0 + clip_range[0], 1.0 + clip_range[1]
                            )
                            policy_loss = torch.mean(
                                torch.maximum(-adv * ratio, -adv * clipped_ratio)
                            )
                            loss = loss + policy_loss

                            if self.enable_kl_loss:
                                kl_div = self._compute_kl(forward_inputs, output)
                                kl_loss = self.training_args.kl_beta * kl_div
                                loss = loss + kl_loss
                                loss_info["kl_div"].append(kl_div.detach())
                                loss_info["kl_loss"].append(kl_loss.detach())

                        # 3. Backward + dual optimizer step
                        self.accelerator.backward(loss)
                        policy_grad_norm = None
                        critic_grad_norm = None
                        if self.accelerator.sync_gradients:
                            critic_grad_norm = self.accelerator.clip_grad_norm_(
                                self.critic.parameters(), self.training_args.max_grad_norm
                            )
                            if not in_warmup:
                                policy_grad_norm = self.accelerator.clip_grad_norm_(
                                    self.adapter.get_trainable_parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                            self.critic_optimizer.step()
                            self.optimizer.zero_grad()
                            self.critic_optimizer.zero_grad()

                        # 4. Metrics (policy metrics only when the policy is updated)
                        loss_info["value_loss"].append(value_loss.detach())
                        loss_info["explained_variance"].append(
                            self._explained_variance(value.detach(), returns_t).detach()
                        )
                        loss_info["loss"].append(loss.detach())
                        if policy_loss is not None:
                            clip_frac_high = torch.mean((ratio > 1.0 + clip_range[1]).float())
                            clip_frac_low = torch.mean((ratio < 1.0 + clip_range[0]).float())
                            loss_info["ratio"].append(ratio.mean().detach())
                            loss_info["policy_loss"].append(policy_loss.detach())
                            loss_info["clip_frac_high"].append(clip_frac_high.detach())
                            loss_info["clip_frac_low"].append(clip_frac_low.detach())
                            loss_info["clip_frac_total"].append(
                                (clip_frac_high + clip_frac_low).detach()
                            )

                        if self.accelerator.sync_gradients:
                            reduced = reduce_loss_info(self.accelerator, loss_info)
                            reduced["critic_grad_norm"] = critic_grad_norm
                            if policy_grad_norm is not None:
                                reduced["grad_norm"] = policy_grad_norm
                            self.log_data(
                                {f"train/{k}": v for k, v in reduced.items()},
                                step=self.step,
                            )
                            self.step += 1
                            loss_info = defaultdict(list)

    @staticmethod
    def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Fraction of return variance explained by the value predictions."""
        var_returns = returns.var()
        if var_returns <= 0:
            return torch.zeros((), device=values.device)
        return 1.0 - (returns - values).var() / var_returns

    def _compute_kl(self, forward_inputs: dict, output) -> torch.Tensor:
        """KL of the current policy step vs the reference policy (DPOK-style)."""
        with self.autocast():
            with torch.no_grad(), self.adapter.use_ref_parameters():
                ref_inputs = forward_inputs.copy()
                ref_inputs["compute_log_prob"] = False
                if self.training_args.kl_type == "v-based":
                    ref_inputs["return_kwargs"] = ["noise_pred"]
                else:
                    ref_inputs["return_kwargs"] = ["next_latents_mean"]
                ref_output = self.adapter.forward(**ref_inputs)

            if self.training_args.kl_type == "v-based":
                kl_div = torch.mean(
                    (output.noise_pred - ref_output.noise_pred) ** 2,
                    dim=tuple(range(1, output.noise_pred.ndim)),
                    keepdim=True,
                )
            else:
                kl_div = torch.mean(
                    (output.next_latents_mean - ref_output.next_latents_mean) ** 2,
                    dim=tuple(range(1, output.next_latents_mean.ndim)),
                    keepdim=True,
                )
        return torch.mean(kl_div)

    # =========================== Checkpoint ===========================
    def save_checkpoint(self, save_directory: str, epoch: Optional[int] = None):
        """Save the policy (via adapter) and the value critic side-by-side."""
        super().save_checkpoint(save_directory, epoch=epoch)
        if epoch is not None:
            save_directory = os.path.join(save_directory, f"checkpoint-{epoch}")
        if self.accelerator.is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            critic = self.accelerator.unwrap_model(self.critic)
            torch.save(critic.state_dict(), os.path.join(save_directory, "critic.pt"))
            logger.info(f"Saved PPO value critic to {save_directory}/critic.pt")
        self.accelerator.wait_for_everyone()

    def load_checkpoint(self, path: str, resume_type: Optional[str] = None):
        """Load the policy (via adapter) and the value critic if present."""
        super().load_checkpoint(path, resume_type=resume_type)
        critic_path = os.path.join(path, "critic.pt")
        if os.path.exists(critic_path):
            state_dict = torch.load(critic_path, map_location=self.accelerator.device)
            self.accelerator.unwrap_model(self.critic).load_state_dict(state_dict)
            logger.info(f"Loaded PPO value critic from {critic_path}")
        else:
            logger.warning(
                f"No critic checkpoint at {critic_path}; keeping freshly initialized critic."
            )
        self.accelerator.wait_for_everyone()
