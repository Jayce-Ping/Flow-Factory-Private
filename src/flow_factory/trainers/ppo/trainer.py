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
and critic use *separate* backward passes and their own optimizers, which lets
their update frequencies be decoupled (``critic_update_interval`` /
``policy_update_interval``); see :meth:`PPOTrainer.optimize`.
"""
import os
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, List, Optional

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

    # Metric keys logged on the critic (``critic_step``) axis; everything else in an
    # optimize round goes on the policy (``step``) axis. See ``_log_optimize_round``.
    _CRITIC_METRIC_KEYS = ("value_loss", "explained_variance", "critic_grad_norm", "critic_updated")

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

        # Manual gradient-accumulation counter (0..gas-1). We drive accumulation
        # ourselves instead of ``accelerator.accumulate`` so the critic and policy
        # can step at different intervals; this counter persists across batches and
        # epochs so the ``gas`` window stays continuous.
        self.micro_in_round = 0
        # Periodic critic re-warmup state: last triggered bucket (self.step // interval)
        # and a monotonic call counter used to seed each burst's resampling distinctly.
        self._last_rewarmup_bucket = 0
        self._rewarmup_calls = 0
        # Logging axes. PPO emits three semantic timelines that are not jointly monotonic
        # (and the critic/feedback advance during warmup while ``self.step`` is frozen):
        #   - ``self.step``        -> policy optimizer rounds      (train/policy/* panels)
        #   - ``self.critic_step`` -> critic optimizer steps       (train/critic/* panels)
        #   - ``self.rollout_step``-> rollouts / feedback events   (train/rollout/* panels)
        # Each is logged as a value and bound to a panel via ``define_step_metric``. The
        # backend itself receives ``_global_log_step`` (monotonic) so no point is dropped.
        self.critic_step = 0
        self.rollout_step = 0
        self._global_log_step = 0

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

        self._validate_per_sample_sde()
        if self.per_sample_sde and self.accelerator.is_local_main_process:
            logger.info(
                "PPO single-step SDE mode (sde_step_selection='per_sample'): one SDE "
                "step per sample, exact bandit advantage A = R - V(z_k, t_k)."
            )

    @property
    def enable_kl_loss(self) -> bool:
        """Whether the optional KL-to-reference penalty is enabled."""
        return self.training_args.kl_beta > 0.0

    @property
    def per_sample_sde(self) -> bool:
        """True when single-step (per-sample) SDE selection is active.

        In this mode each rollout sample injects noise at exactly one denoising step
        (the rest are ODE), so the denoising MDP collapses to a per-step contextual
        bandit: the advantage is the exact single-step ``A = R - V(z_k, t_k)`` and the
        optimize loop trains only that one transition per sample (see
        ``docs/ppo/single_step_sde_ppo``).
        """
        return getattr(self.adapter.scheduler, "sde_step_selection", "global") == "per_sample"

    def _validate_per_sample_sde(self) -> None:
        """Fail fast if per-sample SDE is requested but unsupported by the scheduler."""
        if not self.per_sample_sde:
            return
        scheduler = self.adapter.scheduler
        if scheduler.dynamics_type == "ODE":
            raise ValueError(
                "sde_step_selection='per_sample' needs a stochastic dynamics_type "
                f"(Flow-SDE/Dance-SDE/CPS), got {scheduler.dynamics_type!r}."
            )
        if not hasattr(scheduler, "_resolve_per_sample_noise_level"):
            raise ValueError(
                "sde_step_selection='per_sample' requires FlowMatchEulerDiscreteSDEScheduler; "
                f"got {type(scheduler).__name__}."
            )

    # =========================== Logging axes ===========================
    def log_data(self, data: Dict[str, Any], step: int) -> None:
        """Route every PPO log through a monotonic engine step.

        Single-global-step backends (wandb/swanlab) drop non-monotonic logs, and PPO's
        three semantic axes (policy ``step`` / ``critic_step`` / ``rollout_step``) are not
        jointly monotonic -- warmup also logs while ``self.step`` is frozen. We therefore
        pass an ever-increasing ``_global_log_step`` to the backend and carry the semantic
        axis as a logged value (bound to a panel by :meth:`_configure_log_axes`). The
        ``step`` argument from callers is intentionally ignored here.
        """
        self._global_log_step += 1
        super().log_data(data, step=self._global_log_step)

    def _configure_log_axes(self) -> None:
        """Bind each metric namespace to its semantic x-axis (wandb; no-op elsewhere)."""
        if self.logger is None:
            return
        self.logger.define_step_metric("train/policy/*", "train/step")
        self.logger.define_step_metric("train/critic/*", "train/critic_step")
        self.logger.define_step_metric("train/rollout/*", "train/rollout_step")
        self.logger.define_step_metric("eval/*", "train/step")

    def _log_optimize_round(
        self,
        reduced: Dict[str, Any],
        *,
        do_critic: bool,
        critic_grad_norm: Optional[Any],
        policy_grad_norm: Optional[Any],
        critic_step_now: bool,
        policy_step_now: bool,
    ) -> None:
        """Split one optimize round's reduced metrics into policy + critic panels.

        Policy metrics go to ``train/policy/*`` (``step`` axis); critic metrics (mode A
        only) go to ``train/critic/*`` (``critic_step`` axis). Both are emitted in one
        payload carrying both axis values, so wandb's ``define_metric`` still routes them
        to separate panels while the console keeps a single line per round. Advances
        ``self.critic_step`` when critic metrics are emitted.
        """
        reduced["policy_updated"] = float(policy_step_now)
        if policy_grad_norm is not None:
            reduced["grad_norm"] = policy_grad_norm

        payload: Dict[str, Any] = {}
        has_critic = False
        if do_critic:
            reduced["critic_updated"] = float(critic_step_now)
            if critic_grad_norm is not None:
                reduced["critic_grad_norm"] = critic_grad_norm
            for key in self._CRITIC_METRIC_KEYS:
                if key in reduced:
                    payload[f"train/critic/{key}"] = reduced.pop(key)
                    has_critic = True

        for key, value in reduced.items():
            payload[f"train/policy/{key}"] = value
        payload["train/step"] = self.step
        if has_critic:
            payload["train/critic_step"] = self.critic_step

        self.log_data(payload, step=self.step)
        if has_critic:
            self.critic_step += 1

    def _log_critic_only(self, reduced: Dict[str, Any]) -> None:
        """Log a critic-only fit step (warmup bursts) on the ``critic_step`` axis."""
        critic_metrics = {f"train/critic/{key}": v for key, v in reduced.items()}
        critic_metrics["train/critic_step"] = self.critic_step
        self.log_data(critic_metrics, step=self.critic_step)
        self.critic_step += 1

    # =========================== Main Loop ===========================
    def start(self):
        """Main training loop (Stages 2-6 per epoch)."""
        # Bind metric namespaces to their semantic x-axes before any logging.
        self._configure_log_axes()
        # Initial critic-only warmup (bootstrap) before any policy update.
        self._warmup_critic(self.training_args.critic_warmup_steps, "bootstrap")
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

            # Periodic critic re-warmup (resample + critic-only fit to a step target) to catch
            # the critic up to the moving policy; does not advance self.step / self.epoch.
            interval = self.training_args.critic_warmup_interval
            if interval > 0 and self.step // interval > self._last_rewarmup_bucket:
                self._last_rewarmup_bucket = self.step // interval
                self._warmup_critic(self.training_args.critic_rewarmup_steps, "rewarmup")

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

    def sample_batch(self, batch: Dict[str, Any], reward_buffer=None, **extra_inference_kwargs):
        """Per-batch sampling; in per-sample SDE mode, tag each sample with its step ``k``.

        The scheduler chooses a single SDE step per sample during the rollout and records
        it on ``scheduler._per_sample_k`` (batch order). We read it back here (one rollout
        per ``sample_batch`` call) and stamp ``sde_step_k`` onto each sample so feedback /
        optimize can gather the per-sample transition. No-op in the default global mode.
        """
        samples = super().sample_batch(batch, reward_buffer=reward_buffer, **extra_inference_kwargs)
        if not self.per_sample_sde or self.adapter.scheduler.is_eval:
            return samples
        per_sample_k = self.adapter.scheduler._per_sample_k
        if per_sample_k is None:
            raise RuntimeError(
                "per-sample SDE selection is on but the scheduler did not record "
                "`_per_sample_k` during rollout (expected one SDE step per sample)."
            )
        if len(per_sample_k) != len(samples):
            raise RuntimeError(
                f"per-sample SDE mismatch: scheduler recorded {len(per_sample_k)} steps "
                f"but {len(samples)} samples were produced for this batch."
            )
        for j, sample in enumerate(samples):
            sample.extra_kwargs["sde_step_k"] = torch.tensor(int(per_sample_k[j].item()), dtype=torch.long)
        return samples

    # =========================== Feedback (Stages 4-5) ===========================
    def prepare_feedback(self, samples: List[BaseSample], metrics_prefix: str = "train") -> None:
        """Finalize rewards, run the old critic + GAE, and store per-step targets.

        Stores ``step_advantages``/``step_returns``/``step_old_values`` (each of
        shape ``(S,)`` aligned with the sorted SDE steps) on every sample's
        ``extra_kwargs``; these are stacked to ``(B, S)`` and indexed per step in
        :meth:`optimize`.

        Args:
            samples: Rollout samples to score and annotate with GAE targets.
            metrics_prefix: Marks the rollout phase for feedback diagnostics
                (``"train_rewarmup"`` for re-warmup bursts, ``"train"`` otherwise).
                Metrics always log to the ``train/rollout/*`` panel on the
                ``rollout_step`` axis; the phase is recorded as ``train/rollout/phase``.
        """
        if self.per_sample_sde:
            self._prepare_feedback_bandit(samples, metrics_prefix=metrics_prefix)
            return
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

        self._log_feedback_metrics(
            terminal_rewards, old_values, advantages, returns, metrics_prefix=metrics_prefix
        )

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
        metrics_prefix: str = "train",
    ) -> None:
        """Log reward + value/return/advantage diagnostics on the ``rollout_step`` axis.

        Both the normal epoch and warmup resample bursts log here, so the rollout curves
        stay continuous across warmup and training. ``metrics_prefix`` is recorded only as
        a ``phase`` marker (1.0 for re-warmup bursts, 0.0 otherwise).
        """
        gathered = self.accelerator.gather(terminal_rewards.to(self.accelerator.device))
        self.log_data(
            {
                "train/rollout/reward_mean": gathered.mean().item(),
                "train/rollout/reward_std": gathered.std().item(),
                "train/rollout/value_mean": old_values.mean().item(),
                "train/rollout/return_mean": returns.mean().item(),
                "train/rollout/advantage_mean": advantages.mean().item(),
                "train/rollout/advantage_std": advantages.std().item(),
                "train/rollout/phase": 1.0 if "rewarmup" in metrics_prefix else 0.0,
                "train/rollout_step": self.rollout_step,
            },
            step=self.step,
        )
        self.rollout_step += 1

    # ================= Per-sample single-SDE (bandit) path =================
    def _bandit_gather(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
        """Gather each sample's single SDE transition by its own chosen step ``k``.

        In per-sample mode every trajectory position is stored, so we index per sample by
        ``sde_step_k`` through the shared latent / log-prob index maps. Returns the
        per-sample state ``z_k`` (and its timestep ``t_k``), the action ``z_{k+1}`` (and
        ``t_{k+1}``), and the rollout log-prob ``log pi_old(z_{k+1}|z_k)`` -- all ``(B, ...)``.
        """
        k = batch["sde_step_k"].to(device).long()  # (B,)
        batch_size = k.shape[0]
        ar = torch.arange(batch_size, device=device)
        latent_index_map = batch["latent_index_map"].to(device)  # (T+1,)
        log_prob_index_map = batch["log_prob_index_map"].to(device)  # (T+1,)
        all_latents = batch["all_latents"]  # (B, P, ...)
        timesteps = batch["timesteps"]  # (B, T)
        log_probs = batch["log_probs"]  # (B, L)
        num_timesteps = timesteps.shape[1]
        k_next = torch.clamp(k + 1, max=num_timesteps - 1)

        z_k = all_latents[ar, latent_index_map[k]]
        z_k_next = all_latents[ar, latent_index_map[k_next]]
        t_k = timesteps[ar, k]
        has_next = (k + 1) < num_timesteps
        t_k_next = torch.where(has_next, timesteps[ar, k_next], torch.zeros_like(t_k))
        old_log_prob = log_probs[ar, log_prob_index_map[k]]
        return {
            "z_k": z_k,
            "z_k_next": z_k_next,
            "t_k": t_k,
            "t_k_next": t_k_next,
            "old_log_prob": old_log_prob,
        }

    def _compute_old_values_bandit(self, samples: List[BaseSample]) -> torch.Tensor:
        """Old-critic value ``V(z_k, t_k)`` per sample, shape ``(N,)`` (no grad)."""
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        offload = self.training_args.offload_samples_to_cpu

        self.critic.eval()
        chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(samples), per_device_batch_size):
                batch_samples = [
                    s.to(device) for s in samples[start : start + per_device_batch_size]
                ]
                batch = BaseSample.stack(batch_samples)
                g = self._bandit_gather(batch, device)
                with self.autocast():
                    value = self.critic(g["z_k"], g["t_k"], self.latent_channel_dim)
                chunks.append(value.float().cpu())  # (b,)
                if offload:
                    for s in batch_samples:
                        s.to("cpu")
        return torch.cat(chunks, dim=0)  # (N,)

    def _prepare_feedback_bandit(self, samples: List[BaseSample], metrics_prefix: str = "train") -> None:
        """Bandit feedback: exact single-step advantage ``A = R - V(z_k, t_k)``.

        With a single stochastic step the return is exactly the terminal reward and the
        advantage is the exact single-action advantage (no GAE/bootstrap; see
        ``docs/ppo/single_step_sde_ppo``). Stores per-sample scalars
        ``bandit_advantage`` / ``bandit_return`` / ``bandit_old_value`` on ``extra_kwargs``
        (stacked to ``(B,)`` in :meth:`_optimize_bandit`).
        """
        rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
        terminal_rewards = self._compute_terminal_rewards(samples, rewards)  # (N,)
        old_values = self._compute_old_values_bandit(samples)  # (N,)
        advantages = terminal_rewards - old_values  # (N,) exact single-step advantage
        if self.training_args.normalize_advantage:
            advantages = whiten(advantages.to(self.accelerator.device), self.accelerator).cpu()
        returns = terminal_rewards.clone()  # (N,) exact Monte-Carlo return G = R

        for i, sample in enumerate(samples):
            sample.extra_kwargs["bandit_advantage"] = advantages[i].clone()
            sample.extra_kwargs["bandit_return"] = returns[i].clone()
            sample.extra_kwargs["bandit_old_value"] = old_values[i].clone()

        self._log_feedback_metrics(
            terminal_rewards, old_values, advantages, returns, metrics_prefix=metrics_prefix
        )

    def _optimize_bandit(self, samples: List[BaseSample]) -> None:
        """Single-step (bandit) PPO: one transition per sample, no per-timestep loop.

        Mirrors :meth:`optimize` (dual optimizers, manual ``gas`` accumulation, decoupled
        ``critic``/``policy`` update intervals, optional KL) but trains exactly the one SDE
        transition each sample took, weighted by ``A = R - V(z_k, t_k)``.
        """
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        clip_range = self.training_args.clip_range
        adv_clip_range = self.training_args.adv_clip_range
        vf_coef = self.training_args.vf_coef
        gas = self.accelerator.gradient_accumulation_steps
        ci = self.training_args.critic_update_interval
        pi = self.training_args.policy_update_interval
        do_critic = self.training_args.update_critic_in_optimize

        self.accelerator.sync_gradients = True

        # One micro-step per (inner_epoch, micro-batch): the per-timestep factor is 1.
        if self.training_args._manual_gradient_accumulation_steps:
            steps_per_optimize = self.training_args.num_inner_epochs * num_batches
            if steps_per_optimize % gas != 0:
                raise ValueError(
                    f"PPO manual gradient_accumulation_steps={gas} must divide the per-optimize "
                    f"micro-step count ({steps_per_optimize} = num_inner_epochs "
                    f"{self.training_args.num_inner_epochs} x num_batches {num_batches}); "
                    "otherwise an accumulation window spans two epochs' rollouts. Use a divisor "
                    "or gradient_accumulation_steps='auto'."
                )

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
                for key in ("bandit_advantage", "bandit_return", "bandit_old_value"):
                    batch[key] = batch[key].to(device)
                g = self._bandit_gather(batch, device)

                round_idx = self.step
                boundary = self.micro_in_round == gas - 1
                critic_step_now = do_critic and boundary and (round_idx + 1) % ci == 0
                policy_step_now = boundary and (round_idx + 1) % pi == 0

                t = g["t_k"]
                latents = g["z_k"]

                # 1. Critic value loss (mode A only).
                value_critic_term = None
                if do_critic:
                    returns_t = batch["bandit_return"]
                    old_values = batch["bandit_old_value"]
                    critic_ctx = (
                        nullcontext()
                        if critic_step_now
                        else self.accelerator.no_sync(self.critic)
                    )
                    with critic_ctx:
                        with self.autocast():
                            value = self.critic(latents, t, self.latent_channel_dim)
                        value = value.float()
                        value_loss = self._critic_value_loss(value, returns_t, old_values)
                        self.accelerator.backward(vf_coef * value_loss / ci)
                    value_critic_term = (vf_coef * value_loss).detach()
                    loss_info["value_loss"].append(value_loss.detach())
                    loss_info["explained_variance"].append(
                        self._explained_variance(value.detach(), returns_t).detach()
                    )

                # 2. Policy update (PPO-clip [+ KL]) on the single transition.
                old_log_prob = g["old_log_prob"]
                adv = torch.clamp(
                    batch["bandit_advantage"], adv_clip_range[0], adv_clip_range[1]
                )
                forward_inputs = {
                    **self.training_args,
                    "t": t,
                    "t_next": g["t_k_next"],
                    "latents": latents,
                    "next_latents": g["z_k_next"],
                    "compute_log_prob": True,
                    "noise_level": self.adapter.scheduler.noise_level,
                    **batch,
                }
                forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)
                if self.enable_kl_loss:
                    if self.training_args.kl_type == "v-based":
                        forward_inputs["return_kwargs"] = ["log_prob", "noise_pred", "dt"]
                    else:
                        forward_inputs["return_kwargs"] = ["log_prob", "next_latents_mean", "dt"]
                else:
                    forward_inputs["return_kwargs"] = ["log_prob", "dt"]

                kl_loss = None
                policy_ctx = (
                    nullcontext()
                    if policy_step_now
                    else self.accelerator.no_sync(self.model_bundle)
                )
                with policy_ctx:
                    with self.autocast():
                        output = self.adapter.forward(**forward_inputs)
                    ratio = torch.exp(output.log_prob - old_log_prob)
                    clipped_ratio = torch.clamp(ratio, 1.0 + clip_range[0], 1.0 + clip_range[1])
                    policy_loss = torch.mean(torch.maximum(-adv * ratio, -adv * clipped_ratio))
                    total = policy_loss
                    if self.enable_kl_loss:
                        kl_div = self._compute_kl(forward_inputs, output)
                        kl_loss = self.training_args.kl_beta * kl_div
                        total = total + kl_loss
                        loss_info["kl_div"].append(kl_div.detach())
                        loss_info["kl_loss"].append(kl_loss.detach())
                    self.accelerator.backward(total / pi)

                # 3. Independent optimizer steps at each network's window end.
                critic_grad_norm = None
                policy_grad_norm = None
                if critic_step_now:
                    critic_grad_norm = self.accelerator.clip_grad_norm_(
                        self.critic.parameters(), self.training_args.max_grad_norm
                    )
                    self.critic_optimizer.step()
                    self.critic_optimizer.zero_grad()
                if policy_step_now:
                    policy_grad_norm = self.accelerator.clip_grad_norm_(
                        self.adapter.get_trainable_parameters(),
                        self.training_args.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # 4. Metrics.
                combined = policy_loss.detach()
                if kl_loss is not None:
                    combined = combined + kl_loss.detach()
                if value_critic_term is not None:
                    combined = combined + value_critic_term
                clip_frac_high = torch.mean((ratio > 1.0 + clip_range[1]).float())
                clip_frac_low = torch.mean((ratio < 1.0 + clip_range[0]).float())
                loss_info["ratio"].append(ratio.mean().detach())
                loss_info["policy_loss"].append(policy_loss.detach())
                loss_info["clip_frac_high"].append(clip_frac_high.detach())
                loss_info["clip_frac_low"].append(clip_frac_low.detach())
                loss_info["clip_frac_total"].append((clip_frac_high + clip_frac_low).detach())
                loss_info["loss"].append(combined)

                # 5. Round bookkeeping at the gas-window boundary.
                if boundary:
                    reduced = reduce_loss_info(self.accelerator, loss_info)
                    self._log_optimize_round(
                        reduced,
                        do_critic=do_critic,
                        critic_grad_norm=critic_grad_norm,
                        policy_grad_norm=policy_grad_norm,
                        critic_step_now=critic_step_now,
                        policy_step_now=policy_step_now,
                    )
                    self.step += 1
                    self.micro_in_round = 0
                    loss_info = defaultdict(list)
                else:
                    self.micro_in_round += 1

    def _fit_critic_only_bandit(self, samples: List[BaseSample], num_passes: int) -> int:
        """Critic-only fit (per-sample single-SDE): regress ``V(z_k, t_k) -> R`` per sample.

        Bandit analogue of :meth:`_fit_critic_only`; never forwards/steps the policy and
        never advances ``self.step`` / ``self.epoch``. Returns the number of critic steps.
        """
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        vf_coef = self.training_args.vf_coef
        gas = self.accelerator.gradient_accumulation_steps

        self.critic.train()
        self.accelerator.sync_gradients = True
        self.critic_optimizer.zero_grad()
        critic_steps = 0
        micro = 0
        loss_info = defaultdict(list)
        for pass_idx in range(num_passes):
            shuffled_samples = self._order_samples_for_optimize(samples, pass_idx)
            for batch_idx in tqdm(
                range(num_batches),
                total=num_batches,
                desc=f"Rewarmup {self._rewarmup_calls}",
                leave=False,
                disable=not self.show_progress_bar,
            ):
                start = batch_idx * per_device_batch_size
                batch_samples = [
                    s.to(device) for s in shuffled_samples[start : start + per_device_batch_size]
                ]
                batch = BaseSample.stack(batch_samples)
                for key in ("bandit_return", "bandit_old_value"):
                    batch[key] = batch[key].to(device)
                g = self._bandit_gather(batch, device)
                boundary = micro == gas - 1
                returns_t = batch["bandit_return"]
                old_values = batch["bandit_old_value"]

                critic_ctx = (
                    nullcontext() if boundary else self.accelerator.no_sync(self.critic)
                )
                with critic_ctx:
                    with self.autocast():
                        value = self.critic(g["z_k"], g["t_k"], self.latent_channel_dim)
                    value = value.float()
                    value_loss = self._critic_value_loss(value, returns_t, old_values)
                    self.accelerator.backward(vf_coef * value_loss)

                loss_info["value_loss"].append(value_loss.detach())
                loss_info["explained_variance"].append(
                    self._explained_variance(value.detach(), returns_t).detach()
                )
                if boundary:
                    self.accelerator.clip_grad_norm_(
                        self.critic.parameters(), self.training_args.max_grad_norm
                    )
                    self.critic_optimizer.step()
                    self.critic_optimizer.zero_grad()
                    critic_steps += 1
                    reduced = reduce_loss_info(self.accelerator, loss_info)
                    self._log_critic_only(reduced)
                    loss_info = defaultdict(list)
                    micro = 0
                else:
                    micro += 1
        if micro != 0:
            self.critic_optimizer.zero_grad()
        return critic_steps

    # =========================== Optimization (Stage 6) ===========================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Per-step PPO-clipped policy loss + (mode A) clipped value loss; dual optimizers.

        Critic and policy use **separate** backward passes (each only traverses its
        own sub-graph), so DDP gradient sync, accumulation and stepping are
        independent. We do not use ``accelerator.accumulate``; instead a manual
        ``micro_in_round`` counter marks each ``gas``-micro-step round, and each
        network accumulates ``interval * gas`` micro-steps before one optimizer step
        (``critic_update_interval`` / ``policy_update_interval``, in rounds).
        ``accelerator.backward`` already divides by ``gas``; we additionally divide
        each loss by its ``interval`` so the gradient is the mean over the full window.

        The policy trains every micro-step here (the initial critic warmup is an
        up-front bootstrap in :meth:`_warmup_critic`, not a frozen-policy phase).
        The critic trains here only when ``update_critic_in_optimize`` is True
        (mode A); in mode B it is trained solely by the bootstrap + periodic
        :meth:`_warmup_critic` bursts.

        In per-sample single-SDE mode the per-timestep loop collapses to a single
        bandit transition per sample (:meth:`_optimize_bandit`).
        """
        if self.per_sample_sde:
            self._optimize_bandit(samples)
            return
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        clip_range = self.training_args.clip_range
        adv_clip_range = self.training_args.adv_clip_range
        vf_coef = self.training_args.vf_coef
        # gas = base accumulation window (one optimizer "round"); ci/pi = per-network
        # round multipliers. accelerator.backward divides by gas, the extra /ci|/pi
        # makes each grad the mean over its interval*gas-micro-step window.
        gas = self.accelerator.gradient_accumulation_steps
        ci = self.training_args.critic_update_interval
        pi = self.training_args.policy_update_interval
        do_critic = self.training_args.update_critic_in_optimize
        sorted_ts = self._sorted_train_timesteps()

        # AcceleratedOptimizer.step()/zero_grad() gate on accelerator.sync_gradients; PPO drives
        # accumulation manually (no accelerator.accumulate), so pin it True so the boundary
        # step/zero always fire regardless of any prior state.
        self.accelerator.sync_gradients = True

        # Manual accumulation: micro_in_round is continuous across epochs, so one optimize()'s
        # micro-step count must be a multiple of gas, else a window straddles two epochs'
        # rollouts (mismatched old policy / advantages). auto-gas guarantees this by construction.
        if self.training_args._manual_gradient_accumulation_steps:
            steps_per_optimize = self.training_args.num_inner_epochs * num_batches * len(sorted_ts)
            if steps_per_optimize % gas != 0:
                raise ValueError(
                    f"PPO manual gradient_accumulation_steps={gas} must divide the per-optimize "
                    f"micro-step count ({steps_per_optimize} = num_inner_epochs "
                    f"{self.training_args.num_inner_epochs} x num_batches {num_batches} x "
                    f"num_sde_steps {len(sorted_ts)}); otherwise an accumulation window spans "
                    f"two epochs' rollouts. Use a divisor or gradient_accumulation_steps='auto'."
                )

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
                    round_idx = self.step
                    boundary = self.micro_in_round == gas - 1
                    # Critic trains here only in mode A; the policy trains every
                    # micro-step (the initial warmup is an up-front bootstrap, not a
                    # frozen-policy phase). Each net steps at its own window end.
                    critic_step_now = do_critic and boundary and (round_idx + 1) % ci == 0
                    policy_step_now = boundary and (round_idx + 1) % pi == 0

                    t = batch["timesteps"][:, timestep_index]
                    latents = batch["all_latents"][:, latents_index_map[timestep_index]]

                    # 1. Critic value loss (mode A only; mode B trains it via bursts).
                    #    no_sync defers the DDP all-reduce until the critic's window end.
                    value_critic_term = None  # detached vf_coef*value_loss for the logged total
                    if do_critic:
                        returns_t = batch["step_returns"][:, step_idx]
                        old_values = batch["step_old_values"][:, step_idx]
                        critic_ctx = (
                            nullcontext()
                            if critic_step_now
                            else self.accelerator.no_sync(self.critic)
                        )
                        with critic_ctx:
                            with self.autocast():
                                value = self.critic(latents, t, self.latent_channel_dim)
                            value = value.float()
                            value_loss = self._critic_value_loss(value, returns_t, old_values)
                            # accelerate divides by gas; the extra /ci -> mean over ci*gas.
                            self.accelerator.backward(vf_coef * value_loss / ci)
                        value_critic_term = (vf_coef * value_loss).detach()
                        loss_info["value_loss"].append(value_loss.detach())
                        loss_info["explained_variance"].append(
                            self._explained_variance(value.detach(), returns_t).detach()
                        )

                    # 2. Policy update (PPO-clip [+ KL]) -- every micro-step.
                    old_log_prob = batch["log_probs"][:, log_probs_index_map[timestep_index]]
                    t_next = (
                        batch["timesteps"][:, timestep_index + 1]
                        if timestep_index + 1 < num_timesteps
                        else torch.tensor(0, device=device)
                    )
                    next_latents = batch["all_latents"][:, latents_index_map[timestep_index + 1]]
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
                            forward_inputs["return_kwargs"] = ["log_prob", "noise_pred", "dt"]
                        else:
                            forward_inputs["return_kwargs"] = [
                                "log_prob",
                                "next_latents_mean",
                                "dt",
                            ]
                    else:
                        forward_inputs["return_kwargs"] = ["log_prob", "dt"]

                    kl_loss = None
                    policy_ctx = (
                        nullcontext()
                        if policy_step_now
                        else self.accelerator.no_sync(self.model_bundle)
                    )
                    with policy_ctx:
                        with self.autocast():
                            output = self.adapter.forward(**forward_inputs)
                        ratio = torch.exp(output.log_prob - old_log_prob)
                        clipped_ratio = torch.clamp(ratio, 1.0 + clip_range[0], 1.0 + clip_range[1])
                        policy_loss = torch.mean(torch.maximum(-adv * ratio, -adv * clipped_ratio))
                        total = policy_loss
                        if self.enable_kl_loss:
                            kl_div = self._compute_kl(forward_inputs, output)
                            kl_loss = self.training_args.kl_beta * kl_div
                            total = total + kl_loss
                            loss_info["kl_div"].append(kl_div.detach())
                            loss_info["kl_loss"].append(kl_loss.detach())
                        # accelerate divides by gas; the extra /pi -> mean over pi*gas.
                        self.accelerator.backward(total / pi)

                    # 3. Independent optimizer steps at each network's own window end.
                    critic_grad_norm = None
                    policy_grad_norm = None
                    if critic_step_now:
                        critic_grad_norm = self.accelerator.clip_grad_norm_(
                            self.critic.parameters(), self.training_args.max_grad_norm
                        )
                        self.critic_optimizer.step()
                        self.critic_optimizer.zero_grad()
                    if policy_step_now:
                        policy_grad_norm = self.accelerator.clip_grad_norm_(
                            self.adapter.get_trainable_parameters(),
                            self.training_args.max_grad_norm,
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # 4. Metrics (policy every micro-step; critic metrics logged in step 1).
                    combined = policy_loss.detach()
                    if kl_loss is not None:
                        combined = combined + kl_loss.detach()
                    if value_critic_term is not None:
                        combined = combined + value_critic_term
                    clip_frac_high = torch.mean((ratio > 1.0 + clip_range[1]).float())
                    clip_frac_low = torch.mean((ratio < 1.0 + clip_range[0]).float())
                    loss_info["ratio"].append(ratio.mean().detach())
                    loss_info["policy_loss"].append(policy_loss.detach())
                    loss_info["clip_frac_high"].append(clip_frac_high.detach())
                    loss_info["clip_frac_low"].append(clip_frac_low.detach())
                    loss_info["clip_frac_total"].append((clip_frac_high + clip_frac_low).detach())
                    loss_info["loss"].append(combined)

                    # 5. Round bookkeeping at the gas-window boundary.
                    if boundary:
                        reduced = reduce_loss_info(self.accelerator, loss_info)
                        self._log_optimize_round(
                            reduced,
                            do_critic=do_critic,
                            critic_grad_norm=critic_grad_norm,
                            policy_grad_norm=policy_grad_norm,
                            critic_step_now=critic_step_now,
                            policy_step_now=policy_step_now,
                        )
                        self.step += 1
                        self.micro_in_round = 0
                        loss_info = defaultdict(list)
                    else:
                        self.micro_in_round += 1

    def _critic_value_loss(
        self,
        value: torch.Tensor,
        returns_t: torch.Tensor,
        old_values: torch.Tensor,
    ) -> torch.Tensor:
        """Clipped (PPO-style) value-regression loss, shared by optimize() + re-warmup."""
        value_clip_range = self.training_args.value_clip_range
        if value_clip_range is not None:
            value_clipped = old_values + torch.clamp(
                value - old_values, -value_clip_range, value_clip_range
            )
            return 0.5 * torch.mean(
                torch.maximum((value - returns_t) ** 2, (value_clipped - returns_t) ** 2)
            )
        return 0.5 * torch.mean((value - returns_t) ** 2)

    # =========================== Critic re-warmup (Stage 6b) ===========================
    def _warmup_critic(self, target_steps: int, label: str = "warmup") -> None:
        """Critic-only warmup: resample fresh rollouts and fit ONLY the critic until it has
        taken ~``target_steps`` critic optimizer steps (the policy is untouched).

        Shared by the initial bootstrap (``critic_warmup_steps``, before the epoch loop) and
        the periodic re-warmup (``critic_rewarmup_steps``, every ``critic_warmup_interval``
        rounds); the two targets are independent. No-op when ``target_steps <= 0``.

        Args:
            target_steps: Critic optimizer steps to accumulate across resampled bursts.
            label: Short tag for logging (e.g. ``"bootstrap"`` / ``"rewarmup"``).
        """
        if target_steps <= 0:
            return
        if self.accelerator.is_local_main_process:
            logger.info(f"PPO critic {label}: ~{target_steps} critic steps via resampled bursts.")
        done = 0
        while done < target_steps:
            steps = self._rewarmup_critic()
            if steps == 0:
                logger.warning(
                    f"PPO critic {label} made no progress (gas > buffer?); stopping early."
                )
                break
            done += steps

    def _rewarmup_critic(self, num_passes: int = 1) -> int:
        """One critic re-warmup burst: resample fresh data, then a critic-only fit.

        Args:
            num_passes: Critic-only passes over the freshly sampled buffer.

        Returns:
            Number of critic optimizer steps taken in this burst.
        """
        # Distinct, reproducible seed per burst so each resample sees fresh noise.
        self._rewarmup_calls += 1
        self.adapter.scheduler.set_seed(self.training_args.seed + 1_000_000 + self._rewarmup_calls)
        samples = self.sample()
        self.prepare_feedback(samples, metrics_prefix="train_rewarmup")
        return self._fit_critic_only(samples, num_passes)

    def _fit_critic_only(self, samples: List[BaseSample], num_passes: int) -> int:
        """Critic-only fit over ``samples`` for ``num_passes`` passes (policy untouched).

        Mirrors the critic path of :meth:`optimize` (manual ``gas`` accumulation +
        per-step ``no_sync``) but never forwards/steps the policy and never advances
        ``self.step`` / ``self.epoch``. Logs to the ``train/critic/*`` panel on the
        ``critic_step`` axis (continuous across bootstrap / re-warmup / optimize).

        Returns:
            Number of critic optimizer steps taken.
        """
        if self.per_sample_sde:
            return self._fit_critic_only_bandit(samples, num_passes)
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        vf_coef = self.training_args.vf_coef
        gas = self.accelerator.gradient_accumulation_steps
        sorted_ts = self._sorted_train_timesteps()

        self.critic.train()
        # AcceleratedOptimizer.step()/zero_grad() gate on accelerator.sync_gradients; pin it
        # True here too so bootstrap/burst critic steps fire (see optimize()).
        self.accelerator.sync_gradients = True
        # Clean slate so a burst never mixes in stale grads from a prior phase.
        self.critic_optimizer.zero_grad()
        critic_steps = 0
        micro = 0
        loss_info = defaultdict(list)
        for pass_idx in range(num_passes):
            shuffled_samples = self._order_samples_for_optimize(samples, pass_idx)
            for batch_idx in tqdm(
                range(num_batches),
                total=num_batches,
                desc=f"Rewarmup {self._rewarmup_calls}",
                leave=False,
                disable=not self.show_progress_bar,
            ):
                start = batch_idx * per_device_batch_size
                batch_samples = [
                    s.to(device) for s in shuffled_samples[start : start + per_device_batch_size]
                ]
                batch = BaseSample.stack(batch_samples)
                for key in ("step_returns", "step_old_values"):
                    batch[key] = batch[key].to(device)
                latents_index_map = batch["latent_index_map"]
                for step_idx, timestep_index in enumerate(sorted_ts):
                    boundary = micro == gas - 1
                    t = batch["timesteps"][:, timestep_index]
                    latents = batch["all_latents"][:, latents_index_map[timestep_index]]
                    returns_t = batch["step_returns"][:, step_idx]
                    old_values = batch["step_old_values"][:, step_idx]

                    critic_ctx = (
                        nullcontext() if boundary else self.accelerator.no_sync(self.critic)
                    )
                    with critic_ctx:
                        with self.autocast():
                            value = self.critic(latents, t, self.latent_channel_dim)
                        value = value.float()
                        value_loss = self._critic_value_loss(value, returns_t, old_values)
                        self.accelerator.backward(vf_coef * value_loss)

                    loss_info["value_loss"].append(value_loss.detach())
                    loss_info["explained_variance"].append(
                        self._explained_variance(value.detach(), returns_t).detach()
                    )
                    if boundary:
                        self.accelerator.clip_grad_norm_(
                            self.critic.parameters(), self.training_args.max_grad_norm
                        )
                        self.critic_optimizer.step()
                        self.critic_optimizer.zero_grad()
                        critic_steps += 1
                        reduced = reduce_loss_info(self.accelerator, loss_info)
                        self._log_critic_only(reduced)
                        loss_info = defaultdict(list)
                        micro = 0
                    else:
                        micro += 1
        # Discard any trailing partial window (manual gas not dividing the buffer) so
        # it never leaks into the next burst or the main optimize loop.
        if micro != 0:
            self.critic_optimizer.zero_grad()
        return critic_steps

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
