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

# src/flow_factory/trainers/motv.py
"""
MoTV (Mixture of Temporal Velocities) Trainer.

Learns optimal per-timestep softmax mixing weights (lambda_k_i) over K frozen
teacher flow-matching velocities, optimized via the DiffusionNFT algorithm
with external reward feedback.

Theoretical basis: The linear combination ``sum_k lambda_k_i * v_k(x, t_i)``
is itself a valid flow-matching velocity field (ODE linearity), enabling
direct application of DiffusionNFT for reward-based RL.

Total learnable parameters: K × num_inference_steps (one weight per teacher
per denoising step, with softmax normalization over K).
"""
import os
from typing import List, Dict, Any, Optional
from functools import partial
from collections import defaultdict
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .abc import BaseTrainer
from ..hparams import MoTVTrainingArguments
from ..samples import BaseSample
from ..rewards import RewardBuffer
from ..ema import EMAModuleWrapper
from ..utils.base import filter_kwargs, create_generator, to_broadcast_tensor
from ..utils.logger_utils import setup_logger
from ..utils.noise_schedule import TimeSampler, flow_match_sigma
from ..utils.dist import reduce_loss_info
from .opd.common import load_teachers, cache_forward_signature, filter_forward_kwargs
from .ensemble_eval.common import (
    cache_scheduler_step_signature,
    _build_scheduler_step_kwargs,
)

logger = setup_logger(__name__)


class MoTVTrainer(BaseTrainer):
    """
    MoTV (Mixture of Temporal Velocities): learns per-timestep softmax mixing
    weights over K frozen teacher velocities, optimized via DiffusionNFT with
    external reward.

    Key differences from DiffusionNFTTrainer:
    - No model (LoRA) weights are trained; only K×T logits get gradient.
    - ``new_v_pred`` and ``old_v_pred`` are lambda-weighted combinations of
      frozen teacher velocity predictions.
    - EMA is maintained on the logits (not model weights) for off-policy.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: MoTVTrainingArguments

        # ---- Unpack config ----
        self.nft_beta = self.training_args.nft_beta
        self.off_policy = self.training_args.off_policy
        self.temperature = self.training_args.temperature
        self.time_sampling_strategy = self.training_args.time_sampling_strategy
        self.time_shift = self.training_args.time_shift
        self.timestep_range = self.training_args.timestep_range
        # T = num_inference_steps (1:1 mapping)
        self.num_train_timesteps = self.training_args.num_inference_steps

        # ---- Load K teachers ----
        self._teacher_names: List[str] = load_teachers(
            self.adapter,
            list(self.training_args.teacher_paths),
            self.training_args.teacher_param_device,
        )
        self.K = len(self._teacher_names)
        logger.info(
            f"MoTV: {self.K} teacher(s) loaded, "
            f"T={self.num_train_timesteps} timesteps, "
            f"total learnable params = {self.K * self.num_train_timesteps}"
        )

        # ---- Learnable lambda logits: shape (K, T) ----
        self._lambda_logits = self._init_lambda_logits(self.K, self.num_train_timesteps)

        # ---- Custom optimizer for logits only ----
        self._logits_optimizer = torch.optim.AdamW(
            [self._lambda_logits],
            lr=self.training_args.logits_lr,
            betas=self.training_args.adam_betas,
            weight_decay=0.0,  # No weight decay on logits
        )

        # ---- EMA for logits (off-policy old policy) ----
        self._logits_ema = EMAModuleWrapper(
            parameters=[self._lambda_logits],
            decay=self.training_args.logits_ema_decay,
            update_step_interval=1,
            device=self.accelerator.device,
            decay_schedule="power",
        )

        # ---- Cache forward signature for cheap kwarg filtering ----
        self._forward_param_names, self._forward_accepts_var_kwargs = (
            cache_forward_signature(self.adapter.forward)
        )

        # ---- Cache scheduler step signature ----
        self._sched_cache = cache_scheduler_step_signature(self.adapter.scheduler.step)

    # =========================================================================
    # Initialization Helpers
    # =========================================================================

    def _init_lambda_logits(self, K: int, T: int) -> nn.Parameter:
        """Create the (K, T) learnable logits parameter on accelerator device.

        - ``'zeros'``: All logits are 0 → softmax gives exactly uniform 1/K.
          All teachers start with equal weight; symmetry broken only by gradient.
        - ``'random'``: Small Gaussian noise (std=0.01) → softmax is near-uniform
          but with broken symmetry, allowing faster differentiation early on.
        """
        if self.training_args.logits_init == "zeros":
            data = torch.zeros(K, T)
        elif self.training_args.logits_init == "random":
            data = torch.randn(K, T) * 0.01
        else:
            raise ValueError(
                f"Invalid logits_init: {self.training_args.logits_init!r}. "
                f"Valid options are: ['zeros', 'random']."
            )

        param = nn.Parameter(data.to(self.accelerator.device))
        return param

    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Override: freeze adapter LoRA params; return dummy optimizer.

        BaseTrainer calls this and passes result to accelerator.prepare().
        For MoTV, all adapter parameters are frozen — we manage
        _logits_optimizer separately (not wrapped by accelerator since
        the parameter is a single small tensor, no DDP needed).
        """
        # Freeze all adapter LoRA parameters
        for p in self.adapter.get_trainable_parameters():
            p.requires_grad_(False)

        # Empty optimizer satisfies BaseTrainer interface without creating dummy params
        self.optimizer = torch.optim.AdamW([], lr=0)
        return self.optimizer

    # =========================================================================
    # Lambda Weights
    # =========================================================================

    def _get_lambda_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute softmax weights from logits.

        Args:
            logits: Tensor of shape (K, T).

        Returns:
            Weights of shape (K, T), normalized over K (dim=0) per timestep.
        """
        return F.softmax(logits / self.temperature, dim=0)

    @property
    def enable_kl_loss(self) -> bool:
        """Check if KL penalty is enabled."""
        return self.training_args.kl_beta > 0.0

    # =========================================================================
    # Context Managers
    # =========================================================================

    @contextmanager
    def sampling_context(self):
        """Use EMA logits for sampling when off-policy."""
        if self.off_policy:
            with self._logits_ema.use_ema_parameters([self._lambda_logits]):
                yield
        else:
            yield

    @contextmanager
    def _motv_inference_context(self):
        """Patch adapter.forward to return lambda-combined teacher velocity.

        During sampling (rollout), replaces adapter.forward so that each
        denoising step:
        1. Runs each teacher forward → collects K noise_pred tensors
        2. Combines them with current lambda weights (detached, no grad in sampling)
        3. Passes combined noise_pred to scheduler.step

        Step counter provides 1:1 mapping to logits columns.
        """
        original_forward = self.adapter.forward
        step_counter = [0]

        # Precompute weights (detached since sampling is under no_grad)
        with torch.no_grad():
            weights = self._get_lambda_weights(self._lambda_logits)  # (K, T)

        def patched_forward(**kwargs):
            # Current denoising step index (1:1 with logits column)
            t_idx = min(step_counter[0], self.num_train_timesteps - 1)
            step_counter[0] += 1

            # Get noise_pred from each teacher
            noise_only_kwargs = dict(kwargs)
            noise_only_kwargs["return_kwargs"] = ["noise_pred"]

            velocities = []
            for name in self._teacher_names:
                with self.adapter.use_named_parameters(name):
                    out = original_forward(**noise_only_kwargs)
                velocities.append(out.noise_pred)

            # Combine using lambda weights at this timestep
            stacked = torch.stack(velocities, dim=0)  # (K, B, ...)
            w_i = weights[:, t_idx]  # (K,)
            expand_shape = (self.K,) + (1,) * (stacked.ndim - 1)
            combined_noise_pred = (w_i.view(*expand_shape) * stacked).sum(dim=0)

            # Run scheduler step with combined prediction
            scheduler_kwargs = _build_scheduler_step_kwargs(
                kwargs, combined_noise_pred, self._sched_cache
            )
            return self.adapter.scheduler.step(**scheduler_kwargs)

        self.adapter.forward = patched_forward  # type: ignore[method-assign]
        # Disable autocast cache (weight swaps via .data.copy_)
        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        try:
            step_counter[0] = 0
            yield
        finally:
            torch.set_autocast_cache_enabled(prev_cache)
            self.adapter.forward = original_forward  # type: ignore[method-assign]

    # =========================================================================
    # Teacher Velocity Computation
    # =========================================================================

    def _compute_teacher_velocities(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Forward each teacher and return stacked velocities (all detached).

        Disables autocast weight cache during the loop because
        ``use_named_parameters`` swaps LoRA weights via ``.data.copy_()``,
        which preserves ``data_ptr``. The autocast cache (keyed by data_ptr)
        would otherwise serve stale casted weights from the first teacher
        to all subsequent teachers.

        Args:
            batch: Batch containing prompt embeddings and other inputs.
            timestep: Timestep tensor of shape (B,) in scheduler scale [0, 1000].
            noised_latents: Interpolated latents x_t.

        Returns:
            Tensor of shape (K, B, *latent_dims) with all teacher predictions.
        """
        forward_kwargs = self._build_forward_kwargs(batch, timestep, noised_latents)
        velocities = []

        # Must disable autocast cache when swapping named parameters
        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        try:
            for name in self._teacher_names:
                with self.adapter.use_named_parameters(name):
                    output = self.adapter.forward(**forward_kwargs)
                velocities.append(output.noise_pred.detach())
        finally:
            torch.set_autocast_cache_enabled(prev_cache)

        return torch.stack(velocities, dim=0)  # (K, B, C, ...)

    def _combine_velocities(
        self,
        teacher_velocities: torch.Tensor,
        t_idx: int,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Combine teacher velocities using precomputed lambda weights at timestep t_idx.

        Args:
            teacher_velocities: (K, B, *latent_dims) — detached teacher preds.
            t_idx: Index into T dimension of weights.
            weights: Precomputed softmax weights of shape (K, T).

        Returns:
            Combined velocity of shape (B, *latent_dims). Gradient flows
            through weights (and thus logits) only.
        """
        w_i = weights[:, t_idx]  # (K,)

        # Broadcast: w_i (K,) → (K, 1, 1, ...) to match teacher_velocities
        expand_shape = (self.K,) + (1,) * (teacher_velocities.ndim - 1)
        w_expanded = w_i.view(*expand_shape)

        # Weighted sum over K dimension
        combined = (w_expanded * teacher_velocities).sum(dim=0)  # (B, C, ...)
        return combined

    def _build_forward_kwargs(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
    ) -> Dict[str, Any]:
        """Build kwargs for a single adapter.forward call (noise_pred only)."""
        t_b = timestep.view(-1)
        full_kwargs: Dict[str, Any] = {
            **self.training_args,
            't': t_b,
            't_next': torch.zeros_like(t_b),
            'latents': noised_latents,
            'compute_log_prob': False,
            'return_kwargs': ['noise_pred'],
            'noise_level': 0.0,
            **{k: v for k, v in batch.items()
               if k not in ['all_latents', 'timesteps', 'advantage']},
        }
        forward_kwargs = filter_forward_kwargs(
            full_kwargs, self._forward_param_names, self._forward_accepts_var_kwargs
        )
        forward_kwargs['return_kwargs'] = ['noise_pred']
        return forward_kwargs

    # =========================================================================
    # Timestep Sampling
    # =========================================================================

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample training timesteps. Returns shape (T, B) in [0, 1000].

        Since T = num_inference_steps, we sample all T timesteps.
        """
        device = self.accelerator.device
        strategy = self.time_sampling_strategy.lower()
        available = ['logit_normal', 'uniform', 'discrete', 'discrete_with_init', 'discrete_wo_init']

        if strategy == 'logit_normal':
            return TimeSampler.logit_normal_shifted(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
                stratified=True,
            )
        elif strategy == 'uniform':
            return TimeSampler.uniform(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
            )
        elif strategy.startswith('discrete'):
            discrete_config = {
                'discrete': (True, False),
                'discrete_with_init': (True, True),
                'discrete_wo_init': (False, False),
            }
            if strategy not in discrete_config:
                raise ValueError(
                    f"Unknown time_sampling_strategy: {strategy}. Available: {available}"
                )
            include_init, force_init = discrete_config[strategy]
            return TimeSampler.discrete(
                batch_size=batch_size,
                num_train_timesteps=self.num_train_timesteps,
                scheduler_timesteps=self.adapter.scheduler.timesteps,
                timestep_range=self.timestep_range,
                include_init=include_init,
                force_init=force_init,
            )
        else:
            raise ValueError(
                f"Unknown time_sampling_strategy: {strategy}. Available: {available}"
            )

    # =========================================================================
    # Advantage Computation
    # =========================================================================

    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func=None,
    ) -> torch.Tensor:
        """Compute advantages — delegates to AdvantageProcessor."""
        aggregation_func = aggregation_func or self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
        )

    # =========================================================================
    # Main Training Loop
    # =========================================================================

    def start(self):
        """Main training loop."""
        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)

            # Checkpoint
            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    'checkpoints',
                )
                self._save_motv_checkpoint(save_dir, epoch=self.epoch)

            # Evaluation
            if (
                self.eval_args.eval_freq > 0
                and self.epoch % self.eval_args.eval_freq == 0
            ):
                self.evaluate()

            # Sampling: use EMA logits if off-policy
            with self.sampling_context():
                samples = self.sample()

            self.prepare_feedback(samples)
            self.optimize(samples)

            # Update EMA of logits
            self._logits_ema.step([self._lambda_logits], optimization_step=self.epoch)
            self.epoch += 1

    # =========================================================================
    # Sampling
    # =========================================================================

    def sample(self) -> List[BaseSample]:
        """Generate rollouts using lambda-combined teacher velocity."""
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []
        data_iter = iter(self.dataloader)

        with torch.no_grad(), self.autocast(), self._motv_inference_context():
            for _ in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': False,
                    'trajectory_indices': [-1],  # Only keep final latents
                    **batch,
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                self._maybe_offload_samples_to_cpu(sample_batch)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)

        return samples

    # =========================================================================
    # Feedback
    # =========================================================================

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Finalize rewards, compute advantages, and log metrics."""
        rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')
        self.compute_advantages(samples, rewards, store_to_samples=True)
        adv_metrics = self.advantage_processor.pop_advantage_metrics()
        if adv_metrics:
            self.log_data(adv_metrics, step=self.step)

    # =========================================================================
    # Optimization
    # =========================================================================

    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization: NFT loss on lambda-weighted teacher velocities.

        For each micro-batch:
        1. Precompute old_v_pred using EMA logits at all T timesteps (no_grad).
           Cache noised_latents but NOT teacher velocities (too large for video).
        2. Recompute teacher velocities per timestep (deterministic, frozen) and
           combine with current logits for new_v_pred.
        3. Accumulate gradients over all T timesteps, single optimizer step per batch.
        """
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size

        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle samples
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]

            loss_info = defaultdict(list)

            for batch_idx in tqdm(
                range(num_batches),
                total=num_batches,
                desc=f'Epoch {self.epoch} Training',
                position=0,
                disable=not self.show_progress_bar,
            ):
                start = batch_idx * per_device_batch_size
                batch_samples = [
                    sample.to(device)
                    for sample in shuffled_samples[start:start + per_device_batch_size]
                ]
                batch = BaseSample.stack(batch_samples)
                batch_size = batch['all_latents'].shape[0]
                clean_latents = batch['all_latents'][:, -1]

                # ---- Phase 1: Precompute noised_latents + old_v_pred ----
                # Teacher velocities are NOT cached (too large); recomputed in Phase 2.
                self.adapter.rollout()
                with torch.no_grad(), self.autocast():
                    all_timesteps = self._sample_timesteps(batch_size)  # (T, B)
                    all_noised_latents: List[torch.Tensor] = []
                    all_sigma_broadcast: List[torch.Tensor] = []
                    old_v_pred_list: List[torch.Tensor] = []

                    # EMA weights for old_v_pred (read once under EMA context)
                    with self.sampling_context():
                        ema_weights = self._get_lambda_weights(self._lambda_logits)  # (K, T)

                    for t_idx in range(self.num_train_timesteps):
                        t_flat = all_timesteps[t_idx]  # (B,) in [0, 1000]
                        sigma_broadcast = to_broadcast_tensor(
                            flow_match_sigma(t_flat), clean_latents
                        )
                        noise = randn_tensor(
                            clean_latents.shape,
                            device=clean_latents.device,
                            dtype=clean_latents.dtype,
                        )
                        noised_latents = (
                            (1 - sigma_broadcast) * clean_latents
                            + sigma_broadcast * noise
                        )
                        all_noised_latents.append(noised_latents)
                        all_sigma_broadcast.append(sigma_broadcast)

                        # Get teacher velocities and combine with EMA weights
                        teacher_velocities = self._compute_teacher_velocities(
                            batch, t_flat, noised_latents
                        )  # (K, B, ...)
                        w_i = ema_weights[:, t_idx]  # (K,)
                        expand_shape = (self.K,) + (1,) * (teacher_velocities.ndim - 1)
                        old_v = (w_i.view(*expand_shape) * teacher_velocities).sum(dim=0)
                        old_v_pred_list.append(old_v.detach())

                # ---- Phase 2: Train with current logits ----
                # Precompute current weights ONCE (gradient retained for backward)
                current_weights = self._get_lambda_weights(self._lambda_logits)  # (K, T)

                with self.autocast():
                    for t_idx in tqdm(
                        range(self.num_train_timesteps),
                        desc=f'Epoch {self.epoch} Timestep',
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    ):
                        t_flat = all_timesteps[t_idx]
                        sigma_broadcast = all_sigma_broadcast[t_idx]
                        noised_latents = all_noised_latents[t_idx]
                        old_v_pred = old_v_pred_list[t_idx]

                        # Recompute teacher velocities (deterministic, no_grad)
                        with torch.no_grad():
                            teacher_velocities = self._compute_teacher_velocities(
                                batch, t_flat, noised_latents
                            )

                        # Combine with CURRENT weights → new_v_pred (grad through logits)
                        new_v_pred = self._combine_velocities(
                            teacher_velocities, t_idx, current_weights
                        )

                        # NFT loss computation
                        adv = batch['advantage']
                        adv_clip_range = self.training_args.adv_clip_range
                        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])

                        # Normalize advantage to [0, 1]
                        normalized_adv = (adv / max(adv_clip_range)) / 2.0 + 0.5
                        r = torch.clamp(normalized_adv, 0, 1).view(
                            -1, *([1] * (new_v_pred.dim() - 1))
                        )

                        # Positive/negative predictions
                        positive_pred = (
                            self.nft_beta * new_v_pred
                            + (1 - self.nft_beta) * old_v_pred
                        )
                        negative_pred = (
                            (1.0 + self.nft_beta) * old_v_pred
                            - self.nft_beta * new_v_pred
                        )

                        # Positive loss
                        x0_pred = noised_latents - sigma_broadcast * positive_pred
                        with torch.no_grad():
                            weight = (
                                torch.abs(x0_pred.double() - clean_latents.double())
                                .mean(dim=tuple(range(1, clean_latents.ndim)), keepdim=True)
                                .clip(min=1e-5)
                            )
                        positive_loss = (
                            (x0_pred - clean_latents) ** 2 / weight
                        ).mean(dim=tuple(range(1, clean_latents.ndim)))

                        # Negative loss
                        neg_x0_pred = noised_latents - sigma_broadcast * negative_pred
                        with torch.no_grad():
                            neg_weight = (
                                torch.abs(neg_x0_pred.double() - clean_latents.double())
                                .mean(dim=tuple(range(1, clean_latents.ndim)), keepdim=True)
                                .clip(min=1e-5)
                            )
                        negative_loss = (
                            (neg_x0_pred - clean_latents) ** 2 / neg_weight
                        ).mean(dim=tuple(range(1, clean_latents.ndim)))

                        # Combined loss
                        ori_policy_loss = (
                            r.squeeze() * positive_loss
                            + (1.0 - r.squeeze()) * negative_loss
                        ) / self.nft_beta
                        policy_loss = (ori_policy_loss * adv_clip_range[1]).mean()
                        loss = policy_loss

                        # Optional KL penalty (v-based, to base model)
                        if self.enable_kl_loss:
                            with torch.no_grad(), self.adapter.use_ref_parameters():
                                ref_forward_kwargs = self._build_forward_kwargs(
                                    batch, t_flat, noised_latents
                                )
                                ref_output = self.adapter.forward(**ref_forward_kwargs)
                            kl_div = torch.mean(
                                (new_v_pred - ref_output.noise_pred) ** 2,
                                dim=tuple(range(1, new_v_pred.ndim)),
                            )
                            kl_loss = self.training_args.kl_beta * kl_div.mean()
                            loss = loss + kl_loss
                            loss_info['kl_div'].append(kl_div.detach())
                            loss_info['kl_loss'].append(kl_loss.detach())

                        loss_info['policy_loss'].append(policy_loss.detach())
                        loss_info['unweighted_policy_loss'].append(
                            ori_policy_loss.mean().detach()
                        )
                        loss_info['loss'].append(loss.detach())

                        # Accumulate gradients (single step per batch below)
                        (loss / self.num_train_timesteps).backward()

                # Single optimizer step per batch (after all T timesteps)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [self._lambda_logits],
                    self.training_args.max_grad_norm,
                )
                self._logits_optimizer.step()
                self._logits_optimizer.zero_grad()

                # Log batch-level metrics
                loss_info_reduced = reduce_loss_info(self.accelerator, loss_info)
                loss_info_reduced['grad_norm'] = grad_norm
                with torch.no_grad():
                    log_weights = self._get_lambda_weights(self._lambda_logits)
                    for k in range(self.K):
                        loss_info_reduced[f'lambda_teacher_{k}_mean'] = (
                            log_weights[k].mean().item()
                        )
                self.log_data(
                    {f'train/{k}': v for k, v in loss_info_reduced.items()},
                    step=self.step,
                )
                self.step += 1
                loss_info = defaultdict(list)

    # =========================================================================
    # Evaluation (uses lambda-combined inference)
    # =========================================================================

    @contextmanager
    def _eval_inference_context(self):
        """Override: use MoTV inference patching for evaluation too."""
        with self._motv_inference_context():
            yield

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def _save_motv_checkpoint(self, save_directory: str, epoch: Optional[int] = None):
        """Save MoTV-specific state (lambda logits + EMA)."""
        if epoch is not None:
            save_directory = os.path.join(save_directory, f"checkpoint-{epoch}")

        if self.accelerator.is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            state = {
                'lambda_logits': self._lambda_logits.detach().cpu(),
                'logits_ema': self._logits_ema.state_dict(),
                'epoch': self.epoch,
                'step': self.step,
                'K': self.K,
                'T': self.num_train_timesteps,
            }
            save_path = os.path.join(save_directory, 'motv_state.pt')
            torch.save(state, save_path)
            logger.info(f"MoTV checkpoint saved to {save_path}")

        self.accelerator.wait_for_everyone()

    def load_motv_checkpoint(self, path: str):
        """Load MoTV state from checkpoint."""
        state_path = os.path.join(path, 'motv_state.pt')
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"MoTV checkpoint not found at {state_path}")

        state = torch.load(state_path, map_location=self.accelerator.device)

        # Validate dimensions
        if state['K'] != self.K or state['T'] != self.num_train_timesteps:
            raise ValueError(
                f"Checkpoint K×T = {state['K']}×{state['T']} does not match "
                f"current config K×T = {self.K}×{self.num_train_timesteps}"
            )

        self._lambda_logits.data.copy_(state['lambda_logits'].to(self.accelerator.device))
        self._logits_ema.load_state_dict(state['logits_ema'])
        self.epoch = state['epoch']
        self.step = state['step']
        logger.info(f"MoTV checkpoint loaded from {state_path} (epoch={self.epoch})")
