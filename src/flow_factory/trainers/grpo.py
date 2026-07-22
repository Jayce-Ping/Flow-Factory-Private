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

# src/flow_factory/trainers/grpo.py
"""
Group Relative Policy Optimization (GRPO) Trainer.
Implements GRPO algorithm for flow matching models.
"""
import os
from typing import List, Dict, Optional, Any, Union, Literal, Callable
from functools import partial
from collections import defaultdict
import torch
import numpy as np
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .abc import BaseTrainer
from ..hparams import GRPOTrainingArguments
from ..samples import BaseSample
from ..utils.base import filter_kwargs, create_generator_by_prompt
from ..utils.logger_utils import setup_logger
from ..utils.trajectory_collector import TrajectoryCollector, compute_trajectory_indices
from ..utils.dist import reduce_loss_info

logger = setup_logger(__name__)


# ============================ GRPO Trainer ============================
class GRPOTrainer(BaseTrainer):
    """
    GRPO Trainer for Flow Matching models.
    Implements group-based advantage computation and PPO-style clipping.
    References:
    [1] Flow-GRPO: Training Flow Matching Models via Online RL
        - https://arxiv.org/abs/2505.05470
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args : GRPOTrainingArguments
        self.num_train_timesteps = self.adapter.scheduler.num_sde_steps

    @property
    def enable_kl_loss(self) -> bool:
        """Check if KL penalty is enabled."""
        return self.training_args.kl_beta > 0.0

    def start(self):
        """Main training loop."""
        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)
            
            # Save checkpoint
            if (
                self.log_args.save_freq > 0 and 
                self.epoch % self.log_args.save_freq == 0 and 
                self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    'checkpoints',
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            # Evaluation
            if (
                self.eval_args.eval_freq > 0 and
                self.epoch % self.eval_args.eval_freq == 0
            ):
                self.evaluate()

            samples = self.sample()
            self.prepare_feedback(samples)
            self.optimize(samples)

            self.adapter.ema_step(step=self.epoch)

            self.epoch += 1

    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for GRPO (stores full trajectory + log-probs)."""
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.scheduler.train_timesteps,
            num_inference_steps=self.training_args.num_inference_steps,
        )
        return self.generate_samples(
            reward_buffer=self.reward_buffer,
            compute_log_prob=True,
            trajectory_indices=trajectory_indices,
        )

    # =========================== Reward / advantage (Stages 4--5) ============================
    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Finalize rewards from the buffer, compute advantages, and log advantage metrics."""
        rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')
        self.compute_advantages(samples, rewards, store_to_samples=True)
        adv_metrics = self.advantage_processor.pop_advantage_metrics()
        if adv_metrics:
            self.log_data(adv_metrics, step=self.step)

    # =========================== Optimization Loop ============================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): step / sum / gspo policy losses + optional KL."""
        policy_loss_type = self.training_args.policy_loss_type
        if policy_loss_type == 'step':
            self._optimize_step(samples)
        elif policy_loss_type in ('sum', 'gspo'):
            self._optimize_traj(samples, policy_loss_type=policy_loss_type)
        else:
            raise ValueError(
                f"Unsupported policy_loss_type: {policy_loss_type!r}. "
                "Expected one of ['step', 'sum', 'gspo']."
            )

    def _optimize_step(self, samples: List[BaseSample]) -> None:
        """Per-SDE-step importance ratio + clip (original Flow-GRPO)."""
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        for inner_epoch in range(self.training_args.num_inner_epochs):
            shuffled_samples = self._order_samples_for_optimize(samples, inner_epoch)
            self.adapter.train()
            loss_info = defaultdict(list)

            for batch in tqdm(
                self._iter_prefetched_batches(shuffled_samples, per_device_batch_size),
                total=num_batches,
                desc=f'Epoch {self.epoch} Training',
                position=0,
                disable=not self.show_progress_bar,
            ):
                for timestep_index in tqdm(
                    self.adapter.scheduler.train_timesteps,
                    desc=f'Epoch {self.epoch} Timestep',
                    position=1,
                    leave=False,
                    disable=not self.show_progress_bar,
                ):
                    with self.accumulate_gradients():
                        old_log_prob, forward_inputs, return_kwargs = self._prepare_timestep_forward(
                            batch, timestep_index,
                        )
                        forward_inputs['return_kwargs'] = return_kwargs
                        with self.autocast():
                            output = self.adapter.forward(**forward_inputs)

                        adv = self._clipped_advantages(batch)
                        ratio_clip_range = self.training_args.clip_range
                        ratio = torch.exp(output.log_prob - old_log_prob)
                        unclipped_loss = -adv * ratio
                        clipped_loss = -adv * torch.clamp(
                            ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1],
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        loss = policy_loss

                        if self.enable_kl_loss:
                            kl_div, kl_loss = self._compute_step_kl(output, forward_inputs)
                            loss = loss + kl_loss
                            loss_info['kl_div'].append(kl_div.detach())
                            loss_info['kl_loss'].append(kl_loss.detach())

                        loss_info['ratio'].append(ratio.detach())
                        loss_info['unclipped_loss'].append(unclipped_loss.detach())
                        loss_info['clipped_loss'].append(clipped_loss.detach())
                        loss_info['policy_loss'].append(policy_loss.detach())
                        loss_info['loss'].append(loss.detach())
                        clip_frac_high = torch.mean((ratio > 1.0 + ratio_clip_range[1]).float())
                        clip_frac_low = torch.mean((ratio < 1.0 + ratio_clip_range[0]).float())
                        loss_info['clip_frac_high'].append(clip_frac_high.detach())
                        loss_info['clip_frac_low'].append(clip_frac_low.detach())
                        loss_info['clip_frac_total'].append((clip_frac_high + clip_frac_low).detach())

                        loss_info = self._backward_and_maybe_step(loss, loss_info)

    def _optimize_traj(self, samples: List[BaseSample], policy_loss_type: str) -> None:
        """Trajectory-level sum (REINFORCE) or gspo (traj IS + one clip).

        All SDE steps in ``scheduler.train_timesteps`` share one advantage and
        one backward. Epoch-level ``set_seed`` keeps the SDE index set shared
        between sampling and this loop.
        """
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        train_timesteps = self.adapter.scheduler.train_timesteps
        num_sde = int(len(train_timesteps))

        for inner_epoch in range(self.training_args.num_inner_epochs):
            shuffled_samples = self._order_samples_for_optimize(samples, inner_epoch)
            self.adapter.train()
            loss_info = defaultdict(list)

            for batch in tqdm(
                self._iter_prefetched_batches(shuffled_samples, per_device_batch_size),
                total=num_batches,
                desc=f'Epoch {self.epoch} Training ({policy_loss_type})',
                position=0,
                disable=not self.show_progress_bar,
            ):
                with self.accumulate_gradients():
                    new_log_probs = []
                    old_log_probs = []
                    kl_divs = []

                    for timestep_index in tqdm(
                        train_timesteps,
                        desc=f'Epoch {self.epoch} Timestep',
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    ):
                        old_log_prob, forward_inputs, return_kwargs = self._prepare_timestep_forward(
                            batch, timestep_index,
                        )
                        forward_inputs['return_kwargs'] = return_kwargs
                        with self.autocast():
                            output = self.adapter.forward(**forward_inputs)

                        new_log_probs.append(output.log_prob)
                        old_log_probs.append(old_log_prob)

                        if self.enable_kl_loss:
                            kl_div, _ = self._compute_step_kl(output, forward_inputs)
                            kl_divs.append(kl_div)

                    new_lp = torch.stack(new_log_probs, dim=1)  # (B, K)
                    old_lp = torch.stack(old_log_probs, dim=1)  # (B, K)
                    adv = self._clipped_advantages(batch)
                    logprob_sum = new_lp.sum(dim=1)  # (B,)

                    if policy_loss_type == 'sum':
                        # REINFORCE: -A * sum_k ell_new_k
                        policy_loss = torch.mean(-adv * logprob_sum)
                        loss = policy_loss
                        loss_info['logprob_sum'].append(logprob_sum.detach().mean())
                    else:
                        # gspo: rho_tau = exp(sum (ell_new - ell_old)), one clip
                        ratio_clip_range = self.training_args.clip_range
                        rho = torch.exp((new_lp - old_lp).sum(dim=1))
                        unclipped_loss = -adv * rho
                        clipped_loss = -adv * torch.clamp(
                            rho, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1],
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        loss = policy_loss
                        loss_info['rho'].append(rho.detach())
                        loss_info['unclipped_loss'].append(unclipped_loss.detach())
                        loss_info['clipped_loss'].append(clipped_loss.detach())
                        loss_info['logprob_sum'].append(logprob_sum.detach().mean())
                        clip_frac_high = torch.mean((rho > 1.0 + ratio_clip_range[1]).float())
                        clip_frac_low = torch.mean((rho < 1.0 + ratio_clip_range[0]).float())
                        loss_info['clip_frac_high'].append(clip_frac_high.detach())
                        loss_info['clip_frac_low'].append(clip_frac_low.detach())
                        loss_info['clip_frac_total'].append((clip_frac_high + clip_frac_low).detach())

                    if self.enable_kl_loss and kl_divs:
                        kl_div = torch.stack(kl_divs).mean()
                        kl_loss = self.training_args.kl_beta * kl_div
                        loss = loss + kl_loss
                        loss_info['kl_div'].append(kl_div.detach())
                        loss_info['kl_loss'].append(kl_loss.detach())

                    loss_info['policy_loss'].append(policy_loss.detach())
                    loss_info['loss'].append(loss.detach())
                    loss_info['num_sde_steps'].append(
                        torch.tensor(float(num_sde), device=loss.device),
                    )
                    loss_info = self._backward_and_maybe_step(loss, loss_info)

    def _clipped_advantages(self, batch: Dict[str, Any]) -> torch.Tensor:
        adv = batch['advantage']
        adv_clip_range = self.training_args.adv_clip_range
        return torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])

    def _prepare_timestep_forward(
        self,
        batch: Dict[str, Any],
        timestep_index: int,
    ) -> tuple[torch.Tensor, Dict[str, Any], List[str]]:
        """Build old log-prob + filtered forward kwargs for one SDE index."""
        latents_index_map = batch['latent_index_map']
        log_probs_index_map = batch['log_prob_index_map']
        old_log_prob = batch['log_probs'][:, log_probs_index_map[timestep_index]]
        num_timesteps = batch['timesteps'].shape[1]
        t = batch['timesteps'][:, timestep_index]
        t_next = (
            batch['timesteps'][:, timestep_index + 1]
            if timestep_index + 1 < num_timesteps
            else torch.tensor(0, device=self.accelerator.device)
        )
        latents = batch['all_latents'][:, latents_index_map[timestep_index]]
        next_latents = batch['all_latents'][:, latents_index_map[timestep_index + 1]]
        forward_inputs = {
            **self.training_args,
            't': t,
            't_next': t_next,
            'latents': latents,
            'next_latents': next_latents,
            'compute_log_prob': True,
            'noise_level': self.adapter.scheduler.noise_level,
            **batch,
        }
        forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)
        if self.enable_kl_loss:
            if self.training_args.kl_type == 'v-based':
                return_kwargs = ['log_prob', 'noise_pred', 'dt']
            elif self.training_args.kl_type == 'x-based':
                return_kwargs = ['log_prob', 'next_latents', 'next_latents_mean', 'dt']
            else:
                raise ValueError(f"Invalid kl_type: {self.training_args.kl_type}")
        else:
            return_kwargs = ['log_prob', 'dt']
        return old_log_prob, forward_inputs, return_kwargs

    def _compute_step_kl(
        self,
        output: Any,
        forward_inputs: Dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """KL vs ref at one timestep. Returns (kl_div, kl_loss) scalars."""
        with self.autocast():
            with torch.no_grad(), self.adapter.use_ref_parameters():
                ref_forward_inputs = forward_inputs.copy()
                ref_forward_inputs['compute_log_prob'] = False
                if self.training_args.kl_type == 'v-based':
                    ref_forward_inputs['return_kwargs'] = ['noise_pred']
                    ref_output = self.adapter.forward(**ref_forward_inputs)
                elif self.training_args.kl_type == 'x-based':
                    ref_forward_inputs['return_kwargs'] = ['next_latents_mean']
                    ref_output = self.adapter.forward(**ref_forward_inputs)
                else:
                    raise ValueError(f"Invalid kl_type: {self.training_args.kl_type}")

            # Outside no_grad so the student side keeps a correct graph when needed.
            # See: issue #122, PR #123
            if self.training_args.kl_type == 'v-based':
                kl_div = torch.mean(
                    ((output.noise_pred - ref_output.noise_pred) ** 2),
                    dim=tuple(range(1, output.noise_pred.ndim)),
                    keepdim=True,
                )
            else:
                kl_div = torch.mean(
                    ((output.next_latents_mean - ref_output.next_latents_mean) ** 2),
                    dim=tuple(range(1, output.next_latents_mean.ndim)),
                    keepdim=True,
                )
            kl_div = torch.mean(kl_div)
            kl_loss = self.training_args.kl_beta * kl_div
        return kl_div, kl_loss

    def _backward_and_maybe_step(
        self,
        loss: torch.Tensor,
        loss_info: Dict[str, list],
    ) -> Dict[str, list]:
        """Backward; on sync_gradients, clip/step/log and reset loss_info."""
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            grad_norm = self.accelerator.clip_grad_norm_(
                self.adapter.get_trainable_parameters(),
                self.training_args.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_info = reduce_loss_info(self.accelerator, loss_info)
            loss_info['grad_norm'] = grad_norm
            self.log_data(
                {f'train/{k}': v for k, v in loss_info.items()},
                step=self.step,
            )
            self.step += 1
            loss_info = defaultdict(list)
        return loss_info

    # =========================== Advantage Computation ============================
    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func: Optional[Union[Literal['sum', 'gdpo'], Callable]] = None,
    ) -> torch.Tensor:
        """Compute advantages — delegates to AdvantageProcessor.

        Args:
            samples: List of BaseSample instances
            rewards: Dict of reward_name to reward tensors aligned with samples
            store_to_samples: Whether to store computed advantages back to samples' extra_kwargs
            aggregation_func: Method to aggregate advantages within each group.
                Options: 'sum' (default GRPO), 'gdpo' (GDPO-style), or a custom callable.
        Returns:
            advantages: Tensor of shape (num_samples, ) with computed advantages
        """
        aggregation_func = aggregation_func or self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
        )


# ============================ GRPO-Guard Trainer ============================
class GRPOGuardTrainer(GRPOTrainer):
    """
    GRPOGuard Trainer with reweighted loss.
    References:
    [1] GRPO-Guard: https://arxiv.org/abs/2510.22319
    [2] Temp-FlowGRPO: https://arxiv.org/abs/2508.04324
    """

    # =========================== Sampling Loop ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts for GRPO."""
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []
        data_iter = iter(self.dataloader)
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.scheduler.train_timesteps,
            num_inference_steps=self.training_args.num_inference_steps,
        )

        with torch.no_grad(), self.autocast():
            for batch_index in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': True,
                    'trajectory_indices': trajectory_indices, # Selectively store required trajectory positions for memory efficiency
                    'extra_call_back_kwargs': ['next_latents_mean'], # For GRPO-Guard, we need to store `next_latents_mean` for ratio normalization
                    **batch,
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                # Deterministic D2H so reward_buffer sees CPU-resident samples
                # (no-op when offload_samples_to_cpu is False).
                self._maybe_offload_samples_to_cpu(sample_batch)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)

        return samples

    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): GRPO-Guard reweighted loss and optional KL."""
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle unless disabled for pack-composition-dependent adapters.
            shuffled_samples = self._order_samples_for_optimize(samples, inner_epoch)

            self.adapter.train()
            loss_info = defaultdict(list)

            # Reload each micro-batch onto the device (H2D when offloaded);
            # _iter_prefetched_batches overlaps the next batch's H2D with compute.
            for batch in tqdm(
                self._iter_prefetched_batches(shuffled_samples, per_device_batch_size),
                total=num_batches,
                desc=f'Epoch {self.epoch} Training',
                position=0,
                disable=not self.show_progress_bar,
            ):
                latents_index_map = batch['latent_index_map']  # (T+1,) LongTensor
                log_probs_index_map = batch['log_prob_index_map']  # (T,) LongTensor
                callback_index_map = batch['callback_index_map'][0]  # (T,) LongTensor, shared across batch.
                # Iterate through timesteps
                for idx, timestep_index in enumerate(tqdm(
                    self.adapter.scheduler.train_timesteps,
                    desc=f'Epoch {self.epoch} Timestep',
                    position=1,
                    leave=False,
                    disable=not self.show_progress_bar,
                )):
                    with self.accumulate_gradients():
                        # 1. Prepare inputs
                        # Get old log prob
                        old_log_prob = batch['log_probs'][:, log_probs_index_map[timestep_index]]
                        # Get current timestep data
                        num_timesteps = batch['timesteps'].shape[1]
                        t = batch['timesteps'][:, timestep_index]
                        t_next = (
                            batch['timesteps'][:, timestep_index + 1]
                            if timestep_index + 1 < num_timesteps
                            else torch.tensor(0, device=self.accelerator.device)
                        )
                        # Get latents
                        latents = batch['all_latents'][:, latents_index_map[timestep_index]]
                        next_latents = batch['all_latents'][:, latents_index_map[timestep_index + 1]]
                        # Prepare forward input
                        forward_inputs = {
                            **self.training_args, # Pass kwargs like `guidance_scale` and `do_classifier_free_guidance`
                            't': t,
                            't_next': t_next,
                            'latents': latents,
                            'next_latents': next_latents,
                            'compute_log_prob': True,
                            'noise_level': self.adapter.scheduler.noise_level,
                            **batch
                        }
                        forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)
                        # 2. Forward pass
                        return_kwargs = set(['log_prob', 'next_latents_mean', 'std_dev_t', 'dt'])
                        if self.enable_kl_loss:
                            if self.training_args.kl_type == 'v-based':
                                return_kwargs.add('noise_pred')
                            elif self.training_args.kl_type == 'x-based':
                                return_kwargs.add('next_latents_mean')

                        forward_inputs['return_kwargs'] = list(return_kwargs)
                        with self.autocast():
                            output = self.adapter.forward(**forward_inputs)

                        # 3. Compute loss
                        # Clip advantages
                        adv = batch['advantage']
                        adv_clip_range = self.training_args.adv_clip_range
                        adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                        # Reweighted ratio
                        scale_factor = torch.sqrt(-output.dt) * output.std_dev_t
                        old_next_latents_mean = batch['next_latents_mean'][:, callback_index_map[timestep_index]]
                        mse = (output.next_latents_mean - old_next_latents_mean).flatten(1).pow(2).mean(dim=1)
                        ratio = torch.exp((output.log_prob - old_log_prob) * scale_factor + mse / (2 * scale_factor))
                        # PPO-style clipped loss
                        ratio_clip_range = self.training_args.clip_range

                        unclipped_loss = -adv * ratio
                        clipped_loss = -adv * torch.clamp(ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1])
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        loss = policy_loss

                        # 4. Compute KL-div
                        if self.enable_kl_loss:
                            with self.autocast():
                                with torch.no_grad(), self.adapter.use_ref_parameters():
                                    ref_forward_inputs = forward_inputs.copy()
                                    ref_forward_inputs['compute_log_prob'] = False
                                    if self.training_args.kl_type == 'v-based':
                                        # KL in velocity space
                                        ref_forward_inputs['return_kwargs'] = ['noise_pred']
                                        ref_output = self.adapter.forward(**ref_forward_inputs)
                                    elif self.training_args.kl_type == 'x-based':
                                        # KL in latent space
                                        ref_forward_inputs['return_kwargs'] = ['next_latents_mean']
                                        ref_output = self.adapter.forward(**ref_forward_inputs)

                                # kl_div must be computed outside `torch.no_grad()` for correct gradient behavior.
                                # See: issue #122, PR #123 (https://github.com/X-GenGroup/Flow-Factory/pull/123)
                                if self.training_args.kl_type == 'v-based':
                                    kl_div = torch.mean(
                                        ((output.noise_pred - ref_output.noise_pred) ** 2),
                                        dim=tuple(range(1, output.noise_pred.ndim)), keepdim=True
                                    )
                                elif self.training_args.kl_type == 'x-based':
                                    kl_div = torch.mean(
                                        ((output.next_latents_mean - ref_output.next_latents_mean) ** 2),
                                        dim=tuple(range(1, output.next_latents_mean.ndim)), keepdim=True
                                    )

                                kl_div = torch.mean(kl_div)
                                kl_loss = self.training_args.kl_beta * kl_div
                                loss += kl_loss
                                loss_info['kl_div'].append(kl_div.detach())
                                loss_info['kl_loss'].append(kl_loss.detach())

                        # 5. Log per-timestep info
                        loss_info['ratio'].append(ratio.detach())
                        loss_info['unclipped_loss'].append(unclipped_loss.detach())
                        loss_info['clipped_loss'].append(clipped_loss.detach())
                        loss_info['policy_loss'].append(policy_loss.detach())
                        loss_info['loss'].append(loss.detach())
                        clip_frac_high = torch.mean((ratio > 1.0 + ratio_clip_range[1]).float())
                        clip_frac_low = torch.mean((ratio < 1.0 + ratio_clip_range[0]).float())
                        loss_info["clip_frac_high"].append(clip_frac_high.detach())
                        loss_info["clip_frac_low"].append(clip_frac_low.detach())
                        loss_info['clip_frac_total'].append((clip_frac_high + clip_frac_low).detach())

                        # 6. Backward and optimizer step
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.adapter.get_trainable_parameters(),
                                self.training_args.max_grad_norm,
                            )
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            # Communicate and log losses
                            loss_info = reduce_loss_info(self.accelerator, loss_info)
                            loss_info['grad_norm'] = grad_norm
                            self.log_data(
                                {f'train/{k}': v for k, v in loss_info.items()},
                                step=self.step,
                            )
                            self.step += 1
                            loss_info = defaultdict(list)