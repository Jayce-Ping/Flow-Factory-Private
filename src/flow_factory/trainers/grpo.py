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
from ..rewards import RewardBuffer
from ..samples import BaseSample
from ..utils.base import filter_kwargs, create_generator, create_generator_by_prompt, stitch_batch_metadata
from ..utils.logger_utils import setup_logger
from ..utils.noise_schedule import compute_transition_sigma
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
        # Cache mask-related attributes for DPPO
        self.mask_type = getattr(self.training_args, 'mask_type', 'none')
        self.kl_mask_threshold = float(getattr(self.training_args, 'kl_mask_threshold', 1e-5))
        self.add_kl_coefficient = getattr(self.training_args, 'add_kl_coefficient', True)

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
        """Generate rollouts for GRPO."""
        self.adapter.rollout()
        self.reward_buffer.clear() # Clear reward buffer
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
                # Determine extra callback kwargs based on mask_type
                extra_call_back_kwargs = []
                if self.mask_type in ('kl', 'kl_adv'):
                    if self.training_args.kl_type == 'v-based':
                        extra_call_back_kwargs = ['noise_pred']
                    else:  # x-based
                        extra_call_back_kwargs = ['next_latents_mean', 'std_dev_t', 'dt']

                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': True,
                    'trajectory_indices': trajectory_indices, # Selectively store required trajectory positions for memory efficiency
                    'extra_call_back_kwargs': extra_call_back_kwargs,
                    **batch,
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                # Stitch dataset metadata (e.g. include, tag, __source__) onto
                # generated samples for downstream reward routing.
                stitch_batch_metadata(batch, sample_batch)
                # Deterministic D2H so reward_buffer sees CPU-resident samples
                # (no-op when offload_samples_to_cpu is False).
                self._maybe_offload_samples_to_cpu(sample_batch)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)

        return samples

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
        """Policy optimization (Stage 6): PPO-style clipped loss and optional KL."""
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle samples at the beginning of each inner epoch
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]

            self.adapter.train()
            loss_info = defaultdict(list)

            # Lazy per-batch reload: only the current micro-batch lives on GPU.
            # When samples are GPU-resident `sample.to(device)` is a no-op; when
            # they are CPU-resident (offload pipeline) this is the H2D point.
            with self.autocast():
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
                    latents_index_map = batch['latent_index_map']  # (T+1,) LongTensor
                    log_probs_index_map = batch['log_prob_index_map']  # (T,) LongTensor
                    # callback_index_map is needed for mask_type in ('kl', 'kl_adv')
                    callback_index_map = batch.get('callback_index_map', None)
                    if callback_index_map is not None:
                        callback_index_map = callback_index_map[0]  # (T,) LongTensor, shared across batch
                    # Iterate through timesteps
                    for idx, timestep_index in enumerate(tqdm(
                        self.adapter.scheduler.train_timesteps,
                        desc=f'Epoch {self.epoch} Timestep',
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    )):
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
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
                            if self.enable_kl_loss:
                                if self.training_args.kl_type == 'v-based':
                                    return_kwargs = ['log_prob', 'noise_pred', 'dt']
                                elif self.training_args.kl_type == 'x-based':
                                    return_kwargs = ['log_prob', 'next_latents', 'next_latents_mean', 'dt']
                            else:
                                return_kwargs = ['log_prob', 'dt']

                            # Extend return_kwargs for mask_type KL computation
                            if self.mask_type in ('kl', 'kl_adv'):
                                if self.training_args.kl_type == 'v-based':
                                    if 'noise_pred' not in return_kwargs:
                                        return_kwargs.append('noise_pred')
                                else:  # x-based
                                    for k in ['next_latents_mean', 'std_dev_t', 'dt']:
                                        if k not in return_kwargs:
                                            return_kwargs.append(k)

                            forward_inputs['return_kwargs'] = return_kwargs
                            output = self.adapter.forward(**forward_inputs)

                            # 3. Compute loss
                            # Clip advantages
                            adv = batch['advantage']
                            adv_clip_range = self.training_args.adv_clip_range
                            adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                            # PPO-style ratio
                            ratio = torch.exp(output.log_prob - old_log_prob)
                            ratio_clip_range = self.training_args.clip_range

                            unclipped_loss = -adv * ratio

                            # Apply mask_type-dependent loss
                            if self.mask_type == 'none':
                                # Standard GRPO: PPO clipping
                                clipped_loss = -adv * torch.clamp(ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1])
                                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                            elif self.mask_type == 'clip':
                                # Explicit PPO clipping (alias for standard behavior)
                                clipped_loss = -adv * torch.clamp(ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1])
                                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                            elif self.mask_type in ('kl', 'kl_adv'):
                                # Compute new-vs-old KL divergence for masking
                                if self.training_args.kl_type == 'v-based':
                                    old_noise_pred = batch['noise_pred'][:, callback_index_map[timestep_index]]
                                    sq = (output.noise_pred - old_noise_pred) ** 2
                                    kl_new_old_per_sample = sq.mean(dim=tuple(range(1, sq.ndim)))
                                else:  # x-based
                                    old_next_latents_mean = batch['next_latents_mean'][:, callback_index_map[timestep_index]]
                                    if self.add_kl_coefficient:
                                        old_std_dev_t = batch['std_dev_t'][:, callback_index_map[timestep_index]]
                                        old_dt = batch['dt'][:, callback_index_map[timestep_index]]
                                        sigma_t = compute_transition_sigma(
                                            old_std_dev_t, old_dt, self.adapter.scheduler.dynamics_type
                                        )
                                    else:
                                        sigma_t = torch.ones_like(output.dt)
                                    diff = output.next_latents_mean - old_next_latents_mean
                                    kl_new_old_per_sample = (diff ** 2 / (2 * sigma_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) ** 2)).mean(
                                        dim=tuple(range(1, diff.ndim))
                                    )

                                # Compute KL mask: keep samples where KL < threshold
                                kl_mask = (kl_new_old_per_sample < self.kl_mask_threshold).detach()

                                if self.mask_type == 'kl':
                                    # Pure KL mask: zero out loss for high-KL samples
                                    policy_loss = (unclipped_loss * kl_mask.float()).mean()
                                else:  # kl_adv
                                    # DPPO: KL-advantage masking
                                    # Remove samples where policy diverged too much AND
                                    # the ratio/advantage combination is "dangerous"
                                    pos_rm_mask = (~kl_mask) & (ratio > 1.0) & (adv > 0)
                                    neg_rm_mask = (~kl_mask) & (ratio < 1.0) & (adv < 0)
                                    rm_mask = pos_rm_mask | neg_rm_mask
                                    keep_adv_mask = (~rm_mask).float().detach()
                                    policy_loss = (unclipped_loss * keep_adv_mask).mean()

                            loss = policy_loss

                            # 4. Compute KL-div
                            if self.enable_kl_loss:
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
                            loss_info['policy_loss'].append(policy_loss.detach())
                            loss_info['loss'].append(loss.detach())
                            clip_frac_high = torch.mean((ratio > 1.0 + ratio_clip_range[1]).float())
                            clip_frac_low = torch.mean((ratio < 1.0 + ratio_clip_range[0]).float())
                            loss_info["clip_frac_high"].append(clip_frac_high.detach())
                            loss_info["clip_frac_low"].append(clip_frac_low.detach())
                            loss_info['clip_frac_total'].append((clip_frac_high + clip_frac_low).detach())

                            # Mask-specific logging
                            if self.mask_type in ('kl', 'kl_adv'):
                                loss_info['kl_new_old'].append(kl_new_old_per_sample.mean().detach())
                            if self.mask_type == 'kl_adv':
                                loss_info['keep_adv_mask_ratio'].append(keep_adv_mask.mean().detach())
                            elif self.mask_type == 'kl':
                                loss_info['kl_mask_ratio'].append(kl_mask.float().mean().detach())

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

    # =========================== Advantage Computation ============================
    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func: Optional[Union[Literal['sum', 'gdpo', 'smart_grpo'], Callable]] = None,
    ) -> torch.Tensor:
        """Compute advantages — delegates to AdvantageProcessor.

        Args:
            samples: List of BaseSample instances
            rewards: Dict of reward_name to reward tensors aligned with samples
            store_to_samples: Whether to store computed advantages back to samples' extra_kwargs
            aggregation_func: Method to aggregate advantages within each group.
                Options: 'sum' (default GRPO), 'gdpo' (GDPO-style),
                         'smart_grpo' (Softmax-Centered, Group-Mean Temperature),
                         or a custom callable.
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


    def compute_advantages_smart_grpo(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True
    ) -> torch.Tensor:
        """
        Compute advantages using SMART-GRPO: Softmax-Centered, Normalized Group-Mean Temperature.

        For each group, the advantage is:
            A_i = K * softmax((r_i - mu_G) / tau) - 1
        where:
            mu_G = group mean reward
            tau  = |mu_G| / mean(|mu|)  (normalized group-mean temperature)
            K    = group size

        The temperature tau is the group's |mu_G| divided by the batch-wide
        average |mu| across all groups. This preserves two key properties:
            1. Harder groups (lower |mu_G|) get tau < 1 -> sharper softmax
               -> concentrates advantage on the few good samples
            2. Easier groups (higher |mu_G|) get tau > 1 -> flatter softmax
               -> distributes advantage more evenly
        while normalizing the temperature to ~1 scale regardless of the
        absolute reward magnitude (e.g. PickScore ~20 vs [0,1] rewards).

        Other properties:
            - Sum = 0 per group (compatible with PG frameworks)
            - Bounded in [-1, K - 1] (K-scaled, no hyperparameters)
            - Below-average samples get NEGATIVE advantage

        References:
            SMART-GRPO notebook (SMART-GRPO.ipynb)
        """
        # 1. Gather rewards across processes
        rewards = {key: torch.as_tensor(value).to(self.accelerator.device) for key, value in rewards.items()}
        gathered_rewards = {
            key: self.accelerator.gather(value).cpu().numpy()
            for key, value in rewards.items()
        }

        # 2. Aggregate rewards if multiple reward models (weighted sum)
        aggregated_rewards = np.zeros_like(next(iter(gathered_rewards.values())), dtype=np.float64)
        for key, reward_array in gathered_rewards.items():
            aggregated_rewards += reward_array * self.reward_models[key].config.weight

        # 3. Group rewards by unique_ids
        unique_ids = torch.tensor([s.unique_id for s in samples], dtype=torch.int64, device=self.accelerator.device)
        gathered_ids = self.accelerator.gather(unique_ids).cpu().numpy()
        _unique_ids, group_indices, _counts = np.unique(gathered_ids, return_inverse=True, return_counts=True)

        # 4. Compute SMART-GRPO advantages: softmax-centered with normalized group-mean temperature
        from scipy.special import softmax as scipy_softmax
        advantages = np.zeros_like(aggregated_rewards, dtype=np.float64)
        eps = 1e-8

        # Pre-compute per-group |mu_G| to derive the normalizer mean(|mu|)
        unique_group_ids = np.unique(group_indices)
        abs_mus = np.array([
            np.abs(np.mean(aggregated_rewards[group_indices == gid]))
            for gid in unique_group_ids
        ])
        mean_abs_mu = max(np.mean(abs_mus), eps)

        for group_id in unique_group_ids:
            mask = (group_indices == group_id)
            group_rewards = aggregated_rewards[mask]
            K = len(group_rewards)
            assert K == self.training_args.group_size, \
                f"Group size mismatch: expected {self.training_args.group_size}, got {K}"

            mu = np.mean(group_rewards, axis=0, keepdims=True)
            tau = np.clip(np.abs(mu) / mean_abs_mu, eps, None)
            sm = scipy_softmax((group_rewards - mu) / tau)
            advantages[mask] = K * sm - 1.0

        # 5. Log statistics
        _log_data = {
            f'train/reward_{key}_mean': np.mean(value)
            for key, value in gathered_rewards.items()
        }
        _log_data.update({
            f'train/reward_{key}_std': np.std(value)
            for key, value in gathered_rewards.items()
        })
        for key, reward_array in gathered_rewards.items():
            g_means, g_stds = RewardProcessor.compute_group_reward_stats(reward_array, group_indices)
            _log_data.update({
                f'train/reward_{key}_group_std_mean':  float(np.mean(g_stds)),
                f'train/reward_{key}_group_std_max':   float(np.max(g_stds)),
                f'train/reward_{key}_group_std_min':   float(np.min(g_stds)),
                f'train/reward_{key}_group_mean_std':  float(np.std(g_means)),
            })
        zero_std_ratio = RewardProcessor.compute_group_zero_std_ratio(aggregated_rewards, group_indices)
        _log_data['train/reward_zero_std_ratio'] = zero_std_ratio
        _log_data.update({
            'train/reward_mean': np.mean(aggregated_rewards),
            'train/reward_std': np.std(aggregated_rewards),
        })
        g_means, g_stds = RewardProcessor.compute_group_reward_stats(aggregated_rewards, group_indices)
        _log_data.update({
            'train/reward_group_std_mean': float(np.mean(g_stds)),
            'train/reward_group_std_max':  float(np.max(g_stds)),
            'train/reward_group_mean_std': float(np.std(g_means)),
        })
        _log_data.update({
            'train/adv_max': np.max(advantages),
            'train/adv_min': np.min(advantages),
            'train/adv_abs_mean': np.mean(np.abs(advantages)),
        })
        _log_data['train_samples'] = samples[:30]

        self.log_data(_log_data, step=self.step)

        # 6. Scatter advantages back to align with samples
        advantages = torch.as_tensor(advantages).reshape(
            self.accelerator.num_processes, -1, *advantages.shape[1:]
        )[self.accelerator.process_index].to(self.accelerator.device)

        if store_to_samples:
            for sample, adv in zip(samples, advantages):
                sample.extra_kwargs['advantage'] = adv

        return advantages


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
                # GRPO-Guard always needs 'next_latents_mean' for ratio reweighting.
                # For mask_type with v-based KL, also store 'noise_pred'.
                # For mask_type with x-based KL, also store 'std_dev_t' and 'dt'.
                extra_call_back_kwargs = ['next_latents_mean']
                if self.mask_type in ('kl', 'kl_adv'):
                    if self.training_args.kl_type == 'v-based':
                        extra_call_back_kwargs.append('noise_pred')
                    else:  # x-based: next_latents_mean already included
                        for k in ['std_dev_t', 'dt']:
                            if k not in extra_call_back_kwargs:
                                extra_call_back_kwargs.append(k)

                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': True,
                    'trajectory_indices': trajectory_indices, # Selectively store required trajectory positions for memory efficiency
                    'extra_call_back_kwargs': extra_call_back_kwargs,
                    **batch,
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                # Stitch dataset metadata (e.g. include, tag, __source__) onto
                # generated samples for downstream reward routing.
                stitch_batch_metadata(batch, sample_batch)
                # Deterministic D2H so reward_buffer sees CPU-resident samples
                # (no-op when offload_samples_to_cpu is False).
                self._maybe_offload_samples_to_cpu(sample_batch)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)

        return samples

    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization (Stage 6): GRPO-Guard reweighted loss and optional KL."""
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        for inner_epoch in range(self.training_args.num_inner_epochs):
            # Shuffle samples at the beginning of each inner epoch
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]

            self.adapter.train()
            loss_info = defaultdict(list)

            # Lazy per-batch reload: only the current micro-batch lives on GPU.
            with self.autocast():
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
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
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
                            # Extend for mask_type KL computation
                            if self.mask_type in ('kl', 'kl_adv'):
                                if self.training_args.kl_type == 'v-based':
                                    return_kwargs.add('noise_pred')

                            forward_inputs['return_kwargs'] = list(return_kwargs)
                            output = self.adapter.forward(**forward_inputs)

                            # 3. Compute loss
                            # Clip advantages
                            adv = batch['advantage']
                            adv_clip_range = self.training_args.adv_clip_range
                            adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])
                            # Reweighted ratio (GRPO-Guard dynamics-aware ratio)
                            scale_factor = torch.sqrt(-output.dt) * output.std_dev_t
                            old_next_latents_mean = batch['next_latents_mean'][:, callback_index_map[timestep_index]]
                            mse = (output.next_latents_mean - old_next_latents_mean).flatten(1).pow(2).mean(dim=1)
                            ratio = torch.exp((output.log_prob - old_log_prob) * scale_factor + mse / (2 * scale_factor))
                            ratio_clip_range = self.training_args.clip_range

                            unclipped_loss = -adv * ratio

                            # Apply mask_type-dependent loss
                            if self.mask_type == 'none':
                                clipped_loss = -adv * torch.clamp(ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1])
                                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                            elif self.mask_type == 'clip':
                                clipped_loss = -adv * torch.clamp(ratio, 1.0 + ratio_clip_range[0], 1.0 + ratio_clip_range[1])
                                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                            elif self.mask_type in ('kl', 'kl_adv'):
                                # Compute new-vs-old KL divergence for masking
                                if self.training_args.kl_type == 'v-based':
                                    old_noise_pred = batch['noise_pred'][:, callback_index_map[timestep_index]]
                                    sq = (output.noise_pred - old_noise_pred) ** 2
                                    kl_new_old_per_sample = sq.mean(dim=tuple(range(1, sq.ndim)))
                                else:  # x-based: use already-retrieved old_next_latents_mean
                                    if self.add_kl_coefficient:
                                        old_std_dev_t = batch['std_dev_t'][:, callback_index_map[timestep_index]]
                                        old_dt = batch['dt'][:, callback_index_map[timestep_index]]
                                        sigma_t = compute_transition_sigma(
                                            old_std_dev_t, old_dt, self.adapter.scheduler.dynamics_type
                                        )
                                    else:
                                        sigma_t = torch.ones_like(output.dt)
                                    diff = output.next_latents_mean - old_next_latents_mean
                                    kl_new_old_per_sample = (diff ** 2 / (2 * sigma_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) ** 2)).mean(
                                        dim=tuple(range(1, diff.ndim))
                                    )

                                kl_mask = (kl_new_old_per_sample < self.kl_mask_threshold).detach()

                                if self.mask_type == 'kl':
                                    policy_loss = (unclipped_loss * kl_mask.float()).mean()
                                else:  # kl_adv
                                    pos_rm_mask = (~kl_mask) & (ratio > 1.0) & (adv > 0)
                                    neg_rm_mask = (~kl_mask) & (ratio < 1.0) & (adv < 0)
                                    rm_mask = pos_rm_mask | neg_rm_mask
                                    keep_adv_mask = (~rm_mask).float().detach()
                                    policy_loss = (unclipped_loss * keep_adv_mask).mean()

                            loss = policy_loss

                            # 4. Compute KL-div
                            if self.enable_kl_loss:
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
                            loss_info['policy_loss'].append(policy_loss.detach())
                            loss_info['loss'].append(loss.detach())
                            clip_frac_high = torch.mean((ratio > 1.0 + ratio_clip_range[1]).float())
                            clip_frac_low = torch.mean((ratio < 1.0 + ratio_clip_range[0]).float())
                            loss_info["clip_frac_high"].append(clip_frac_high.detach())
                            loss_info["clip_frac_low"].append(clip_frac_low.detach())
                            loss_info['clip_frac_total'].append((clip_frac_high + clip_frac_low).detach())

                            # Mask-specific logging
                            if self.mask_type in ('kl', 'kl_adv'):
                                loss_info['kl_new_old'].append(kl_new_old_per_sample.mean().detach())
                            if self.mask_type == 'kl_adv':
                                loss_info['keep_adv_mask_ratio'].append(keep_adv_mask.mean().detach())
                            elif self.mask_type == 'kl':
                                loss_info['kl_mask_ratio'].append(kl_mask.float().mean().detach())

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