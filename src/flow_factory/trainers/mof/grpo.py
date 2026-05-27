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

# src/flow_factory/trainers/mof/grpo.py
"""MoF-GRPO Trainer: GRPO (PPO-clipped ratio) optimization for Mixture-of-Flow."""
from typing import List
from collections import defaultdict
from functools import partial

import torch
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .common import MoFTrainerBase
from ...hparams import MoFGRPOTrainingArguments
from ...samples import BaseSample
from ...utils.base import create_generator
from ...utils.trajectory_collector import compute_trajectory_indices
from ...utils.noise_schedule import compute_transition_sigma
from ...utils.dist import reduce_loss_info
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class MoFGRPOTrainer(MoFTrainerBase):
    """MoF trainer with GRPO (PPO-clipped ratio) optimization.

    Inherits all MoF infrastructure and replaces the optimization algorithm:
    - Samples full trajectories with log_probs (on-policy)
    - Optimizes via PPO-style ratio loss: ratio = exp(new_log_prob - old_log_prob)
    - Gradient flows through: logits → softmax → weights → v_combined → log_prob → ratio → loss

    Register as trainer_type: 'mof-grpo'.
    """

    training_args: MoFGRPOTrainingArguments

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # GRPO computes log-probabilities via σ_trans² in the denominator.
        # Under ODE dynamics (noise_level=0), σ_trans=0 → division by zero → NaN.
        dynamics_type = self.adapter.scheduler.dynamics_type
        if dynamics_type == "ODE":
            raise ValueError(
                f"MoF-GRPO requires stochastic dynamics (CPS, Flow-SDE, or Dance-SDE) "
                f"but scheduler.dynamics_type='{dynamics_type}'. "
                f"ODE has σ_trans=0, causing NaN in log-probability computation. "
                f"Use MoF-NFT for deterministic (ODE) dynamics."
            )

        # Pre-compute trajectory indices and extra callback kwargs for sampling
        self._trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.scheduler.train_timesteps,
            num_inference_steps=self.training_args.num_inference_steps,
        )
        self._extra_call_back_kwargs = []
        if self.training_args.mask_type in ('kl', 'kl_adv'):
            self._extra_call_back_kwargs = ['next_latents_mean', 'std_dev_t', 'dt']

    def _build_sample_kwargs(self, batch):
        """GRPO sampling: full trajectory with log_prob for ratio computation."""
        return {
            **self.training_args,
            'compute_log_prob': True,
            'trajectory_indices': self._trajectory_indices,
            'extra_call_back_kwargs': self._extra_call_back_kwargs,
            **{k: v for k, v in batch.items() if k != '__source__'},
        }

    def optimize(self, samples: List[BaseSample]) -> None:
        """GRPO-style optimization: PPO-clipped ratio loss on lambda logits.

        For each stored trajectory timestep:
        1. Retrieve old log_prob (from sampling)
        2. Recompute combined velocity with CURRENT λ logits (differentiable)
        3. Compute new log_prob via scheduler step
        4. ratio = exp(new_log_prob - old_log_prob)
        5. Clipped PPO loss: max(-adv*ratio, -adv*clip(ratio))
        6. Backward → gradients flow through λ logits
        """
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        clip_range = self.training_args.clip_range
        adv_clip_range = self.training_args.adv_clip_range
        mask_type = self.training_args.mask_type
        kl_threshold = self.training_args.kl_mask_threshold
        add_kl_coefficient = self.training_args.add_kl_coefficient

        for inner_epoch in range(self.training_args.num_inner_epochs):
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]
            loss_info = defaultdict(list)

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
                    set_ids = self._get_sample_set_ids(batch_samples)  # (B,)

                    latents_index_map = batch['latent_index_map']
                    log_probs_index_map = batch['log_prob_index_map']
                    # callback_index_map needed for mask_type='kl_adv' (stored noise_pred)
                    callback_index_map = batch.get('callback_index_map', None)
                    if callback_index_map is not None:
                        callback_index_map = callback_index_map[0]

                    # Iterate through train timesteps
                    for _, timestep_index in enumerate(tqdm(
                        self.adapter.scheduler.train_timesteps,
                        desc=f'Epoch {self.epoch} Timestep',
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    )):
                        with self.accelerator.accumulate(self._mixing_module):
                            # 1. Retrieve stored trajectory data
                            old_log_prob = batch['log_probs'][:, log_probs_index_map[timestep_index]]
                            num_timesteps = batch['timesteps'].shape[1]
                            t = batch['timesteps'][:, timestep_index]
                            t_next = (
                                batch['timesteps'][:, timestep_index + 1]
                                if timestep_index + 1 < num_timesteps
                                else torch.zeros_like(t)
                            )
                            latents = batch['all_latents'][:, latents_index_map[timestep_index]]
                            next_latents = batch['all_latents'][:, latents_index_map[timestep_index + 1]]

                            # 2. Compute teacher velocities (frozen, detached)
                            teacher_velocities = self._compute_teacher_velocities(
                                batch, t, latents
                            )  # (K, B, C, H, W)

                            # 3. Combine with CURRENT weights (differentiable)
                            v_combined = self._compute_combined_velocity(
                                teacher_velocities, t, batch,
                                timestep_index=timestep_index, set_ids=set_ids,
                            )  # (B, C, H, W) — gradient flows through mixing weights

                            # 4. Compute new log_prob via scheduler step
                            return_kwargs = ['log_prob']
                            if mask_type in ('kl', 'kl_adv'):
                                return_kwargs.append('next_latents_mean')

                            sched_out = self.adapter.scheduler.step(
                                noise_pred=v_combined,
                                timestep=t,
                                latents=latents,
                                next_latents=next_latents.detach(),
                                timestep_next=t_next,
                                compute_log_prob=True,
                                noise_level=self.adapter.scheduler.get_noise_level_for_timestep(t),
                                return_dict=True,
                                return_kwargs=return_kwargs,
                            )
                            new_log_prob = sched_out.log_prob

                            # 5. PPO-style clipped loss
                            adv = torch.as_tensor(
                                batch['advantage'], dtype=torch.float32, device=device
                            )
                            adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])

                            ratio = torch.exp(new_log_prob - old_log_prob)

                            unclipped_loss = -adv * ratio

                            if mask_type in ('none', 'clip'):
                                clipped_loss = -adv * torch.clamp(
                                    ratio, 1.0 + clip_range[0], 1.0 + clip_range[1]
                                )
                                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                            elif mask_type in ('kl', 'kl_adv'):
                                # x-based KL: ||next_latents_mean_new - next_latents_mean_old||² / (2σ²)
                                old_next_latents_mean = batch['next_latents_mean'][:, callback_index_map[timestep_index]]
                                # new next_latents_mean comes from scheduler step with current v_combined
                                new_next_latents_mean = sched_out.next_latents_mean

                                if add_kl_coefficient:
                                    old_std_dev_t = batch['std_dev_t'][:, callback_index_map[timestep_index]]
                                    old_dt = batch['dt'][:, callback_index_map[timestep_index]]
                                    sigma_t = compute_transition_sigma(
                                        old_std_dev_t, old_dt, self.adapter.scheduler.dynamics_type
                                    )
                                else:
                                    sigma_t = torch.ones_like(sched_out.std_dev_t, device=device)

                                diff = new_next_latents_mean - old_next_latents_mean
                                kl_new_old = (diff ** 2 / (2 * sigma_t ** 2)).mean(
                                    dim=tuple(range(1, diff.ndim))
                                )

                                kl_mask = (kl_new_old < kl_threshold).detach()

                                if mask_type == 'kl':
                                    policy_loss = (unclipped_loss * kl_mask.float()).mean()
                                else:  # kl_adv
                                    pos_rm = (~kl_mask) & (ratio > 1.0) & (adv > 0)
                                    neg_rm = (~kl_mask) & (ratio < 1.0) & (adv < 0)
                                    keep_mask = (~(pos_rm | neg_rm)).float().detach()
                                    policy_loss = (unclipped_loss * keep_mask).mean()
                            else:
                                clipped_loss = -adv * torch.clamp(
                                    ratio, 1.0 + clip_range[0], 1.0 + clip_range[1]
                                )
                                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                            loss = policy_loss

                            # 6. Logging
                            loss_info['ratio'].append(ratio.detach())
                            loss_info['policy_loss'].append(policy_loss.detach())
                            loss_info['loss'].append(loss.detach())
                            clip_frac_high = torch.mean((ratio > 1.0 + clip_range[1]).float())
                            clip_frac_low = torch.mean((ratio < 1.0 + clip_range[0]).float())
                            loss_info['clip_frac_high'].append(clip_frac_high.detach())
                            loss_info['clip_frac_low'].append(clip_frac_low.detach())
                            if mask_type in ('kl', 'kl_adv'):
                                loss_info['kl_new_old'].append(kl_new_old.mean().detach())
                            if mask_type == 'kl_adv':
                                loss_info['keep_adv_mask_ratio'].append(keep_mask.mean().detach())
                            elif mask_type == 'kl':
                                loss_info['kl_mask_ratio'].append(kl_mask.float().mean().detach())

                            # 7. Backward and optimizer step
                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self._mixing_module.parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                # Reduce and log
                                loss_info_reduced = reduce_loss_info(self.accelerator, loss_info)
                                loss_info_reduced['grad_norm'] = grad_norm
                                if not self._is_router_mode:
                                    with torch.no_grad():
                                        log_weights = self._get_lambda_weights(self._lambda_logits)
                                        mean_weights = log_weights.mean(dim=1)  # (K, S)
                                        for k in range(self.K):
                                            teacher_name = self._teacher_names[k]
                                            for s in range(self.S):
                                                src_name = self._set_id_to_source.get(s, str(s))
                                                loss_info_reduced[f'lambda_{teacher_name}_{src_name}_mean'] = (
                                                    mean_weights[k, s].item()
                                                )
                                self.log_data(
                                    {f'train/{k}': v for k, v in loss_info_reduced.items()},
                                    step=self.step,
                                )
                                self.step += 1
                                loss_info = defaultdict(list)
