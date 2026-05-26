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

# src/flow_factory/trainers/mof/nft.py
"""MoF-NFT Trainer: DiffusionNFT optimization for Mixture-of-Flow."""
from typing import List
from collections import defaultdict
from functools import partial

import torch
from diffusers.utils.torch_utils import randn_tensor
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .common import MoFTrainerBase
from ...hparams import MoFNFTTrainingArguments
from ...samples import BaseSample
from ...utils.base import (
    filter_kwargs,
    create_generator,
    to_broadcast_tensor,
    stitch_batch_metadata,
)
from ...utils.noise_schedule import flow_match_sigma
from ...utils.dist import reduce_loss_info


class MoFNFTTrainer(MoFTrainerBase):
    """MoF trainer with DiffusionNFT optimization.

    Uses NFT's self-normalized MSE loss on x₀ reconstruction, weighted by
    advantages. The combined velocity v = Σ λ_k·v_k is evaluated against
    positive/negative predictions interpolated by β.

    Register as trainer_type: 'mof-nft'.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: MoFNFTTrainingArguments
        self.nft_beta = self.training_args.nft_beta

    @staticmethod
    def _nft_weighted_mse(
        v_pred: torch.Tensor,
        noised_latents: torch.Tensor,
        sigma_broadcast: torch.Tensor,
        clean_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NFT's self-normalized MSE loss (per-sample scalar).

        Computes ||x0_pred - x0||^2 / weight, where weight is the detached
        mean absolute error (prevents gradient through normalization).

        Returns:
            Per-sample loss of shape (B,).
        """
        x0_pred = noised_latents - sigma_broadcast * v_pred
        with torch.no_grad():
            weight = (
                torch.abs(x0_pred.double() - clean_latents.double())
                .mean(dim=tuple(range(1, clean_latents.ndim)), keepdim=True)
                .clip(min=1e-5)
            )
        return (
            (x0_pred - clean_latents) ** 2 / weight
        ).mean(dim=tuple(range(1, clean_latents.ndim)))

    def sample(self) -> List[BaseSample]:
        """Generate rollouts using per-set lambda-combined teacher velocity.

        NFT mode: compute_log_prob=False, stores only final clean latents.
        """
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []

        if self.train_dataloaders_by_source:
            data_iter = self._interleaved_source_iter()
        else:
            data_iter = iter(self.dataloader)

        with torch.no_grad(), self.autocast():
            for _ in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                set_id = self._get_batch_set_id(batch)

                with self._mof_inference_context(set_id):
                    sample_kwargs = {
                        **self.training_args,
                        'compute_log_prob': False,
                        'trajectory_indices': [-1],
                        **{k: v for k, v in batch.items() if k != '__source__'},
                    }
                    sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                    sample_batch = self.adapter.inference(**sample_kwargs)

                stitch_batch_metadata(batch, sample_batch)
                self._maybe_offload_samples_to_cpu(sample_batch)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)

        return samples

    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization: NFT loss on per-set lambda-weighted teacher velocities.

        For each micro-batch:
        1. Extract per-sample set_ids from __source__ metadata.
        2. Precompute old_v_pred using EMA logits (per-sample set-specific).
        3. Recompute teacher velocities and combine with current logits.
        4. Accumulate gradients over T timesteps, single optimizer step.
        """
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size

        for inner_epoch in range(self.training_args.num_inner_epochs):
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

                # Per-sample set IDs
                set_ids = self._get_sample_set_ids(batch_samples)  # (B,)

                # ---- Phase 1: Precompute noised_latents + old_v_pred ----
                self.adapter.rollout()
                with torch.no_grad(), self.autocast():
                    all_timesteps = self._sample_timesteps(batch_size)
                    all_noised_latents: List[torch.Tensor] = []
                    all_sigma_broadcast: List[torch.Tensor] = []
                    all_teacher_velocities: List[torch.Tensor] = []
                    old_v_pred_list: List[torch.Tensor] = []

                    with self.sampling_context():
                        ema_weights = self._get_lambda_weights(self._lambda_logits)

                    for t_idx in range(self.num_train_timesteps):
                        t_flat = all_timesteps[t_idx]
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

                        teacher_velocities = self._compute_teacher_velocities(
                            batch, t_flat, noised_latents
                        )
                        all_teacher_velocities.append(teacher_velocities)

                        old_v = self._combine_velocities_per_sample(
                            teacher_velocities, t_idx, ema_weights, set_ids
                        )
                        old_v_pred_list.append(old_v.detach())

                # ---- Phase 2: Train with current logits ----
                self.adapter.train()
                with self.autocast():
                    for t_idx in tqdm(
                        range(self.num_train_timesteps),
                        desc=f'Epoch {self.epoch} Timestep',
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    ):
                        with self.accelerator.accumulate(self._mixing_module):
                            t_flat = all_timesteps[t_idx]
                            sigma_broadcast = all_sigma_broadcast[t_idx]
                            noised_latents = all_noised_latents[t_idx]
                            old_v_pred = old_v_pred_list[t_idx]
                            teacher_velocities = all_teacher_velocities[t_idx]

                            current_weights = self._get_lambda_weights(self._lambda_logits)
                            new_v_pred = self._combine_velocities_per_sample(
                                teacher_velocities, t_idx, current_weights, set_ids
                            )

                            # NFT loss computation
                            adv = torch.as_tensor(
                                batch['advantage'], dtype=torch.float32, device=device
                            )
                            adv_clip_range = self.training_args.adv_clip_range
                            adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])

                            normalized_adv = (adv / max(adv_clip_range)) / 2.0 + 0.5
                            r = torch.clamp(normalized_adv, 0, 1).view(
                                -1, *([1] * (new_v_pred.dim() - 1))
                            )

                            positive_pred = (
                                self.nft_beta * new_v_pred
                                + (1 - self.nft_beta) * old_v_pred
                            )
                            negative_pred = (
                                (1.0 + self.nft_beta) * old_v_pred
                                - self.nft_beta * new_v_pred
                            )

                            positive_loss = self._nft_weighted_mse(
                                positive_pred, noised_latents, sigma_broadcast, clean_latents
                            )
                            negative_loss = self._nft_weighted_mse(
                                negative_pred, noised_latents, sigma_broadcast, clean_latents
                            )

                            ori_policy_loss = (
                                r.squeeze() * positive_loss
                                + (1.0 - r.squeeze()) * negative_loss
                            ) / self.nft_beta
                            policy_loss = (ori_policy_loss * adv_clip_range[1]).mean()
                            loss = policy_loss

                            # Optional KL penalty
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

                            self.accelerator.backward(loss)

                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self._mixing_module.parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()

                                loss_info_reduced = reduce_loss_info(self.accelerator, loss_info)
                                loss_info_reduced['grad_norm'] = grad_norm
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
