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

# src/flow_factory/trainers/mof/dmin.py
"""MoF D-minimization trainer: reward-free off-manifold-drift loss.

Optimizes the mixing weights by minimizing the teacher-disagreement variance

    D(x_t, t) = sum_k w_k * ||v^(k)(x_t,t) - v_lambda(x_t,t)||^2 ,

where v_lambda = sum_k w_k v^(k) is the current mixture. No reward or
advantage enters the training loss; the three rewards (geneval/pickscore/ocr)
are computed only at evaluation (inherited from MoFTrainerBase) to monitor
training.

This is a validation trainer: because D = 0 at every single-teacher vertex,
minimizing D drives the mixture toward a single teacher (collapse). Watching
the eval rewards reveals that collapse empirically -- it demonstrates that
off-manifold avoidance is a guardrail, not a standalone objective.

Register as trainer_type: 'mof-dmin'.
"""
from typing import Any, Dict, List
from collections import defaultdict
from functools import partial

import torch
from diffusers.utils.torch_utils import randn_tensor
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .common import MoFTrainerBase
from ...hparams import MoFDMinTrainingArguments
from ...samples import BaseSample
from ...utils.base import create_generator, to_broadcast_tensor
from ...utils.noise_schedule import flow_match_sigma
from ...utils.dist import reduce_loss_info


class MoFDMinTrainer(MoFTrainerBase):
    """MoF trainer with reward-free D-minimization.

    Reuses all MoFTrainerBase infrastructure (teacher loading, mixing module,
    on-policy mixture sampling, eval-time reward monitoring, checkpointing).
    Only the optimization objective differs: minimize the teacher-disagreement
    variance D instead of a reward-weighted loss.

    Register as trainer_type: 'mof-dmin'.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: MoFDMinTrainingArguments

    def _build_sample_kwargs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Sampling kwargs: store only the final latent (re-noised in optimize)."""
        return {
            **self.training_args,
            'compute_log_prob': False,
            'trajectory_indices': [-1],
            **{k: v for k, v in batch.items() if k != '__source__'},
        }

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Reward-free: skip the reward/advantage pipeline.

        D-minimization needs no rewards during training (the three rewards are
        computed at eval instead). We only log a few rollout images so wandb
        still shows qualitative training progress.
        """
        if self.accelerator.is_main_process:
            self.log_data({"train_samples": samples[:30]}, step=self.step)

    def optimize(self, samples: List[BaseSample]) -> None:
        """Minimize D = sum_k w_k ||v_k - v_lambda||^2 over the mixing weights.

        Two-pass per micro-batch (mirrors the NFT trainer to respect the
        autocast-cache / DDP weight-swap invariant in CLAUDE.md):
          Phase 1 (no_grad): re-noise the final latent per timestep and cache
                             the K teacher velocities (weight swaps happen here).
          Phase 2 (grad):    combine cached teacher velocities with the current
                             mixing weights and minimize D.
        """
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size

        for inner_epoch in range(self.training_args.num_inner_epochs):
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]

            loss_info: Dict[str, List[torch.Tensor]] = defaultdict(list)

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

                # Per-sample set IDs (for LUT source routing)
                set_ids = self._get_sample_set_ids(batch_samples)  # (B,)

                # ---- Phase 1: precompute noised_latents + teacher velocities ----
                self.adapter.rollout()
                with torch.no_grad(), self.autocast():
                    all_timesteps = self._sample_timesteps(batch_size)
                    all_noised_latents: List[torch.Tensor] = []
                    all_teacher_velocities: List[torch.Tensor] = []

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

                        teacher_velocities = self._compute_teacher_velocities(
                            batch, t_flat, noised_latents
                        )  # (K, B, *latent_dims), detached
                        all_teacher_velocities.append(teacher_velocities)

                # ---- Phase 2: minimize D with current mixing weights ----
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
                            teacher_velocities = all_teacher_velocities[t_idx]

                            v_lambda, w_kb = self._compute_combined_velocity(
                                teacher_velocities, t_flat, batch,
                                timestep_index=t_idx, set_ids=set_ids,
                                return_weights=True,
                            )  # v_lambda: (B, *), w_kb: (K, B)

                            # D = sum_k w_k * mean_spatial||v_k - v_lambda||^2
                            spatial_dims = tuple(range(2, teacher_velocities.ndim))
                            sq = (
                                (teacher_velocities.float()
                                 - v_lambda.float().unsqueeze(0)) ** 2
                            ).mean(dim=spatial_dims)  # (K, B)
                            d_per_sample = (w_kb.float() * sq).sum(dim=0)  # (B,)
                            loss = d_per_sample.mean()

                            # Soft Sigma w ~= 1 anchor (weight_normalization='none' only)
                            if self._sum_penalty_active:
                                sum_penalty = (
                                    self.training_args.weight_sum_penalty
                                    * self._weight_sum_penalty_value(w_kb)
                                )
                                loss = loss + sum_penalty
                                loss_info['weight_sum_penalty'].append(sum_penalty.detach())

                            self._append_weight_stats(loss_info, w_kb)
                            loss_info['D'].append(d_per_sample.mean().detach())
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
                                if not self._is_router_mode:
                                    with torch.no_grad():
                                        log_weights = self._get_lambda_weights(self._lambda_logits)
                                        mean_weights = log_weights.mean(dim=1)  # (K, S)
                                        for k in range(self.K):
                                            teacher_name = self._teacher_names[k]
                                            for s in range(self.S):
                                                src_name = self._set_id_to_source.get(s, str(s))
                                                loss_info_reduced[
                                                    f'lambda_{teacher_name}_{src_name}_mean'
                                                ] = mean_weights[k, s].item()
                                self.log_data(
                                    {f'train/{k}': v for k, v in loss_info_reduced.items()},
                                    step=self.step,
                                )
                                self.step += 1
                                loss_info = defaultdict(list)
