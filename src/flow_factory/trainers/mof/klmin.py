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

# src/flow_factory/trainers/mof/klmin.py
"""MoF KL-to-base minimization trainer: reward-free distribution-shift loss.

Optimizes the convex mixing weights by minimizing the per-step KL between the
teacher mixture and the pretrained base model. Under flow matching this KL
equals velocity-MSE (docs/opd/kl_weighted_teacher_fusion.tex, Prop. 1), so the
loss is

    L(x_t, t) = mean_b || v_lambda(x_t,t) - v_base(x_t,t) ||^2 ,

where v_lambda = sum_k w_k v^(k) is the current mixture and v_base is the
LoRA-disabled base forward. Because the softmax weights satisfy sum_k w_k = 1,
this equals || sum_k w_k (v^(k) - v_base) ||^2 -- the squared norm of the
convex combination of teacher task vectors tau_k = v^(k) - v_base. No reward or
advantage enters the training loss; the three rewards (geneval/pickscore/ocr)
are computed only at evaluation (inherited from MoFTrainerBase) to monitor the
closeness-to-base vs multi-teacher trade-off.

Caveat: minimizing distribution shift alone pulls the mixture toward the
least-drifted teacher / the base (the alpha<0 inverse-variance regime of the
theory doc, which suppresses the active specialist). Two opt-in regularizers
(default off) counter the collapse and keep multiple teachers active:
  - klmin_entropy_coeff: entropy bonus -coeff * H(w) (maximize weight spread).
  - klmin_uniform_anchor_coeff: coeff * ||w - 1/K||^2 (pull toward uniform).

Register as trainer_type: 'mof-klmin'.
"""
from typing import Any, Dict, List
from collections import defaultdict
from functools import partial

import torch
from diffusers.utils.torch_utils import randn_tensor
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .common import MoFTrainerBase
from ...hparams import MoFKLMinTrainingArguments
from ...samples import BaseSample
from ...utils.base import create_generator, to_broadcast_tensor
from ...utils.noise_schedule import flow_match_sigma
from ...utils.dist import reduce_loss_info


class MoFKLMinTrainer(MoFTrainerBase):
    """MoF trainer with reward-free KL-to-base minimization.

    Reuses all MoFTrainerBase infrastructure (teacher loading, mixing module,
    on-policy mixture sampling, eval-time reward monitoring, checkpointing).
    Only the optimization objective differs: minimize the velocity-MSE between
    the teacher mixture and the base model instead of a reward-weighted loss.

    Register as trainer_type: 'mof-klmin'.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: MoFKLMinTrainingArguments

    # =========================================================================
    # Differentiable loss / regularizer math (staticmethods for CPU testing)
    # =========================================================================

    @staticmethod
    def _klmin_loss(v_lambda: torch.Tensor, v_base: torch.Tensor) -> torch.Tensor:
        """Per-sample velocity-MSE to base: mean_spatial ||v_lambda - v_base||^2.

        Args:
            v_lambda: (B, *latent_dims) mixture velocity (differentiable).
            v_base: (B, *latent_dims) base velocity (detached).

        Returns:
            Per-sample loss of shape (B,), computed in float32.
        """
        if v_lambda.shape != v_base.shape:
            raise ValueError(
                "v_lambda and v_base must share shape, got "
                f"{tuple(v_lambda.shape)} vs {tuple(v_base.shape)}."
            )
        spatial_dims = tuple(range(1, v_lambda.ndim))
        return ((v_lambda.float() - v_base.float()) ** 2).mean(dim=spatial_dims)

    @staticmethod
    def _entropy_bonus(w_kb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Mean per-sample Shannon entropy H(w) = -sum_k w_k log w_k (scalar).

        Maximized at uniform weights (log K). Requires the columns of ``w_kb``
        to be distributions over the K teachers (non-negative, sum to 1), i.e.
        weight_normalization='softmax' -- negative weights make H(w) ill-defined.
        MoFKLMinTrainingArguments.__post_init__ enforces this whenever the
        entropy bonus is enabled. Computed in float32.

        Args:
            w_kb: (K, B) per-sample mixing weights.
        """
        p = w_kb.float()
        h = -(p * p.clamp_min(eps).log()).sum(dim=0)  # (B,)
        return h.mean()

    @staticmethod
    def _uniform_anchor(w_kb: torch.Tensor, num_teachers: int) -> torch.Tensor:
        """Mean per-sample squared distance to uniform: ||w - 1/K||^2 (scalar).

        Zero iff every column equals the uniform vector 1/K. Valid for any
        weight_normalization mode. Computed in float32.

        Args:
            w_kb: (K, B) per-sample mixing weights.
            num_teachers: K, the number of teachers.
        """
        return ((w_kb.float() - 1.0 / num_teachers) ** 2).sum(dim=0).mean()

    # =========================================================================
    # Base velocity (LoRA disabled)
    # =========================================================================

    def _compute_base_velocity(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Forward the base model (LoRA disabled) and return the detached velocity.

        Uses ``adapter.use_ref_parameters`` -- under LoRA finetuning a CLAUDE.md
        "safe path" that toggles ``PeftModel.disable_adapter()`` without
        overwriting weight data, so (unlike the teacher weight swaps in
        ``_compute_teacher_velocities``) it needs no autocast-cache / DDP-bypass
        handling. Mirrors the NFT KL-penalty reference forward (``mof/nft.py``).

        Returns:
            Tensor of shape (B, *latent_dims), detached (no gradient to base).
        """
        forward_kwargs = self._build_forward_kwargs(batch, timestep, noised_latents)
        with torch.no_grad(), self.adapter.use_ref_parameters():
            output = self.adapter.forward(**forward_kwargs)
        return output.noise_pred.detach()

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

        KL-to-base minimization needs no rewards during training (the three
        rewards are computed at eval instead). We only log a few rollout images
        so wandb still shows qualitative training progress.
        """
        if self.accelerator.is_main_process:
            self.log_data({"train_samples": samples[:30]}, step=self.step)

    def optimize(self, samples: List[BaseSample]) -> None:
        """Minimize L = mean_b ||v_lambda - v_base||^2 over the mixing weights.

        Two-pass per micro-batch (mirrors the NFT/D-min trainers to respect the
        autocast-cache / DDP weight-swap invariant in CLAUDE.md):
          Phase 1 (no_grad): re-noise the final latent per timestep and cache
                             the K teacher velocities (weight swaps happen here)
                             and the base velocity (LoRA disabled).
          Phase 2 (grad):    combine cached teacher velocities with the current
                             mixing weights and minimize the velocity-MSE to the
                             cached base velocity (plus opt-in regularizers).
        """
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size

        entropy_coeff = self.training_args.klmin_entropy_coeff
        anchor_coeff = self.training_args.klmin_uniform_anchor_coeff

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

                # ---- Phase 1: precompute per-timestep teacher + base velocities ----
                self.adapter.rollout()
                with torch.no_grad(), self.autocast():
                    all_timesteps = self._sample_timesteps(batch_size)
                    all_teacher_velocities: List[torch.Tensor] = []
                    all_base_velocities: List[torch.Tensor] = []

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

                        teacher_velocities = self._compute_teacher_velocities(
                            batch, t_flat, noised_latents
                        )  # (K, B, *latent_dims), detached
                        all_teacher_velocities.append(teacher_velocities)

                        base_velocity = self._compute_base_velocity(
                            batch, t_flat, noised_latents
                        )  # (B, *latent_dims), detached
                        all_base_velocities.append(base_velocity)

                # ---- Phase 2: minimize KL-to-base with current mixing weights ----
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
                            base_velocity = all_base_velocities[t_idx]

                            v_lambda, w_kb = self._compute_combined_velocity(
                                teacher_velocities, t_flat, batch,
                                timestep_index=t_idx, set_ids=set_ids,
                                return_weights=True,
                            )  # v_lambda: (B, *), w_kb: (K, B)

                            # KL-to-base = mean_b mean_spatial ||v_lambda - v_base||^2
                            kl_per_sample = self._klmin_loss(v_lambda, base_velocity)  # (B,)
                            kl_base = kl_per_sample.mean()
                            loss = kl_base

                            # Opt-in: entropy bonus (maximize weight spread).
                            if entropy_coeff > 0:
                                entropy = self._entropy_bonus(w_kb)
                                loss = loss - entropy_coeff * entropy
                                loss_info['entropy'].append(entropy.detach())

                            # Opt-in: uniform anchor (pull weights toward 1/K).
                            if anchor_coeff > 0:
                                anchor = self._uniform_anchor(w_kb, self.K)
                                loss = loss + anchor_coeff * anchor
                                loss_info['uniform_anchor'].append(anchor.detach())

                            # Soft Σw≈1 anchor (weight_normalization='none' only)
                            if self._sum_penalty_active:
                                sum_penalty = (
                                    self.training_args.weight_sum_penalty
                                    * self._weight_sum_penalty_value(w_kb)
                                )
                                loss = loss + sum_penalty
                                loss_info['weight_sum_penalty'].append(sum_penalty.detach())

                            self._append_weight_stats(loss_info, w_kb)
                            loss_info['kl_base'].append(kl_base.detach())
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
