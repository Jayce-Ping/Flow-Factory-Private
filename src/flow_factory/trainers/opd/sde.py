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

# src/flow_factory/trainers/opd/sde.py
"""On-Policy Distillation (OPD) Trainer for Flow Matching, SDE regime.

Implements the REINFORCE form of the trajectory-level reverse KL
(Eq. 11 in the Flow-OPD paper):

    grad L = E_tau [
        sum_k grad_theta D_k(theta)
        + sum_k R_bar_{k+1} * grad_theta log p_theta(x_{k+1} | x_k)
    ]

where ``D_k`` is the per-step Gaussian KL between the student and a frozen
LoRA teacher (optionally divided by ``2 * sigma_bar_k^2`` via
``normalize_d_k``), and ``R_bar_{k+1}`` aggregates future ``D_j`` by sum
(paper Eq. 11) or mean (``reinforce_future_reduction``). Optionally truncate
to the next ``reinforce_horizon`` steps. REINFORCE may use group-centered
and optionally std-normalized coefficients (``reinforce_group_center``,
``reinforce_group_std``).
One or more teachers can be
attached via ``OPDTrainingArguments.teacher_paths`` and combined either by
per-batch round-robin or per-timestep averaging.
"""

import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ...hparams import OPDTrainingArguments
from ...rewards import RewardBuffer
from ...samples import BaseSample
from ...utils.base import create_generator, create_generator_by_prompt, filter_kwargs
from ...utils.dist import reduce_loss_info
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import compute_trajectory_indices
from ..abc import BaseTrainer
from .common import (
    cache_forward_signature,
    filter_forward_kwargs,
    load_teachers,
    pcgrad_project_gradients,
    teacher_indices_for_batch,
)

logger = setup_logger(__name__)


# Keys reused across student / teacher adapter.forward calls.
_STUDENT_RETURN_KWARGS = ["log_prob", "next_latents_mean", "std_dev_t", "dt"]
_TEACHER_RETURN_KWARGS = ["next_latents_mean", "std_dev_t", "dt"]


class OPDTrainer(BaseTrainer):
    """On-Policy Distillation trainer (SDE regime, Eq. 11).

    Reuses GRPO's coupled / Flow-SDE training topology -- full-trajectory
    rollout with on-policy log-probabilities, then a per-timestep
    forward/backward inside ``optimize()``. Differs from GRPO in three
    ways:

    1. No external reward model -- the per-step Gaussian KL ``D_k`` between
       student and teacher serves as the dense reward signal.
    2. No external reward advantage -- ``R_bar_{k+1}`` is computed from
       future ``D_j`` on the full rank inside ``optimize()`` (sum or mean,
       optional rank-local group centering for REINFORCE).
    3. One or more teacher LoRAs are pre-loaded into named-parameter
       snapshots; ``optimize()`` swaps them in via
       ``adapter.use_named_parameters`` to compute ``v_phi``.

    References:
        Flow-OPD: On-Policy Distillation for Flow Matching Models.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: OPDTrainingArguments

        self._is_ode = self.adapter.scheduler.dynamics_type == "ODE"
        if self._is_ode:
            self.num_train_timesteps = self.training_args.num_inference_steps
        else:
            self.num_train_timesteps = self.adapter.scheduler.num_sde_steps
        self.pathwise_coef = self.training_args.pathwise_coef
        self.reinforce_coef = self.training_args.reinforce_coef
        self.reinforce_horizon = self.training_args.reinforce_horizon
        self.reinforce_future_reduction = self.training_args.reinforce_future_reduction
        self.reinforce_group_center = self.training_args.reinforce_group_center
        self.reinforce_group_std = self.training_args.reinforce_group_std
        self.normalize_d_k = self.training_args.normalize_d_k
        self.teacher_aggregation = self.training_args.teacher_aggregation

        if (self.reinforce_group_center or self.reinforce_group_std) and self.reinforce_coef > 0:
            if self.config.data_args.sampler_type != "group_contiguous":
                raise ValueError(
                    "reinforce_group_center=True and/or reinforce_group_std=True "
                    "require data.sampler_type 'group_contiguous', got "
                    f"{self.config.data_args.sampler_type!r}."
                )

        # Sanity check: warn (do not hard-error) when the configured loss
        # carries no learning signal at all -- e.g. an accidental ablation
        # config. Users may still intentionally hit this path for plumbing
        # tests, so we only emit a warning.
        if self.pathwise_coef == 0 and self.reinforce_coef == 0 and self.training_args.kl_beta == 0:
            logger.warning(
                "OPDTrainer received zero-signal loss config: "
                f"pathwise_coef={self.pathwise_coef}, "
                f"reinforce_coef={self.reinforce_coef}, "
                f"kl_beta={self.training_args.kl_beta}. "
                "All three terms contribute zero gradient; the student "
                "will not move. Set at least one to a positive value."
            )

        # Cache adapter.forward signature once so `_build_forward_kwargs`
        # avoids `inspect.signature` introspection on every per-timestep call.
        self._forward_param_names, self._forward_accepts_var_kwargs = cache_forward_signature(
            self.adapter.forward
        )

        self._teacher_names: List[str] = load_teachers(
            self.adapter,
            list(self.training_args.teacher_paths),
            self.training_args.teacher_param_device,
        )

    @property
    def enable_kl_loss(self) -> bool:
        """KL anchor to pre-trained base is enabled when ``kl_beta > 0``.

        Mirrors :attr:`GRPOTrainer.enable_kl_loss`. When True, every gradient
        step runs an additional reference forward inside
        :meth:`BaseAdapter.use_ref_parameters` (which for LoRA mode disables
        the active LoRA adapter, exposing the underlying base model) and adds
        ``kl_beta * kl_div`` to the per-step loss. Two ``kl_type`` modes are
        supported (see :class:`OPDTrainingArguments.kl_type`):

        - ``'x-based'`` (default): same-variance Gaussian KL on the SDE
          transition mean via :meth:`_compute_per_step_kl` (respects
          ``normalize_d_k``); identical scale to teacher-vs-student ``D_k``.
        - ``'v-based'``: unscaled MSE on the velocity prediction,
          ``mean((noise_pred_s - noise_pred_ref)^2)``; matches GRPO.
        """
        return self.training_args.kl_beta > 0.0

    @property
    def _train_timestep_indices(self):
        """Training timestep indices: all steps for ODE, scheduler-selected for SDE."""
        if self._is_ode:
            return list(range(self.training_args.num_inference_steps))
        return self.adapter.scheduler.train_timesteps

    # =========================== Helper Shims ============================
    def _teacher_indices_for_batch(self, batch_idx: int, inner_epoch: int) -> List[int]:
        """Thin shim around :func:`common.teacher_indices_for_batch` capturing
        the trainer's stateful fields (``self.epoch``, ``training_args``)."""
        return teacher_indices_for_batch(
            teacher_aggregation=self.teacher_aggregation,
            num_teachers=len(self._teacher_names),
            epoch=self.epoch,
            inner_epoch=inner_epoch,
            batch_idx=batch_idx,
            num_inner=self.training_args.num_inner_epochs,
            num_batches=self.training_args.num_batches_per_epoch,
        )

    # =========================== Main Loop ============================
    def start(self):
        """Main training loop (same outer-loop shape as GRPO)."""
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

    # =========================== Sampling ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts (mirrors GRPO: full trajectory + on-policy log-probs)."""
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples: List[BaseSample] = []
        data_iter = iter(self.dataloader)
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self._train_timestep_indices,
            num_inference_steps=self.training_args.num_inference_steps,
        )

        with torch.no_grad(), self.autocast():
            for _ in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f"Epoch {self.epoch} Sampling",
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    "compute_log_prob": True,
                    "trajectory_indices": trajectory_indices,
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

    # =========================== Reward / Advantage (no-op) ============================
    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """OPD has no external advantage stage; teacher KL is the dense reward.

        Three responsibilities, all optional / main-process-only:
          1. Drain any pending async reward workers (when the user configured
             auxiliary reward models purely for logging); the returned rewards
             are intentionally not consumed by :meth:`optimize`.
          2. Log ``train_samples[:30]`` for qualitative inspection on
             wandb / swanlab (matches GRPO's convention so cross-trainer
             panels group cleanly).
          3. Log epoch-level teacher metadata so the cycling teacher slate
             (``round_robin``) or the all-teachers regime (``average``) is
             traceable in the wandb time-series.
        """
        log_data: Dict[str, Any] = {}

        # 1. Aux reward stats (only when reward_models are attached; calling
        # `reward_buffer.finalize` on an empty buffer is a wasted no-op + async-drain).
        if self.reward_models:
            rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
            if rewards and self.accelerator.is_main_process:
                for key, value in rewards.items():
                    value_np = torch.as_tensor(value).cpu().numpy()
                    log_data[f"train/aux_reward_{key}_mean"] = float(np.mean(value_np))
                    log_data[f"train/aux_reward_{key}_std"] = float(np.std(value_np))

        # 2-3. Rollout-sample images + teacher metadata (main process only;
        # samples are rank-local, matching GRPO's `_log_data['train_samples'] = samples[:30]`
        # pattern).
        if self.accelerator.is_main_process:
            log_data["train_samples"] = samples[:30]

            teacher_indices_first_batch = self._teacher_indices_for_batch(
                batch_idx=0, inner_epoch=0
            )
            log_data["train/teacher_index_first_batch"] = float(teacher_indices_first_batch[0])
            log_data["train/num_active_teachers_per_batch"] = float(
                len(teacher_indices_first_batch)
            )

        if log_data:
            self.log_data(log_data, step=self.step)

    # =========================== Optimization ============================
    def _build_forward_kwargs(
        self,
        batch: Dict[str, Any],
        t: torch.Tensor,
        t_next: torch.Tensor,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        compute_log_prob: bool,
        return_kwargs: List[str],
    ) -> Dict[str, Any]:
        """Assemble the per-timestep ``adapter.forward`` kwargs (shared by student / teacher).

        Uses the parameter names cached in ``__init__`` via
        :func:`common.cache_forward_signature` to avoid the
        ``inspect.signature`` introspection that ``filter_kwargs`` would do
        on every call (this helper runs O(num_train_timesteps * num_batches)
        times per inner epoch).
        """
        full_kwargs = {
            **self.training_args,
            "t": t,
            "t_next": t_next,
            "latents": latents,
            "next_latents": next_latents,
            "compute_log_prob": compute_log_prob,
            "noise_level": self.adapter.scheduler.noise_level,
            **batch,
        }
        forward_kwargs = filter_forward_kwargs(
            full_kwargs,
            self._forward_param_names,
            self._forward_accepts_var_kwargs,
        )
        forward_kwargs["return_kwargs"] = return_kwargs
        return forward_kwargs

    def _teacher_next_latents_mean(
        self,
        forward_kwargs: Dict[str, Any],
        teacher_indices: List[int],
    ) -> torch.Tensor:
        """Forward each requested teacher and return the (averaged) ``next_latents_mean``.

        Always detached: the teacher branch contributes no gradient.
        """
        if not teacher_indices:
            raise ValueError("teacher_indices must contain at least one entry.")

        means: List[torch.Tensor] = []
        for t_i in teacher_indices:
            name = self._teacher_names[t_i]
            with self.adapter.use_named_parameters(name):
                out = self.adapter.forward(**forward_kwargs)
            if out.next_latents_mean is None:
                raise RuntimeError(
                    f"Teacher '{name}' forward did not return `next_latents_mean`; "
                    f"check `return_kwargs={forward_kwargs.get('return_kwargs')!r}`."
                )
            means.append(out.next_latents_mean.detach())

        if len(means) == 1:
            return means[0]
        return torch.stack(means, dim=0).mean(dim=0)

    @staticmethod
    def _compute_per_step_kl(
        mu_student: torch.Tensor,
        mu_teacher: torch.Tensor,
        std_dev_t: torch.Tensor,
        dt: torch.Tensor,
        *,
        normalize: bool,
    ) -> torch.Tensor:
        """Per-sample Gaussian transition KL ``D_k`` with optional normalization.

        When ``normalize`` is True: ``mean(||mu_s - mu_t||^2) / (2 * sigma_bar^2)``
        with ``sigma_bar^2 = std_dev_t^2 * (-dt)`` (Flow-SDE, Appendix B).

        When False: ``mean(||mu_s - mu_t||^2)`` only.

        Spatial reduction uses ``mean`` over non-batch dimensions (matching
        GRPO's ``kl_div`` convention in ``trainers/grpo.py``).
        """
        if mu_student.shape != mu_teacher.shape:
            raise ValueError(
                "mu_student and mu_teacher must have the same shape, "
                f"got mu_student.shape={tuple(mu_student.shape)} vs "
                f"mu_teacher.shape={tuple(mu_teacher.shape)}."
            )

        diff_sq = (mu_student.float() - mu_teacher.float()) ** 2
        diff_sq = diff_sq.mean(dim=tuple(range(1, diff_sq.ndim)))  # (B,)

        if not normalize:
            return diff_sq

        # `std_dev_t` and `dt` are produced by the Flow-SDE scheduler in shape
        # `(B, 1, 1)` (per-sample scalars broadcast across spatial dims), so a
        # single `.flatten()` collapses to `(B,)` without a redundant `mean`.
        sigma_bar_sq = ((std_dev_t.float() ** 2) * (-dt.float())).flatten()

        # Under ODE, std_dev_t is zero → σ_bar² = 0. Use σ²=1 convention
        # (plain MSE) to avoid division by zero. Under SDE this gives
        # time-reweighted MSE.
        if sigma_bar_sq.abs().max() < 1e-10:
            return diff_sq

        sigma_bar_sq = sigma_bar_sq.clamp(min=1e-12)

        return diff_sq / (2.0 * sigma_bar_sq)

    def _student_return_kwargs_for_train(self) -> List[str]:
        """Per-step student-forward return keys for the gradient pass.

        Base keys (always needed): ``log_prob`` (REINFORCE), ``next_latents_mean``
        (pathwise D_k + x-based KL), ``std_dev_t`` and ``dt`` (sigma_bar^2).
        Adds ``noise_pred`` only when v-based KL is active; x-based KL reuses
        the already-requested ``next_latents_mean``.
        """
        keys = list(_STUDENT_RETURN_KWARGS)
        if self.enable_kl_loss and self.training_args.kl_type == "v-based":
            keys.append("noise_pred")
        return keys

    def _compute_kl_anchor(
        self,
        student_out: Any,
        forward_kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the KL anchor to the pre-trained base model.

        Runs one reference forward inside ``torch.no_grad() + use_ref_parameters()``
        (LoRA-disable for LoRA mode; EMA snapshot for full fine-tuning) and
        returns ``(kl_div, kl_loss)`` where ``kl_loss = kl_beta * kl_div``.

        Two regimes, selected by ``self.training_args.kl_type``:

        - ``'x-based'``: same-variance Gaussian-KL on the SDE transition mean,
          identical to the teacher-vs-student ``D_k`` formula. Reuses
          :meth:`_compute_per_step_kl`, sharing the
          ``sigma_bar^2 = std_dev_t^2 * (-dt)`` divisor so this term lives on
          the same scale as the teacher pathwise loss.
        - ``'v-based'``: unscaled MSE on the velocity prediction, matching
          GRPO's ``noise_pred`` KL.

        Gradient flow: the reference forward is no-grad (base weights are
        frozen); KL is computed OUTSIDE the no-grad block so autograd records
        the dependency on the student tensor only.
        """
        kl_type = self.training_args.kl_type
        if kl_type == "v-based":
            ref_return_kwargs = ["noise_pred"]
        elif kl_type == "x-based":
            ref_return_kwargs = ["next_latents_mean"]
        else:
            raise ValueError(f"Unknown kl_type={kl_type!r}; expected 'v-based' or 'x-based'.")

        with torch.no_grad(), self.adapter.use_ref_parameters():
            ref_kwargs = forward_kwargs.copy()
            ref_kwargs["compute_log_prob"] = False
            ref_kwargs["return_kwargs"] = ref_return_kwargs
            ref_out = self.adapter.forward(**ref_kwargs)

        if kl_type == "v-based":
            if student_out.noise_pred is None or ref_out.noise_pred is None:
                raise RuntimeError(
                    "v-based KL requires `noise_pred` from both student and "
                    "reference; got "
                    f"student_noise_pred={'set' if student_out.noise_pred is not None else 'None'}, "
                    f"ref_noise_pred={'set' if ref_out.noise_pred is not None else 'None'}."
                )
            kl_div_per_sample = torch.mean(
                (student_out.noise_pred - ref_out.noise_pred) ** 2,
                dim=tuple(range(1, student_out.noise_pred.ndim)),
            )
            kl_div = kl_div_per_sample.mean()
        else:  # x-based
            if student_out.next_latents_mean is None or ref_out.next_latents_mean is None:
                raise RuntimeError(
                    "x-based KL requires `next_latents_mean` from both student "
                    "and reference; got "
                    f"student_next_latents_mean={'set' if student_out.next_latents_mean is not None else 'None'}, "
                    f"ref_next_latents_mean={'set' if ref_out.next_latents_mean is not None else 'None'}."
                )
            # Same Gaussian-KL formula as the teacher-vs-student D_k:
            # mean(||mu_s - mu_ref||^2) / (2 * sigma_bar^2), sigma_bar from the student's scheduler outputs.
            kl_div_per_sample = self._compute_per_step_kl(
                mu_student=student_out.next_latents_mean,
                mu_teacher=ref_out.next_latents_mean,
                std_dev_t=student_out.std_dev_t,
                dt=student_out.dt,
                normalize=self.normalize_d_k,
            )
            kl_div = kl_div_per_sample.mean()

        kl_loss = self.training_args.kl_beta * kl_div
        return kl_div, kl_loss

    @staticmethod
    def _reverse_cumulative(
        d_list: List[torch.Tensor],
        max_future_steps: Optional[int] = None,
        *,
        reduction: Literal["sum", "mean"] = "sum",
    ) -> List[torch.Tensor]:
        """Return per-timestep future KL aggregates for the REINFORCE coefficient.

        Indexed so ``r_per_k[k] == bar_R_{k+1}``: statistics over timesteps
        strictly after ``k``.

        - ``reduction='sum'``: ``bar_R_{k+1} = sum_{j>k} D_j`` (paper Eq. 11).
        - ``reduction='mean'``: ``bar_R_{k+1} = mean_{j>k} D_j``.

        - ``max_future_steps is None``: all future timesteps after ``k``.
        - ``max_future_steps == n``: only ``j in k+1 .. min(k+n, K-1)``.
        """
        if reduction not in ("sum", "mean"):
            raise ValueError(f"expected reduction 'sum' or 'mean', got reduction={reduction!r}.")
        if not d_list:
            return []

        k_len = len(d_list)
        if max_future_steps is not None and max_future_steps < 1:
            raise ValueError(
                f"expected max_future_steps None or >= 1, got max_future_steps={max_future_steps!r}."
            )

        if max_future_steps is None:
            device = d_list[0].device
            dtype = d_list[0].dtype
            shape = d_list[0].shape
            if reduction == "sum":
                running = torch.zeros(shape, device=device, dtype=dtype)
                r_per_k: List[torch.Tensor] = [None] * k_len  # type: ignore[list-item]
                for k in range(k_len - 1, -1, -1):
                    r_per_k[k] = running.clone()
                    running = running + d_list[k]
                return r_per_k

            running_sum = torch.zeros(shape, device=device, dtype=dtype)
            running_count = 0
            r_per_k_mean: List[torch.Tensor] = [None] * k_len  # type: ignore[list-item]
            for k in range(k_len - 1, -1, -1):
                if running_count > 0:
                    r_per_k_mean[k] = running_sum / float(running_count)
                else:
                    r_per_k_mean[k] = torch.zeros(shape, device=device, dtype=dtype)
                running_sum = running_sum + d_list[k]
                running_count += 1
            return r_per_k_mean

        r_per_k: List[torch.Tensor] = []
        for k in range(k_len):
            j_end = min(k + 1 + max_future_steps, k_len)
            if j_end <= k + 1:
                r_per_k.append(torch.zeros_like(d_list[0]))
            else:
                future = torch.stack(d_list[k + 1 : j_end], dim=0)
                if reduction == "sum":
                    r_per_k.append(future.sum(dim=0))
                else:
                    r_per_k.append(future.mean(dim=0))
        return r_per_k

    @staticmethod
    def _group_normalize(
        values: torch.Tensor,
        group_ids: torch.Tensor,
        group_size: int,
        *,
        center: bool = True,
        divide_by_std: bool = False,
        std_eps: float = 1e-6,
        rank_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Per-group mean center and/or std normalization on rank-local ``values``.

        Expects each ``unique_id`` to appear exactly ``group_size`` times
        (``group_contiguous`` sampling on this rank).
        """
        if values.ndim != 1:
            raise ValueError(
                f"expected values.ndim == 1 for group normalization, got values.shape={tuple(values.shape)}."
            )
        if group_ids.shape != values.shape:
            raise ValueError(
                f"group_ids and values must have the same shape, got "
                f"group_ids.shape={tuple(group_ids.shape)} vs values.shape={tuple(values.shape)}."
            )
        if group_size < 2:
            raise ValueError(
                f"expected group_size >= 2 for group normalization, got group_size={group_size!r}."
            )
        if not center and not divide_by_std:
            raise ValueError(
                "expected at least one of center=True or divide_by_std=True for group normalization."
            )

        rank_suffix = f" on rank {rank_index}" if rank_index is not None else ""
        unique_ids = torch.unique(group_ids)
        normalized = values.clone()
        for uid in unique_ids:
            mask = group_ids == uid
            count = int(mask.sum().item())
            if count != group_size:
                raise ValueError(
                    f"expected {group_size} samples for unique_id={uid.item()}, got count={count}"
                    f"{rank_suffix}, num_samples={values.shape[0]}."
                )
            group_vals = values[mask]
            out_vals = group_vals
            if center:
                out_vals = group_vals - group_vals.mean()
            if divide_by_std:
                std = torch.std(group_vals, unbiased=False)
                std = max(float(std.item()), std_eps)
                out_vals = out_vals / std
            normalized[mask] = out_vals
        return normalized

    @staticmethod
    def _group_center(
        values: torch.Tensor,
        group_ids: torch.Tensor,
        group_size: int,
        *,
        rank_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Subtract per-group mean from rank-local ``values`` (shape ``(N,)``)."""
        return OPDTrainer._group_normalize(
            values,
            group_ids,
            group_size,
            center=True,
            divide_by_std=False,
            rank_index=rank_index,
        )

    def _validate_rank_group_layout(
        self,
        group_ids: torch.Tensor,
        group_size: int,
    ) -> None:
        """Fail-fast when this rank's samples do not form complete prompt groups."""
        unique_ids, counts = torch.unique(group_ids, return_counts=True)
        for uid, count in zip(unique_ids, counts):
            if int(count.item()) != group_size:
                raise ValueError(
                    f"expected {group_size} samples for unique_id={uid.item()} on rank "
                    f"{self.accelerator.process_index}, got count={int(count.item())}, "
                    f"num_samples={group_ids.shape[0]}."
                )

    @staticmethod
    def _shuffle_samples_for_optimize(
        samples: List[BaseSample],
        group_size: int,
        group_center: bool,
        generator: torch.Generator,
    ) -> List[BaseSample]:
        """Shuffle samples for ``optimize()``, optionally permuting whole groups."""
        n = len(samples)
        if not group_center:
            perm = torch.randperm(n, generator=generator)
            return [samples[i] for i in perm]

        if n % group_size != 0:
            raise ValueError(
                f"expected len(samples) divisible by group_size for reinforce_group_center, "
                f"got len(samples)={n}, group_size={group_size}."
            )
        num_groups = n // group_size
        group_perm = torch.randperm(num_groups, generator=generator)
        shuffled: List[BaseSample] = []
        for g in group_perm:
            start = int(g.item()) * group_size
            shuffled.extend(samples[start : start + group_size])
        return shuffled

    def _precompute_rank_reinforce_and_teacher(
        self,
        shuffled_samples: List[BaseSample],
        per_device_batch_size: int,
        inner_epoch: int,
        num_batches: int,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """Rank-wide no-grad pre-pass: ``D_k`` buffer, ``R_bar``, optional group center.

        Micro-batches are used only for forward memory; per-timestep ``D_k`` are
        stitched into length-``N`` tensors before reverse-cumulative aggregation
        and rank-local group centering.

        Returns:
            ``(r_per_k, r_per_k_raw, mu_teacher_by_batch)`` where

            - ``r_per_k[k]`` is the REINFORCE coefficient used in the main pass
              (group-centered when ``reinforce_group_center``).
            - ``r_per_k_raw[k]`` is the pre-center ``R_bar`` (``None`` when not
              group-centering).
            - ``mu_teacher_by_batch[batch_idx]`` is the per-timestep teacher
              mean list for that micro-batch's main pass.
        """
        device = self.accelerator.device
        num_samples = len(shuffled_samples)
        group_size = self.training_args.group_size

        rank_group_ids = torch.tensor(
            [sample.unique_id for sample in shuffled_samples],
            device=device,
            dtype=torch.int64,
        )
        if self.reinforce_group_center:
            self._validate_rank_group_layout(rank_group_ids, group_size)

        d_accum: Optional[List[torch.Tensor]] = None
        mu_teacher_by_batch: List[List[torch.Tensor]] = []

        for batch_idx in range(num_batches):
            start = batch_idx * per_device_batch_size
            end = min(start + per_device_batch_size, num_samples)
            batch_size = end - start
            batch_samples = [shuffled_samples[i].to(device) for i in range(start, end)]
            batch = BaseSample.stack(batch_samples)
            latents_index_map = batch["latent_index_map"]
            num_timesteps = batch["timesteps"].shape[1]

            teacher_indices = self._teacher_indices_for_batch(batch_idx, inner_epoch)
            d_list, mu_teacher_list = self._precompute_d_per_timestep(
                batch=batch,
                latents_index_map=latents_index_map,
                num_timesteps=num_timesteps,
                teacher_indices=teacher_indices,
            )
            mu_teacher_by_batch.append(mu_teacher_list)

            if d_accum is None:
                d_accum = [
                    torch.empty(num_samples, device=device, dtype=d_list[0].dtype)
                    for _ in range(len(d_list))
                ]

            for k_idx, d_k in enumerate(d_list):
                if d_k.shape != (batch_size,):
                    raise ValueError(
                        f"expected d_k.shape=({batch_size},) for micro-batch {batch_idx}, "
                        f"got d_k.shape={tuple(d_k.shape)}."
                    )
                d_accum[k_idx][start:end] = d_k.detach()

        if d_accum is None:
            raise RuntimeError(
                "rank pre-pass produced no D_k tensors; expected num_batches >= 1, "
                f"got num_batches={num_batches}, num_samples={num_samples}."
            )

        r_per_k_raw = self._reverse_cumulative(
            d_accum,
            self.reinforce_horizon,
            reduction=self.reinforce_future_reduction,
        )

        if self.reinforce_group_center:
            rank_index = self.accelerator.process_index
            r_per_k = [
                self._group_normalize(
                    r_k,
                    rank_group_ids,
                    group_size,
                    center=True,
                    divide_by_std=self.reinforce_group_std,
                    rank_index=rank_index,
                )
                for r_k in r_per_k_raw
            ]
            return r_per_k, r_per_k_raw, mu_teacher_by_batch

        return r_per_k_raw, None, mu_teacher_by_batch

    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimisation (Stage 6): two-pass per-batch loss.

        For every micro-batch (lazy reload on GPU to support
        ``offload_samples_to_cpu``):

        1. Pre-pass (``no_grad``): compute ``D_k`` and reverse-cumulative
           ``R_bar_{k+1}`` for every training timestep, AND cache the
           frozen teacher's ``mu_phi`` for reuse in the main pass.
        2. Main pass (with grad): per training timestep, run the student
           forward (gradient-flowing), reuse the cached teacher mean,
           assemble ``loss = pathwise_coef * D_k +
           reinforce_coef * R_bar_{k+1}.detach() * log_p`` (plus
           ``kl_beta * KL_anchor`` when ``kl_beta > 0``) and backprop.
           Matches the GRPO per-timestep ``accumulate`` pattern so
           ``gradient_accumulation_steps *= num_train_timesteps``
           (see ``Arguments._adjust_gradient_accumulation``) stays consistent.

        Both passes run in ``self.adapter.train()`` mode (set once below) so
        that the train-time activations underpinning ``R_bar_{k+1}`` agree
        with those underpinning ``D_k(theta)`` and ``log p_theta`` -- if the
        pre-pass used ``rollout()``/``.eval()`` and the main pass used
        ``.train()``, dropout / eval-only normalisations would silently bias
        the REINFORCE coefficient relative to the pathwise term.
        """
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        group_size = self.training_args.group_size

        if self.reinforce_group_center:
            if len(samples) % group_size != 0:
                raise ValueError(
                    f"expected len(samples) divisible by group_size for reinforce_group_center, "
                    f"got len(samples)={len(samples)}, group_size={group_size}."
                )

        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size

        # Single mode swap for the whole optimize() -- mirrors GRPO's pattern.
        self.adapter.train()

        for inner_epoch in range(self.training_args.num_inner_epochs):
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            shuffled_samples = self._shuffle_samples_for_optimize(
                samples,
                group_size=group_size,
                group_center=self.reinforce_group_center,
                generator=perm_gen,
            )

            loss_info: Dict[str, List[torch.Tensor]] = defaultdict(list)

            r_per_k_rank, r_per_k_raw_rank, mu_teacher_by_batch = (
                self._precompute_rank_reinforce_and_teacher(
                    shuffled_samples=shuffled_samples,
                    per_device_batch_size=per_device_batch_size,
                    inner_epoch=inner_epoch,
                    num_batches=num_batches,
                )
            )

            for batch_idx in tqdm(
                range(num_batches),
                total=num_batches,
                desc=f"Epoch {self.epoch} Training",
                position=0,
                disable=not self.show_progress_bar,
            ):
                start = batch_idx * per_device_batch_size
                end = min(start + per_device_batch_size, len(shuffled_samples))
                batch_samples = [shuffled_samples[i].to(device) for i in range(start, end)]
                batch = BaseSample.stack(batch_samples)
                latents_index_map = batch["latent_index_map"]  # (T+1,) LongTensor
                num_timesteps = batch["timesteps"].shape[1]

                teacher_indices = self._teacher_indices_for_batch(batch_idx, inner_epoch)
                loss_info["teacher_idx"].append(
                    torch.as_tensor(float(teacher_indices[0]), device=device)
                )

                r_per_k_mb = [r_k[start:end] for r_k in r_per_k_rank]
                r_per_k_raw_mb = (
                    [r_k[start:end] for r_k in r_per_k_raw_rank]
                    if r_per_k_raw_rank is not None
                    else None
                )

                if self.training_args.teacher_aggregation == "pcgrad":
                    # PCGrad mode: per-teacher D_k + PCGrad projection
                    d_per_teacher_list, mu_teacher_pcgrad = self._precompute_d_per_timestep_pcgrad(
                        batch=batch,
                        latents_index_map=latents_index_map,
                        num_timesteps=num_timesteps,
                        teacher_indices=teacher_indices,
                    )

                    # Compute per-teacher R_bar
                    r_per_k_per_teacher = self._reverse_cumulative_per_teacher(
                        d_per_teacher_list=d_per_teacher_list,
                        num_teachers=len(teacher_indices),
                    )

                    loss_info = self._optimize_train_pass_pcgrad(
                        batch=batch,
                        latents_index_map=latents_index_map,
                        num_timesteps=num_timesteps,
                        mu_teacher_list=mu_teacher_pcgrad,
                        r_per_k_per_teacher=r_per_k_per_teacher,
                        teacher_indices=teacher_indices,
                        loss_info=loss_info,
                    )
                elif self.training_args.teacher_aggregation == "sum":
                    # Sum mode: per-teacher losses summed, single backward (no projection)
                    d_per_teacher_list, mu_teacher_sum = self._precompute_d_per_timestep_pcgrad(
                        batch=batch,
                        latents_index_map=latents_index_map,
                        num_timesteps=num_timesteps,
                        teacher_indices=teacher_indices,
                    )

                    r_per_k_per_teacher = self._reverse_cumulative_per_teacher(
                        d_per_teacher_list=d_per_teacher_list,
                        num_teachers=len(teacher_indices),
                    )

                    loss_info = self._optimize_train_pass_sum(
                        batch=batch,
                        latents_index_map=latents_index_map,
                        num_timesteps=num_timesteps,
                        mu_teacher_list=mu_teacher_sum,
                        r_per_k_per_teacher=r_per_k_per_teacher,
                        teacher_indices=teacher_indices,
                        loss_info=loss_info,
                    )
                else:
                    # Standard mode: averaged teacher D_k
                    loss_info = self._optimize_train_pass(
                        batch=batch,
                        latents_index_map=latents_index_map,
                        num_timesteps=num_timesteps,
                        mu_teacher_list=mu_teacher_by_batch[batch_idx],
                        r_per_k=r_per_k_mb,
                        r_per_k_raw=r_per_k_raw_mb,
                        loss_info=loss_info,
                    )

    def _precompute_d_per_timestep_pcgrad(
        self,
        batch: Dict[str, Any],
        latents_index_map: torch.Tensor,
        num_timesteps: int,
        teacher_indices: List[int],
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """No-grad pass for PCGrad: compute per-teacher D_k and cache per-teacher means.

        In PCGrad mode, we need per-teacher means and per-teacher D_k values so that
        we can compute K separate losses and apply PCGrad projection during the
        main pass.

        Returns:
            ``(d_per_teacher_list, mu_teacher_list)``: per training timestep,
            two lists of length K where each element contains per-sample tensors.
            - ``d_per_teacher_list[t][k]``: shape (B,), D_k for teacher k at timestep t
            - ``mu_teacher_list[t][k]``: latent shape, teacher k's mean at timestep t
        """
        d_per_teacher_list: List[List[torch.Tensor]] = []
        mu_teacher_list: List[List[torch.Tensor]] = []
        device = self.accelerator.device

        with torch.no_grad(), self.autocast():
            # Disable autocast weight cache for the scope of the teacher swap loop.
            # use_named_parameters swaps LoRA weights via .data.copy_() which preserves
            # data_ptr; the autocast cache (keyed by data_ptr) would otherwise serve
            # stale casted weights from the first teacher to subsequent teachers.
            prev_cache = torch.is_autocast_cache_enabled()
            torch.set_autocast_cache_enabled(False)
            try:
                for timestep_index in self._train_timestep_indices:
                    t = batch["timesteps"][:, timestep_index]
                    t_next = (
                        batch["timesteps"][:, timestep_index + 1]
                        if timestep_index + 1 < num_timesteps
                        else torch.tensor(0, device=device)
                    )
                    latents = batch["all_latents"][:, latents_index_map[timestep_index]]
                    next_latents = batch["all_latents"][:, latents_index_map[timestep_index + 1]]

                    forward_kwargs = self._build_forward_kwargs(
                        batch=batch,
                        t=t,
                        t_next=t_next,
                        latents=latents,
                        next_latents=next_latents,
                        compute_log_prob=False,
                        return_kwargs=_TEACHER_RETURN_KWARGS,
                    )

                    # Student forward (for comparison)
                    student_out = self.adapter.forward(**forward_kwargs)
                    if student_out.next_latents_mean is None:
                        raise RuntimeError(
                            "Student forward did not return `next_latents_mean` during pre-pass; "
                            f"requested return_kwargs={_TEACHER_RETURN_KWARGS!r}."
                        )

                    # Per-teacher forward
                    d_teachers_t: List[torch.Tensor] = []
                    mu_teachers_t: List[torch.Tensor] = []

                    for teacher_k in teacher_indices:
                        name = self._teacher_names[teacher_k]
                        with self.adapter.use_named_parameters(name):
                            teacher_out = self.adapter.forward(**forward_kwargs)

                        if teacher_out.next_latents_mean is None:
                            raise RuntimeError(
                                f"Teacher '{name}' forward did not return `next_latents_mean`; "
                                f"check `return_kwargs={forward_kwargs.get('return_kwargs')!r}`."
                            )

                        mu_k = teacher_out.next_latents_mean.detach()
                        mu_teachers_t.append(mu_k)

                        d_k = self._compute_per_step_kl(
                            mu_student=student_out.next_latents_mean,
                            mu_teacher=mu_k,
                            std_dev_t=student_out.std_dev_t,
                            dt=student_out.dt,
                            normalize=self.normalize_d_k,
                        )
                        d_teachers_t.append(d_k.detach())

                    d_per_teacher_list.append(d_teachers_t)
                    mu_teacher_list.append(mu_teachers_t)
            finally:
                torch.set_autocast_cache_enabled(prev_cache)

        return d_per_teacher_list, mu_teacher_list

    def _reverse_cumulative_per_teacher(
        self,
        d_per_teacher_list: List[List[torch.Tensor]],
        num_teachers: int,
    ) -> List[List[torch.Tensor]]:
        """Compute reverse-cumulative R_bar for each teacher independently.

        For each teacher k, aggregate its D_k series into R_bar_k using the
        same :meth:`_reverse_cumulative` logic as the standard path. This
        ensures correct semantics: ``r_per_k_per_teacher[k][t]`` aggregates
        D_j for j strictly after t (i.e., D_{t+1} ... D_{T-1}).

        Args:
            d_per_teacher_list: List of length T, each containing K D_k tensors
                of shape (B,).
            num_teachers: K, number of teachers.

        Returns:
            List of length K, each containing a list of length T with R_bar values.
            ``r_per_k_per_teacher[k][t]`` has shape (B,) and represents the
            future-aggregated reward for teacher k at timestep t.
        """
        T = len(d_per_teacher_list)

        # Reorganize: for each teacher k, gather its D_k series across T timesteps
        r_per_k_per_teacher: List[List[torch.Tensor]] = []
        for teacher_k in range(num_teachers):
            # Gather D_k series for this teacher: [D_0_k, D_1_k, ..., D_{T-1}_k]
            d_series_k = [d_per_teacher_list[t][teacher_k] for t in range(T)]

            # Delegate to the existing _reverse_cumulative which correctly
            # computes r[k] = aggregate(D_{k+1}, ..., D_{T-1}) with proper
            # horizon clipping and sum/mean reduction.
            r_bar_k = self._reverse_cumulative(
                d_series_k,
                self.reinforce_horizon,
                reduction=self.reinforce_future_reduction,
            )
            r_per_k_per_teacher.append(r_bar_k)

        return r_per_k_per_teacher

    def _precompute_d_per_timestep(
        self,
        batch: Dict[str, Any],
        latents_index_map: torch.Tensor,
        num_timesteps: int,
        teacher_indices: List[int],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """No-grad pass: compute ``D_k`` and cache the teacher mean per timestep.

        Returns:
            ``(d_list, mu_teacher_list)``: per training timestep, the detached
            per-sample ``D_k`` tensor of shape ``(B,)`` and the detached
            teacher ``mu_phi`` tensor of latent shape. ``mu_teacher_list`` is
            consumed by :meth:`_optimize_train_pass` so the gradient-bearing
            main pass does NOT re-run the (frozen) teacher forward.
        """
        d_list: List[torch.Tensor] = []
        mu_teacher_list: List[torch.Tensor] = []
        device = self.accelerator.device

        with torch.no_grad(), self.autocast():
            for timestep_index in self._train_timestep_indices:
                t = batch["timesteps"][:, timestep_index]
                t_next = (
                    batch["timesteps"][:, timestep_index + 1]
                    if timestep_index + 1 < num_timesteps
                    else torch.tensor(0, device=device)
                )
                latents = batch["all_latents"][:, latents_index_map[timestep_index]]
                next_latents = batch["all_latents"][:, latents_index_map[timestep_index + 1]]

                forward_kwargs = self._build_forward_kwargs(
                    batch=batch,
                    t=t,
                    t_next=t_next,
                    latents=latents,
                    next_latents=next_latents,
                    compute_log_prob=False,
                    return_kwargs=_TEACHER_RETURN_KWARGS,
                )

                student_out = self.adapter.forward(**forward_kwargs)
                if student_out.next_latents_mean is None:
                    raise RuntimeError(
                        "Student forward did not return `next_latents_mean` during pre-pass; "
                        f"requested return_kwargs={_TEACHER_RETURN_KWARGS!r}."
                    )

                mu_teacher = self._teacher_next_latents_mean(
                    forward_kwargs=forward_kwargs,
                    teacher_indices=teacher_indices,
                )

                d_k = self._compute_per_step_kl(
                    mu_student=student_out.next_latents_mean,
                    mu_teacher=mu_teacher,
                    std_dev_t=student_out.std_dev_t,
                    dt=student_out.dt,
                    normalize=self.normalize_d_k,
                )
                d_list.append(d_k.detach())
                mu_teacher_list.append(mu_teacher.detach())

        return d_list, mu_teacher_list

    def _optimize_train_pass(
        self,
        batch: Dict[str, Any],
        latents_index_map: torch.Tensor,
        num_timesteps: int,
        mu_teacher_list: List[torch.Tensor],
        r_per_k: List[torch.Tensor],
        r_per_k_raw: Optional[List[torch.Tensor]],
        loss_info: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, List[torch.Tensor]]:
        """Main pass: per-timestep student forward + loss + backward.

        The teacher mean ``mu_phi`` for every timestep was already computed
        in the no-grad pre-pass and is consumed via ``mu_teacher_list``.
        Teacher LoRA weights are frozen, so re-running the teacher here
        would produce byte-identical outputs and waste an O(M*K) forward
        pass per micro-batch (and as many CPU-backed
        ``use_named_parameters`` swaps).
        """
        device = self.accelerator.device

        with self.autocast():
            for k_idx, timestep_index in enumerate(
                tqdm(
                    self._train_timestep_indices,
                    desc=f"Epoch {self.epoch} Timestep",
                    position=1,
                    leave=False,
                    disable=not self.show_progress_bar,
                )
            ):
                with self.accelerator.accumulate(*self.adapter.trainable_components):
                    t = batch["timesteps"][:, timestep_index]
                    t_next = (
                        batch["timesteps"][:, timestep_index + 1]
                        if timestep_index + 1 < num_timesteps
                        else torch.tensor(0, device=device)
                    )
                    latents = batch["all_latents"][:, latents_index_map[timestep_index]]
                    next_latents = batch["all_latents"][:, latents_index_map[timestep_index + 1]]

                    forward_kwargs = self._build_forward_kwargs(
                        batch=batch,
                        t=t,
                        t_next=t_next,
                        latents=latents,
                        next_latents=next_latents,
                        compute_log_prob=True,
                        return_kwargs=self._student_return_kwargs_for_train(),
                    )

                    student_out = self.adapter.forward(**forward_kwargs)
                    if student_out.next_latents_mean is None or student_out.log_prob is None:
                        raise RuntimeError(
                            "Student forward must return both `next_latents_mean` "
                            "and `log_prob` for OPD; got "
                            f"next_latents_mean={'set' if student_out.next_latents_mean is not None else 'None'}, "
                            f"log_prob={'set' if student_out.log_prob is not None else 'None'}."
                        )

                    mu_teacher = mu_teacher_list[k_idx]

                    d_k_grad = self._compute_per_step_kl(
                        mu_student=student_out.next_latents_mean,
                        mu_teacher=mu_teacher,
                        std_dev_t=student_out.std_dev_t,
                        dt=student_out.dt,
                        normalize=self.normalize_d_k,
                    )

                    r_kp1 = r_per_k[k_idx].detach()
                    log_prob_new = student_out.log_prob

                    pathwise_loss = d_k_grad.mean()
                    reinforce_loss = (r_kp1 * log_prob_new).mean()
                    loss = self.pathwise_coef * pathwise_loss + self.reinforce_coef * reinforce_loss

                    if self.enable_kl_loss:
                        kl_div, kl_loss = self._compute_kl_anchor(student_out, forward_kwargs)
                        loss = loss + kl_loss
                        loss_info["kl_div"].append(kl_div.detach())
                        loss_info["kl_loss"].append(kl_loss.detach())

                    loss_info["d_k"].append(pathwise_loss.detach())
                    if r_per_k_raw is not None:
                        loss_info["r_bar"].append(r_per_k_raw[k_idx].detach().mean())
                        loss_info["r_bar_adv"].append(r_kp1.mean().detach())
                    else:
                        loss_info["r_bar"].append(r_kp1.mean().detach())
                    loss_info["log_prob"].append(log_prob_new.mean().detach())
                    loss_info["reinforce_loss"].append(reinforce_loss.detach())
                    loss_info["loss"].append(loss.detach())

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.adapter.get_trainable_parameters(),
                            self.training_args.max_grad_norm,
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        loss_info = reduce_loss_info(self.accelerator, loss_info)
                        loss_info["grad_norm"] = grad_norm
                        self.log_data(
                            {f"train/{k}": v for k, v in loss_info.items()},
                            step=self.step,
                        )
                        self.step += 1
                        loss_info = defaultdict(list)

        return loss_info

    def _optimize_train_pass_sum(
        self,
        batch: Dict[str, Any],
        latents_index_map: torch.Tensor,
        num_timesteps: int,
        mu_teacher_list: List[List[torch.Tensor]],
        r_per_k_per_teacher: List[List[torch.Tensor]],
        teacher_indices: List[int],
        loss_info: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, List[torch.Tensor]]:
        """Sum mode: per-teacher losses summed into a single backward per timestep.

        Unlike PCGrad, this mode does NOT project gradients to resolve conflicts.
        Each teacher's loss is computed independently and summed; the combined loss
        is backpropagated in a single pass. Gradients from all K teachers accumulate
        naturally (direct sum in gradient space).

        This serves as the ablation baseline for PCGrad — same per-teacher loss
        decomposition, same per-teacher R_bar, but without conflict resolution.

        Uses the same T-step internal accumulation pattern as PCGrad: all T
        timesteps are processed per batch with a single optimizer.step() at the end.
        """
        device = self.accelerator.device
        K = len(teacher_indices)
        T = len(self._train_timestep_indices)

        with self.accelerator.accumulate(*self.adapter.trainable_components):
            with self.autocast():
                for k_idx, timestep_index in enumerate(
                    tqdm(
                        self._train_timestep_indices,
                        desc=f"Epoch {self.epoch} Timestep",
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    )
                ):
                    t = batch["timesteps"][:, timestep_index]
                    t_next = (
                        batch["timesteps"][:, timestep_index + 1]
                        if timestep_index + 1 < num_timesteps
                        else torch.tensor(0, device=device)
                    )
                    latents = batch["all_latents"][:, latents_index_map[timestep_index]]
                    next_latents = batch["all_latents"][:, latents_index_map[timestep_index + 1]]

                    forward_kwargs = self._build_forward_kwargs(
                        batch=batch,
                        t=t,
                        t_next=t_next,
                        latents=latents,
                        next_latents=next_latents,
                        compute_log_prob=True,
                        return_kwargs=self._student_return_kwargs_for_train(),
                    )

                    # Single student forward (grad flows through all K teacher losses)
                    student_out = self.adapter.forward(**forward_kwargs)
                    if student_out.next_latents_mean is None or student_out.log_prob is None:
                        raise RuntimeError(
                            "Student forward must return both `next_latents_mean` "
                            "and `log_prob` for OPD; got "
                            f"next_latents_mean={'set' if student_out.next_latents_mean is not None else 'None'}, "
                            f"log_prob={'set' if student_out.log_prob is not None else 'None'}."
                        )

                    log_prob_new = student_out.log_prob
                    mu_teachers_k = mu_teacher_list[k_idx]

                    # Sum K per-teacher losses into a single scalar
                    combined_loss = torch.tensor(0.0, device=device)
                    for teacher_k in range(K):
                        d_k = self._compute_per_step_kl(
                            mu_student=student_out.next_latents_mean,
                            mu_teacher=mu_teachers_k[teacher_k],
                            std_dev_t=student_out.std_dev_t,
                            dt=student_out.dt,
                            normalize=self.normalize_d_k,
                        )

                        loss_k = self.pathwise_coef * d_k.mean()
                        if self.reinforce_coef > 0:
                            r_bar_k = r_per_k_per_teacher[teacher_k][k_idx].detach()
                            loss_k = loss_k + self.reinforce_coef * (r_bar_k * log_prob_new).mean()

                        combined_loss = combined_loss + loss_k
                        loss_info[f"d_k_teacher_{teacher_k}"].append(d_k.mean().detach())

                    # Optional KL anchor
                    if self.enable_kl_loss:
                        kl_div, kl_loss = self._compute_kl_anchor(student_out, forward_kwargs)
                        combined_loss = combined_loss + kl_loss
                        loss_info["kl_div"].append(kl_div.detach())
                        loss_info["kl_loss"].append(kl_loss.detach())

                    loss_info["loss"].append(combined_loss.detach())
                    loss_info["log_prob"].append(log_prob_new.mean().detach())

                    # Single backward for the summed loss, divided by T for averaging
                    self.accelerator.backward(combined_loss / T)

        # Clip + step when sync_gradients becomes True (GAS counter reached)
        if self.accelerator.sync_gradients:
            grad_norm = self.accelerator.clip_grad_norm_(
                self.adapter.get_trainable_parameters(),
                self.training_args.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_info = reduce_loss_info(self.accelerator, loss_info)
            loss_info["grad_norm"] = grad_norm
            self.log_data(
                {f"train/{k}": v for k, v in loss_info.items()},
                step=self.step,
            )
            self.step += 1
            loss_info = defaultdict(list)

        return loss_info

    def _optimize_train_pass_pcgrad(
        self,
        batch: Dict[str, Any],
        latents_index_map: torch.Tensor,
        num_timesteps: int,
        mu_teacher_list: List[List[torch.Tensor]],
        r_per_k_per_teacher: List[List[torch.Tensor]],
        teacher_indices: List[int],
        loss_info: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, List[torch.Tensor]]:
        """PCGrad main pass: per-timestep K backward + PCGrad projection + accumulation.

        Unlike the standard _optimize_train_pass, this method:
        1. Computes K per-teacher losses per timestep
        2. Performs K backward passes with retain_graph
        3. Applies PCGrad projection to resolve conflicts
        4. Accumulates projected gradients across all T timesteps
        5. Performs a single optimizer.step() after all timesteps

        This ensures conflicting teacher signals are resolved per-timestep while
        maintaining smooth gradient accumulation across the trajectory.

        GAS compatibility: wraps the batch-level logic with
        ``accelerator.accumulate()`` so cross-batch accumulation (base_GAS > 1)
        is managed by Accelerate's internal counter. The T-step internal
        accumulation is handled here; external GAS only controls how many
        batches to accumulate before stepping.
        """
        device = self.accelerator.device
        trainable_params = list(self.adapter.get_trainable_parameters())
        K = len(teacher_indices)
        T = len(self._train_timestep_indices)

        with self.accelerator.accumulate(*self.adapter.trainable_components):
            # Local buffer for this batch's T-timestep accumulated gradients
            batch_grad = [torch.zeros_like(p) for p in trainable_params]

            with self.autocast():
                for k_idx, timestep_index in enumerate(
                    tqdm(
                        self._train_timestep_indices,
                        desc=f"Epoch {self.epoch} Timestep",
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    )
                ):
                    t = batch["timesteps"][:, timestep_index]
                    t_next = (
                        batch["timesteps"][:, timestep_index + 1]
                        if timestep_index + 1 < num_timesteps
                        else torch.tensor(0, device=device)
                    )
                    latents = batch["all_latents"][:, latents_index_map[timestep_index]]
                    next_latents = batch["all_latents"][:, latents_index_map[timestep_index + 1]]

                    forward_kwargs = self._build_forward_kwargs(
                        batch=batch,
                        t=t,
                        t_next=t_next,
                        latents=latents,
                        next_latents=next_latents,
                        compute_log_prob=True,
                        return_kwargs=self._student_return_kwargs_for_train(),
                    )

                    # Student forward (with grad, shared across K teacher losses)
                    student_out = self.adapter.forward(**forward_kwargs)
                    if student_out.next_latents_mean is None or student_out.log_prob is None:
                        raise RuntimeError(
                            "Student forward must return both `next_latents_mean` "
                            "and `log_prob` for OPD; got "
                            f"next_latents_mean={'set' if student_out.next_latents_mean is not None else 'None'}, "
                            f"log_prob={'set' if student_out.log_prob is not None else 'None'}."
                        )

                    log_prob_new = student_out.log_prob
                    mu_teachers_k = mu_teacher_list[k_idx]

                    # Compute K per-teacher losses
                    per_teacher_losses: List[torch.Tensor] = []
                    for teacher_k in range(K):
                        d_k = self._compute_per_step_kl(
                            mu_student=student_out.next_latents_mean,
                            mu_teacher=mu_teachers_k[teacher_k],
                            std_dev_t=student_out.std_dev_t,
                            dt=student_out.dt,
                            normalize=self.normalize_d_k,
                        )

                        loss_k = self.pathwise_coef * d_k.mean()
                        if self.reinforce_coef > 0:
                            r_bar_k = r_per_k_per_teacher[teacher_k][k_idx].detach()
                            loss_k = loss_k + self.reinforce_coef * (r_bar_k * log_prob_new).mean()

                        per_teacher_losses.append(loss_k)
                        loss_info[f"d_k_teacher_{teacher_k}"].append(d_k.mean().detach())

                    # K backward passes -> K gradient snapshots
                    # retain_graph must stay True through the last teacher if KL loss
                    # will also backward through the same student_out graph.
                    per_teacher_grads: List[List[torch.Tensor]] = []
                    for teacher_k in range(K):
                        self.optimizer.zero_grad()
                        retain = (teacher_k < K - 1) or self.enable_kl_loss
                        per_teacher_losses[teacher_k].backward(retain_graph=retain)
                        grad_snapshot = [
                            p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                            for p in trainable_params
                        ]
                        per_teacher_grads.append(grad_snapshot)

                    # PCGrad projection
                    projected = pcgrad_project_gradients(
                        per_teacher_grads, eps=self.training_args.pcgrad_eps
                    )

                    # Optional KL anchor (shared, not part of PCGrad conflict resolution)
                    if self.enable_kl_loss:
                        self.optimizer.zero_grad()
                        kl_div, kl_loss = self._compute_kl_anchor(student_out, forward_kwargs)
                        kl_loss.backward()
                        for i, p in enumerate(trainable_params):
                            if p.grad is not None:
                                projected[i] = projected[i] + p.grad
                        loss_info["kl_div"].append(kl_div.detach())
                        loss_info["kl_loss"].append(kl_loss.detach())

                    # Accumulate this timestep's projected gradient into the batch buffer
                    for i in range(len(trainable_params)):
                        batch_grad[i] += projected[i]

                    # Log average loss across teachers for this timestep
                    avg_loss = sum(l.detach() for l in per_teacher_losses) / K
                    loss_info["loss"].append(avg_loss)
                    loss_info["log_prob"].append(log_prob_new.mean().detach())

            # Set this batch's gradient contribution (averaged over T timesteps).
            # When GAS > 1, p.grad may already contain contributions from previous
            # batches; add to it rather than replacing.
            self.optimizer.zero_grad()
            for i, p in enumerate(trainable_params):
                p.grad = batch_grad[i] / T

            # Note: accelerator.backward() is not called here; we set p.grad
            # directly from the projected buffer. The accumulate() context
            # manages the sync_gradients flag for GAS tracking.

        # Clip + step when sync_gradients becomes True (GAS counter reached)
        if self.accelerator.sync_gradients:
            grad_norm = self.accelerator.clip_grad_norm_(
                trainable_params,
                self.training_args.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_info = reduce_loss_info(self.accelerator, loss_info)
            loss_info["grad_norm"] = grad_norm
            self.log_data(
                {f"train/{k}": v for k, v in loss_info.items()},
                step=self.step,
            )
            self.step += 1
            loss_info = defaultdict(list)

        return loss_info

