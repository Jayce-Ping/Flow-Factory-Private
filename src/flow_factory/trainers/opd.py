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

# src/flow_factory/trainers/opd.py
"""
On-Policy Distillation (OPD) Trainer for Flow Matching, SDE regime.

Implements the REINFORCE form of the trajectory-level reverse KL
(Eq. 11 in the Flow-OPD paper):

    grad L = E_tau [
        sum_k grad_theta D_k(theta)
        + sum_k R_bar_{k+1} * grad_theta log p_theta(x_{k+1} | x_k)
    ]

where ``D_k = || mu_theta^k - mu_phi^k ||^2 / (2 * sigma_bar_k^2)`` is the
per-step Gaussian KL between the student and a frozen LoRA teacher
``v_phi``, and ``R_bar_{k+1} = sum_{j>k} D_j`` is the closed-form
cumulative-future KL (treated as a constant). One or more teachers can be
attached via ``OPDTrainingArguments.teacher_paths`` and combined either by
per-batch round-robin or per-timestep averaging.
"""

import inspect
import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ..hparams import OPDTrainingArguments
from ..samples import BaseSample
from ..utils.base import (
    create_generator,
    create_generator_by_prompt,
    filter_kwargs,
)
from ..utils.dist import reduce_loss_info
from ..utils.logger_utils import setup_logger
from ..utils.lora_loader import load_lora_as_named_parameters
from ..utils.trajectory_collector import compute_trajectory_indices
from .abc import BaseTrainer

logger = setup_logger(__name__)


# Keys reused across student / teacher adapter.forward calls.
_STUDENT_RETURN_KWARGS = ["log_prob", "next_latents_mean", "std_dev_t", "dt"]
_TEACHER_RETURN_KWARGS = ["next_latents_mean", "std_dev_t", "dt"]


class OPDTrainer(BaseTrainer):
    """On-Policy Distillation trainer (SDE regime, Eq. 11).

    Reuses GRPO's coupled / Flow-SDE training topology — full-trajectory
    rollout with on-policy log-probabilities, then a per-timestep
    forward/backward inside ``optimize()``. Differs from GRPO in three
    ways:

    1. No external reward model — the per-step Gaussian KL ``D_k`` between
       student and teacher serves as the dense reward signal.
    2. No advantage / aggregation step — ``R_bar_{k+1}`` is computed as a
       reverse cumulative sum over ``D_j`` per trajectory inside
       ``optimize()``.
    3. One or more teacher LoRAs are pre-loaded into named-parameter
       snapshots; ``optimize()`` swaps them in via
       ``adapter.use_named_parameters`` to compute ``v_phi``.

    References:
        Flow-OPD: On-Policy Distillation for Flow Matching Models.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: OPDTrainingArguments

        self.num_train_timesteps = self.adapter.scheduler.num_sde_steps
        self.reinforce_coef = self.training_args.reinforce_coef
        self.teacher_aggregation = self.training_args.teacher_aggregation

        # Cache adapter.forward signature once so `_build_forward_kwargs`
        # avoids `inspect.signature` introspection on every per-timestep call.
        sig = inspect.signature(self.adapter.forward)
        self._forward_param_names: frozenset = frozenset(sig.parameters.keys())
        self._forward_accepts_var_kwargs: bool = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        self._teacher_names: List[str] = []
        self._init_teachers()

    # =========================== Initialization ============================
    def _init_teachers(self) -> None:
        """Load each teacher LoRA checkpoint into a named-parameter snapshot."""
        teacher_paths: List[str] = list(self.training_args.teacher_paths)
        if not teacher_paths:
            raise ValueError(
                "OPDTrainer requires at least one teacher LoRA path; "
                f"got teacher_paths={teacher_paths!r}."
            )

        device = self.training_args.teacher_param_device
        for i, path in enumerate(teacher_paths):
            name = f"opd_teacher_{i}"
            load_lora_as_named_parameters(
                adapter=self.adapter,
                name=name,
                lora_path=path,
                device=device,
            )
            self._teacher_names.append(name)
        logger.info(
            f"OPDTrainer initialised with {len(self._teacher_names)} teacher(s): "
            f"{self._teacher_names} (aggregation={self.teacher_aggregation!r}, "
            f"device={device!r})."
        )

    def _teacher_indices_for_batch(self, batch_idx: int, inner_epoch: int) -> List[int]:
        """Return teacher indices used for one micro-batch.

        ``round_robin`` -> single teacher cycling across micro-batches, inner
        epochs, and outer epochs (so different inner epochs of the same outer
        epoch use different teachers; otherwise inner-epoch 0 and inner-epoch 1
        would always pick the same teacher for the same ``batch_idx``).
        ``average`` -> all teachers (forward each, average the velocity).
        """
        num_teachers = len(self._teacher_names)
        if self.teacher_aggregation == "round_robin":
            num_batches = self.training_args.num_batches_per_epoch
            num_inner = self.training_args.num_inner_epochs
            global_batch = (self.epoch * num_inner + inner_epoch) * num_batches + batch_idx
            return [global_batch % num_teachers]
        if self.teacher_aggregation == "average":
            return list(range(num_teachers))
        raise ValueError(
            f"Unknown teacher_aggregation={self.teacher_aggregation!r}; "
            f"expected 'round_robin' or 'average'."
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

    # =========================== Evaluation ============================
    def evaluate(self) -> None:
        """EMA-based evaluation loop (mirrors GRPO/NFT)."""
        if self.test_dataloader is None:
            return

        self.adapter.eval()
        self.eval_reward_buffer.clear()

        with torch.no_grad(), self.autocast(), self.adapter.use_ema_parameters():
            all_samples: List[BaseSample] = []

            for batch in tqdm(
                self.test_dataloader,
                desc="Evaluating",
                disable=not self.show_progress_bar,
            ):
                generator = create_generator_by_prompt(batch["prompt"], self.training_args.seed)
                inference_kwargs = {
                    "compute_log_prob": False,
                    "generator": generator,
                    "trajectory_indices": None,
                    **self.eval_args,
                }
                inference_kwargs.update(**batch)
                inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
                samples = self.adapter.inference(**inference_kwargs)
                all_samples.extend(samples)
                self.eval_reward_buffer.add_samples(samples)

            rewards = self.eval_reward_buffer.finalize(store_to_samples=True, split="pointwise")

            rewards = {
                key: torch.as_tensor(value).to(self.accelerator.device)
                for key, value in rewards.items()
            }
            gathered_rewards = {
                key: self.accelerator.gather(value).cpu().numpy() for key, value in rewards.items()
            }

            if self.accelerator.is_main_process:
                _log_data: Dict[str, Any] = {
                    f"eval/reward_{key}_mean": np.mean(value)
                    for key, value in gathered_rewards.items()
                }
                _log_data.update(
                    {
                        f"eval/reward_{key}_std": np.std(value)
                        for key, value in gathered_rewards.items()
                    }
                )
                _log_data["eval_samples"] = all_samples
                self.log_data(_log_data, step=self.step)
            self.accelerator.wait_for_everyone()

    # =========================== Sampling ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts (mirrors GRPO: full trajectory + on-policy log-probs)."""
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples: List[BaseSample] = []
        data_iter = iter(self.dataloader)
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.scheduler.train_timesteps,
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

        Drains any pending async reward workers (when the user configured
        auxiliary reward models purely for logging); the returned rewards
        are intentionally not consumed by :meth:`optimize`.
        """
        if not self.reward_models:
            return
        rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
        if rewards and self.accelerator.is_main_process:
            log_data: Dict[str, Any] = {}
            for key, value in rewards.items():
                value_np = torch.as_tensor(value).cpu().numpy()
                log_data[f"train/aux_reward_{key}_mean"] = float(np.mean(value_np))
                log_data[f"train/aux_reward_{key}_std"] = float(np.std(value_np))
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

        Uses the parameter names cached in ``__init__`` to avoid the
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
        if self._forward_accepts_var_kwargs:
            forward_kwargs = full_kwargs
        else:
            allowed = self._forward_param_names
            forward_kwargs = {k: v for k, v in full_kwargs.items() if k in allowed}
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
    ) -> torch.Tensor:
        """Compute ``D_k = mean(||mu_s - mu_t||^2) / (2 * sigma_bar^2)`` per batch sample.

        ``sigma_bar^2 = std_dev_t^2 * (-dt)`` follows the Flow-SDE
        discretisation (see Appendix B of the Flow-OPD paper).

        The spatial reduction of ``(mu_s - mu_t)^2`` uses ``mean`` over the
        non-batch dimensions (matching GRPO's reduction convention in
        ``kl_div`` -- see ``trainers/grpo.py``); the OPD-specific
        ``sigma_bar^2`` divisor has no GRPO analogue. The resulting
        per-sample scalar is proportional to the analytical ``D_k`` up to a
        fixed ``(spatial_dims)`` factor that is absorbed into the learning
        rate.
        """
        if mu_student.shape != mu_teacher.shape:
            raise ValueError(
                "mu_student and mu_teacher must have the same shape, "
                f"got mu_student.shape={tuple(mu_student.shape)} vs "
                f"mu_teacher.shape={tuple(mu_teacher.shape)}."
            )

        diff_sq = (mu_student.float() - mu_teacher.float()) ** 2
        diff_sq = diff_sq.mean(dim=tuple(range(1, diff_sq.ndim)))  # (B,)

        # `std_dev_t` and `dt` are produced by the Flow-SDE scheduler in shape
        # `(B, 1, 1)` (per-sample scalars broadcast across spatial dims), so a
        # single `.flatten()` collapses to `(B,)` without a redundant `mean`.
        sigma_bar_sq = ((std_dev_t.float() ** 2) * (-dt.float())).flatten()
        sigma_bar_sq = sigma_bar_sq.clamp(min=1e-12)

        return diff_sq / (2.0 * sigma_bar_sq)

    @staticmethod
    def _reverse_cumulative(
        d_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Return ``[R_1, R_2, ..., R_K]`` with ``R_k = sum_{j > k-1, j in [k, K-1]} D_j``.

        Indexed so ``R_per_k[k] == bar_R_{k+1}`` of the paper: the future KL
        starting AFTER timestep ``k``. The last entry is therefore a
        zero tensor.
        """
        if not d_list:
            return []
        device = d_list[0].device
        dtype = d_list[0].dtype
        shape = d_list[0].shape
        running = torch.zeros(shape, device=device, dtype=dtype)
        r_per_k: List[torch.Tensor] = [None] * len(d_list)  # type: ignore[list-item]
        for k in range(len(d_list) - 1, -1, -1):
            r_per_k[k] = running.clone()
            running = running + d_list[k]
        return r_per_k

    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimisation (Stage 6): two-pass per-batch loss.

        For every micro-batch (lazy reload on GPU to support
        ``offload_samples_to_cpu``):

        1. Pre-pass (``no_grad``): compute ``D_k`` and reverse-cumulative
           ``R_bar_{k+1}`` for every training timestep, AND cache the
           frozen teacher's ``mu_phi`` for reuse in the main pass.
        2. Main pass (with grad): per training timestep, run the student
           forward (gradient-flowing), reuse the cached teacher mean,
           assemble ``loss = D_k + reinforce_coef * R_bar_{k+1}.detach() * log_p``
           and backprop. Matches the GRPO per-timestep ``accumulate``
           pattern so ``gradient_accumulation_steps *= num_train_timesteps``
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
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size

        # Single mode swap for the whole optimize() -- mirrors GRPO's pattern.
        self.adapter.train()

        for inner_epoch in range(self.training_args.num_inner_epochs):
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]

            loss_info: Dict[str, List[torch.Tensor]] = defaultdict(list)

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
                latents_index_map = batch["latent_index_map"]  # (T+1,) LongTensor
                num_timesteps = batch["timesteps"].shape[1]

                teacher_indices = self._teacher_indices_for_batch(batch_idx, inner_epoch)
                loss_info["teacher_idx"].append(
                    torch.as_tensor(float(teacher_indices[0]), device=device)
                )

                # 1. Pre-pass: compute D_k and cache mu_teacher per timestep.
                d_list, mu_teacher_list = self._precompute_d_per_timestep(
                    batch=batch,
                    latents_index_map=latents_index_map,
                    num_timesteps=num_timesteps,
                    teacher_indices=teacher_indices,
                )
                r_per_k = self._reverse_cumulative(d_list)

                # 2. Main pass: per-timestep loss + backward.
                loss_info = self._optimize_train_pass(
                    batch=batch,
                    latents_index_map=latents_index_map,
                    num_timesteps=num_timesteps,
                    mu_teacher_list=mu_teacher_list,
                    r_per_k=r_per_k,
                    loss_info=loss_info,
                )

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
            for timestep_index in self.adapter.scheduler.train_timesteps:
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
                    self.adapter.scheduler.train_timesteps,
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
                        return_kwargs=_STUDENT_RETURN_KWARGS,
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
                    )

                    r_kp1 = r_per_k[k_idx].detach()
                    log_prob_new = student_out.log_prob

                    pathwise_loss = d_k_grad.mean()
                    reinforce_loss = (r_kp1 * log_prob_new).mean()
                    loss = pathwise_loss + self.reinforce_coef * reinforce_loss

                    loss_info["d_k"].append(pathwise_loss.detach())
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
