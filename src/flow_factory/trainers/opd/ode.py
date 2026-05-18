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

# src/flow_factory/trainers/opd/ode.py
"""On-Policy Distillation (OPD) Trainer for Flow Matching, ODE regime.

Implements Algorithm 2 of the Flow-OPD paper (Eq. 13): a fully pathwise
loss with BPTT through a differentiable Euler rollout. No REINFORCE term
and no stochastic-trajectory log-probability -- the entire trajectory is
a deterministic function of theta:

    x_{t_{j+1}} = x_{t_j} + v_theta(x_{t_j}, t_j) * dt_j        (Euler step, WITH grad)

    L(theta) = (1/M) * sum_m sum_{j=0}^{N-1}
                  (dt_j**2 / 2) * mean(|| v_theta - v_phi^(m) ||^2)

    grad L = sum_j  d D_j / d theta             (BPTT through the solver)

Required scheduler config: ``dynamics_type: 'ODE'`` and ``noise_level: 0``.
"""

from __future__ import annotations

import os
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ...hparams import OPDODETrainingArguments
from ...scheduler import set_scheduler_timesteps
from ...utils.base import create_generator, create_generator_by_prompt, filter_kwargs
from ...utils.dist import reduce_loss_info
from ...utils.logger_utils import setup_logger
from ..abc import BaseTrainer
from .common import (
    cache_forward_signature,
    filter_forward_kwargs,
    load_teachers,
    teacher_indices_for_batch,
)

logger = setup_logger(__name__)


def prepare_train_timesteps(
    scheduler,
    *,
    num_inference_steps: int,
    height: int,
    width: int,
    patch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """SD3.5-compatible noise schedule for OPD-ODE Euler training.

    Mirrors ``SD3_5Adapter.inference`` step 5: ``image_seq_len`` from latent
    spatial size, then ``set_scheduler_timesteps`` (shift + retrieve_timesteps).
    Returns scheduler-scale timesteps in inference order (noisy → clean).
    """
    latent_h = height // 8
    latent_w = width // 8
    image_seq_len = (latent_h // patch_size) * (latent_w // patch_size)
    return set_scheduler_timesteps(
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
        seq_len=image_seq_len,
        device=device,
    )


class OPDODETrainer(BaseTrainer):
    """On-Policy Distillation trainer (ODE regime, Eq. 13).

    Unlike :class:`OPDTrainer` (SDE), this trainer does NOT cache trajectories
    from a no-grad rollout. The full N-step Euler rollout is run inside
    :meth:`optimize` with ``requires_grad=True`` on the student so the
    pathwise loss can backprop through every step (BPTT through the ODE
    solver). Each Euler step calls ``accelerator.backward`` (GRPO / SDE-OPD
    pattern) so truncated BPTT (``bptt_steps``) caps peak autograd memory;
    full BPTT remains O(N) unless :attr:`solver_checkpointing` is enabled.

    Teacher administration (LoRA snapshot loading and per-batch
    round-robin / average selection) is shared with :class:`OPDTrainer` via
    :mod:`flow_factory.trainers.opd.common` -- both classes inherit directly
    from :class:`BaseTrainer` to keep the flat trainer hierarchy
    (constraint #11).

    The teacher forward is run inside :meth:`_teacher_frozen_context`, which
    temporarily flips ``requires_grad=False`` on the swapped LoRA Parameter
    slots so the input gradient propagates through ``x_{t_j}`` (needed for
    BPTT) but NO spurious gradient accumulates into the student LoRA
    ``.grad`` channel.

    References:
        Flow-OPD: On-Policy Distillation for Flow Matching Models.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: OPDODETrainingArguments

        if self.adapter.scheduler.dynamics_type != "ODE":
            raise ValueError(
                "OPDODETrainer requires `scheduler.dynamics_type == 'ODE'`, "
                f"got {self.adapter.scheduler.dynamics_type!r}. Set "
                "`scheduler.dynamics_type: 'ODE'` and `scheduler.noise_level: 0` "
                "in your config (or use `OPDTrainer` / `trainer_type: 'opd'` for "
                "the SDE regime)."
            )

        self.pathwise_coef = self.training_args.pathwise_coef
        self.solver_checkpointing = self.training_args.solver_checkpointing
        self.bptt_steps = (
            self.training_args.bptt_steps
        )  # None = full BPTT, int >= 1 = truncated segment length
        self.teacher_aggregation = self.training_args.teacher_aggregation

        if self.pathwise_coef == 0 and self.training_args.kl_beta == 0:
            logger.warning(
                "OPDODETrainer received zero-signal loss config: "
                f"pathwise_coef={self.pathwise_coef}, "
                f"kl_beta={self.training_args.kl_beta}. "
                "Both contribute zero gradient; the student will not move. "
                "Set at least one to a positive value."
            )

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

        For LoRA mode the reference forward disables the active LoRA adapter
        (exposing the underlying base), so base parameters are inherently
        ``requires_grad=False`` and no temporary freezing is needed -- unlike
        the teacher branch which lives in the same LoRA Parameter slots as
        the student.
        """
        return self.training_args.kl_beta > 0.0

    # =========================== Helper Shims ============================
    def _teacher_indices_for_batch(self, batch_idx: int, inner_epoch: int) -> List[int]:
        """Thin shim around :func:`common.teacher_indices_for_batch`."""
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
        """Main training loop (same outer-loop shape as OPDTrainer / GRPO)."""
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
        """EMA-based evaluation loop (mirrors OPDTrainer / GRPO / NFT)."""
        if self.test_dataloader is None:
            return

        self.adapter.eval()
        self.eval_reward_buffer.clear()

        with torch.no_grad(), self.autocast(), self.adapter.use_ema_parameters():
            all_samples: List[Any] = []

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

    # =========================== Sampling (prompt-only) ============================
    def sample(self) -> List[Dict[str, Any]]:
        """Collect encoded-prompt batches from the dataloader -- NO rollout.

        OPD-ODE re-samples initial Gaussian noise and runs a fresh
        differentiable Euler rollout per micro-batch inside :meth:`optimize`,
        so the only thing :meth:`sample` needs to do is collect this epoch's
        prompt batches (already encoded by ``adapter.preprocess_func`` during
        ``_init_dataloader``).
        """
        samples: List[Dict[str, Any]] = []
        data_iter = iter(self.dataloader)
        for _ in tqdm(
            range(self.training_args.num_batches_per_epoch),
            desc=f"Epoch {self.epoch} Sampling (prompts only)",
            disable=not self.show_progress_bar,
        ):
            samples.append(next(data_iter))
        return samples

    # =========================== Reward / Advantage (no-op) ============================
    def prepare_feedback(self, samples: List[Dict[str, Any]]) -> None:
        """OPD-ODE has no external advantage stage; teacher KL is the dense reward.

        Logs only epoch-level teacher metadata (no `train_samples` because
        :meth:`sample` does not decode any images; users who want
        rank-local image previews should run a separate eval loop).
        """
        if not self.accelerator.is_main_process:
            return
        log_data: Dict[str, Any] = {}
        teacher_indices_first_batch = self._teacher_indices_for_batch(batch_idx=0, inner_epoch=0)
        log_data["train/teacher_index_first_batch"] = float(teacher_indices_first_batch[0])
        log_data["train/num_active_teachers_per_batch"] = float(len(teacher_indices_first_batch))
        self.log_data(log_data, step=self.step)

    # =========================== Optimization ============================
    def _build_forward_kwargs(
        self,
        batch: Dict[str, Any],
        t: torch.Tensor,
        t_next: torch.Tensor,
        latents: torch.Tensor,
        return_kwargs: List[str],
    ) -> Dict[str, Any]:
        """Assemble per-Euler-step ``adapter.forward`` kwargs for ODE-OPD.

        ODE never needs ``compute_log_prob=True`` (no stochastic transition)
        and never passes ``next_latents`` (we use the scheduler's
        ``next_latents_mean`` as ``x_next`` for the Euler step).
        """
        full_kwargs = {
            **self.training_args,
            "t": t,
            "t_next": t_next,
            "latents": latents,
            "next_latents": None,
            "compute_log_prob": False,
            "noise_level": 0.0,
            **batch,
        }
        forward_kwargs = filter_forward_kwargs(
            full_kwargs,
            self._forward_param_names,
            self._forward_accepts_var_kwargs,
        )
        forward_kwargs["return_kwargs"] = return_kwargs
        return forward_kwargs

    @contextmanager
    def _teacher_frozen_context(self, name: str):
        """Swap in teacher LoRA + temporarily disable param-grad on swapped slots.

        Why this is necessary: :meth:`BaseAdapter.use_named_parameters` swaps
        the teacher's stored data into the student's LoRA ``Parameter`` slots
        (which have ``requires_grad=True``). Under regular forward this would
        cause the teacher's forward to record an autograd edge into those
        slots, so backward would accumulate spurious gradient there -- which
        is the student's ``.grad``. We don't want that.

        Flipping ``requires_grad=False`` on the swapped slots for the duration
        of the teacher forward stops the edge from being recorded but still
        lets gradient flow to the INPUT (``x_{t_j}`` carries grad from the
        rollout chain). On exit, ``.data`` is restored by
        ``use_named_parameters`` and the original ``requires_grad`` flags are
        restored by this context.
        """
        info = self.adapter._named_parameters[name]
        live = self.adapter._get_component_parameters(info.target_components)
        saved_flags = [p.requires_grad for p in live]
        try:
            with self.adapter.use_named_parameters(name):
                for p in live:
                    p.requires_grad_(False)
                yield
        finally:
            for p, flag in zip(live, saved_flags, strict=True):
                p.requires_grad_(flag)

    def _teacher_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        batch: Dict[str, Any],
        teacher_indices: List[int],
    ) -> torch.Tensor:
        """Average teacher ``noise_pred`` at the (differentiable) state ``x``.

        Returns a tensor with gradient flowing back through ``x`` but NOT
        through any LoRA parameter slot (see :meth:`_teacher_frozen_context`).
        """
        if not teacher_indices:
            raise ValueError("teacher_indices must contain at least one entry.")

        preds: List[torch.Tensor] = []
        for ti in teacher_indices:
            name = self._teacher_names[ti]
            with self._teacher_frozen_context(name):
                fwd_kwargs = self._build_forward_kwargs(
                    batch, t, t_next, x, return_kwargs=["noise_pred"]
                )
                out = self.adapter.forward(**fwd_kwargs)
            if out.noise_pred is None:
                raise RuntimeError(
                    f"Teacher '{name}' forward did not return `noise_pred`; "
                    "check that the requested return_kwargs flow through the adapter."
                )
            preds.append(out.noise_pred)

        if len(preds) == 1:
            return preds[0]
        return torch.stack(preds, dim=0).mean(dim=0)

    def _student_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One Euler step.

        Returns ``(noise_pred, next_latents_mean)``. Under ODE dynamics,
        ``next_latents_mean = latents + noise_pred * dt`` (see
        :func:`UniPCMultistepSDEScheduler.step` ODE branch), so the second
        return value IS the next state of the trajectory.

        Optionally wrapped in :func:`torch.utils.checkpoint.checkpoint` to
        trade O(1) solver-depth memory for ~2x forward compute. Only the
        transformer forward is checkpointed; the resulting tensors flow
        into the regular autograd graph so cross-step gradient propagation
        (``x_{t_{j+1}}`` -> ``x_{t_j}``) stays O(N) instead of O(N**2).
        """

        # Closure captures non-tensor args (t, t_next, batch); checkpoint only sees `x`.
        def _impl(x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            fwd_kwargs = self._build_forward_kwargs(
                batch, t, t_next, x_in, return_kwargs=["noise_pred", "next_latents_mean"]
            )
            out = self.adapter.forward(**fwd_kwargs)
            if out.noise_pred is None or out.next_latents_mean is None:
                raise RuntimeError(
                    "Student forward must return both `noise_pred` and "
                    "`next_latents_mean` for OPD-ODE; got "
                    f"noise_pred={'set' if out.noise_pred is not None else 'None'}, "
                    f"next_latents_mean={'set' if out.next_latents_mean is not None else 'None'}."
                )
            return out.noise_pred, out.next_latents_mean

        if self.solver_checkpointing:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(_impl, x, use_reentrant=False)
        return _impl(x)

    def _sample_initial_latents(self, batch: Dict[str, Any], seed_offset: int) -> torch.Tensor:
        """Sample initial Gaussian noise ``x_{t_0}`` for the rollout.

        Wraps the diffusers-style ``self.adapter.pipeline.prepare_latents``
        with the conventional ``(batch_size, num_channels, height, width,
        dtype, device, generator)`` signature, which works for SD3.5, FLUX,
        and Qwen-Image pipelines. Adapters with non-standard signatures can
        override this method in a subclass.

        Determinism: seeded from ``(seed, epoch, inner_epoch)`` so multi-inner
        epoch repeats are reproducible and all teachers within one micro-batch
        share the same ``x_{t_0}``.
        """
        prompt_embeds = batch.get("prompt_embeds")
        if prompt_embeds is None:
            raise RuntimeError(
                "OPDODETrainer._sample_initial_latents requires 'prompt_embeds' in "
                f"the batch (produced by adapter.preprocess_func); got keys={sorted(batch.keys())}."
            )
        batch_size = prompt_embeds.shape[0]

        pipe = self.adapter.pipeline
        height = self.training_args.height
        width = self.training_args.width
        dtype = pipe.transformer.dtype
        device = self.accelerator.device
        num_channels = pipe.transformer.config.in_channels

        # Reuse `create_generator` so the seed-namespace matches the rest of
        # the codebase (deterministic across ranks for the same integer key).
        generator = create_generator(
            self.training_args.seed, self.epoch, seed_offset, device=device
        )

        return pipe.prepare_latents(
            batch_size,
            num_channels,
            height,
            width,
            dtype,
            device,
            generator,
        )

    def _compute_kl_anchor_ode(
        self,
        student_noise_pred: torch.Tensor,
        student_next_mean: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        dt_scalar: torch.Tensor,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-step KL anchor to the pre-trained base model.

        ``x-based``: ``mean(||mu_student - mu_ref||^2) / (2 * dt_scalar^2)``,
        which under ODE reduces to ``mean((v_s - v_ref)^2) / 2`` (because
        ``mu = x + v * dt``); matches the OPD-ODE pathwise scale up to the
        ``dt_j^2 / 2`` factor (folded in for direct comparability with
        ``D_j``).

        ``v-based``: ``mean((noise_pred_student - noise_pred_ref)^2)``;
        matches GRPO / NFT.

        For LoRA mode the reference forward runs under
        :meth:`BaseAdapter.use_ref_parameters` (which disables the active
        LoRA adapter, exposing the underlying base whose params are
        ``requires_grad=False``); no temporary param-grad toggling needed.
        Input grad on ``x`` is preserved for BPTT.
        """
        kl_type = self.training_args.kl_type
        if kl_type == "v-based":
            ref_return_kwargs = ["noise_pred"]
        elif kl_type == "x-based":
            ref_return_kwargs = ["noise_pred", "next_latents_mean"]
        else:
            raise ValueError(f"Unknown kl_type={kl_type!r}; expected 'v-based' or 'x-based'.")

        with self.adapter.use_ref_parameters():
            fwd_kwargs = self._build_forward_kwargs(
                batch, t, t_next, x, return_kwargs=ref_return_kwargs
            )
            ref_out = self.adapter.forward(**fwd_kwargs)

        if kl_type == "v-based":
            if ref_out.noise_pred is None:
                raise RuntimeError("v-based KL requires `noise_pred` from the reference forward.")
            kl_div = (student_noise_pred - ref_out.noise_pred).pow(2).flatten(1).mean(dim=1).mean()
        else:  # x-based
            if ref_out.next_latents_mean is None:
                raise RuntimeError(
                    "x-based KL requires `next_latents_mean` from the reference forward."
                )
            # mu = x + v * dt; under same-variance Gaussian KL the result is
            # mean(||mu_s - mu_ref||^2) / (2 * dt^2). We fold in dt^2 so the
            # KL scale matches D_j = (dt^2 / 2) * mean((v_s - v_t)^2).
            diff_sq = (
                (student_next_mean.float() - ref_out.next_latents_mean.float())
                .pow(2)
                .flatten(1)
                .mean(dim=1)
            )
            dt_sq = dt_scalar.float().pow(2).clamp(min=1e-12)
            kl_div = (diff_sq / (2.0 * dt_sq)).mean()

        kl_loss = self.training_args.kl_beta * kl_div
        return kl_div, kl_loss

    def optimize(self, samples: List[Dict[str, Any]]) -> None:
        """Policy optimisation (Stage 6): differentiable Euler rollout + BPTT.

        For every micro-batch:

        1. Sample initial Gaussian noise ``x_{t_0}`` (deterministic per
           ``(epoch, inner_epoch)``).
        2. Run an N-step Euler rollout WITH gradient; per Euler step compute
           ``D_j`` (and optional KL anchor) and call
           ``accelerator.backward`` immediately (GRPO / SDE-OPD pattern).
           When ``bptt_steps`` is set, ``x`` is detached every
           ``bptt_steps`` steps so autograd spans at most that many consecutive
           student forwards (truncated BPTT).
        3. ``gradient_accumulation_steps`` is multiplied by
           ``num_inference_steps`` in ``Arguments._adjust_gradient_accumulation``
           (see :meth:`OPDODETrainingArguments.get_num_train_timesteps`).
        """
        device = self.accelerator.device
        height = self.training_args.height
        width = self.training_args.width
        if height is None or width is None:
            raise ValueError(
                "OPD-ODE requires training height and width for scheduler timesteps, "
                f"got height={height!r}, width={width!r}. "
                "Set `training.resolution` or `training.height` / `training.width` in the config."
            )
        patch_size = self.adapter.pipeline.transformer.config.patch_size
        timesteps = prepare_train_timesteps(
            self.adapter.scheduler,
            num_inference_steps=self.training_args.num_inference_steps,
            height=height,
            width=width,
            patch_size=patch_size,
            device=device,
        )
        num_steps = len(timesteps)
        if num_steps != self.training_args.num_inference_steps:
            raise ValueError(
                f"OPD-ODE train timestep count mismatch after set_scheduler_timesteps: "
                f"len(timesteps)={num_steps}, "
                f"expected training.num_inference_steps="
                f"{self.training_args.num_inference_steps}. "
                f"image_seq_len derived from height={height}, "
                f"width={width}, patch_size={patch_size}."
            )

        self.adapter.train()

        for inner_epoch in range(self.training_args.num_inner_epochs):
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled = [samples[i] for i in perm]

            loss_info: Dict[str, List[torch.Tensor]] = defaultdict(list)

            for batch_idx, batch in enumerate(
                tqdm(
                    shuffled,
                    total=len(shuffled),
                    desc=f"Epoch {self.epoch} Training (BPTT)",
                    position=0,
                    disable=not self.show_progress_bar,
                )
            ):
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }

                teacher_indices = self._teacher_indices_for_batch(batch_idx, inner_epoch)
                loss_info["teacher_idx"].append(
                    torch.as_tensor(float(teacher_indices[0]), device=device)
                )

                x = self._sample_initial_latents(batch, seed_offset=inner_epoch).float()
                loss_info = self._optimize_train_pass(
                    batch=batch,
                    x=x,
                    timesteps=timesteps,
                    num_steps=num_steps,
                    teacher_indices=teacher_indices,
                    loss_info=loss_info,
                )

    def _optimize_train_pass(
        self,
        batch: Dict[str, Any],
        x: torch.Tensor,
        timesteps: torch.Tensor,
        num_steps: int,
        teacher_indices: List[int],
        loss_info: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, List[torch.Tensor]]:
        """Differentiable Euler rollout: per-step loss + backward + accumulate.

        Matches the GRPO / SDE-OPD per-timestep ``accumulate`` pattern so
        activations from earlier Euler steps can be freed before later steps
        when ``bptt_steps`` caps the segment length. Full BPTT
        (``bptt_steps is None``) uses ``retain_graph=True`` within each
        segment of length ``num_steps``; peak memory remains O(N) unless
        ``solver_checkpointing`` is enabled.
        """
        device = self.accelerator.device
        segment_len = self.bptt_steps if self.bptt_steps is not None else num_steps

        for j in range(num_steps):
            if self.bptt_steps is not None and j > 0 and j % self.bptt_steps == 0:
                x = x.detach()

            t = timesteps[j].to(device)
            t_next = timesteps[j + 1].to(device) if j + 1 < num_steps else torch.zeros_like(t)
            dt_scalar = (t_next - t) / 1000.0

            with self.accelerator.accumulate(*self.adapter.trainable_components):
                with self.autocast():
                    v_student, x_next = self._student_step(x, t, t_next, batch)
                    v_teacher = self._teacher_velocity(x, t, t_next, batch, teacher_indices)

                    dt_sq = dt_scalar.float().pow(2)
                    d_j = (
                        0.5
                        * dt_sq
                        * (v_student.float() - v_teacher.float()).pow(2).flatten(1).mean(dim=1)
                    )
                    loss_j = self.pathwise_coef * d_j.mean()
                    loss_info["d_j"].append(d_j.mean().detach())

                    if self.enable_kl_loss:
                        kl_div, kl_loss = self._compute_kl_anchor_ode(
                            student_noise_pred=v_student,
                            student_next_mean=x_next,
                            x=x,
                            t=t,
                            t_next=t_next,
                            dt_scalar=dt_scalar,
                            batch=batch,
                        )
                        loss_j = loss_j + kl_loss
                        loss_info["kl_div"].append(kl_div.detach())
                        loss_info["kl_loss"].append(kl_loss.detach())

                    loss_info["loss"].append(loss_j.detach())

                at_segment_end = (j + 1) % segment_len == 0 or j == num_steps - 1
                retain_graph = not at_segment_end

                self.accelerator.backward(loss_j, retain_graph=retain_graph)
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

            x = x_next.float()

        return loss_info
