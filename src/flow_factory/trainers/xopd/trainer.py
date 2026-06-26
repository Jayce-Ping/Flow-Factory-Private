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

# src/flow_factory/trainers/xopd/trainer.py
"""Cross-OPD (XOPD): cross-model on-policy distillation trainer.

Standalone trainer (decoupled from OPDTrainer / MoF trainers) for distilling a
larger frozen teacher model into a smaller student that SHARES the VAE, text
encoder, and scheduler (e.g. FLUX.2-klein-base-9B -> FLUX.2-klein-base-4B).

A single run runs two stages, switching on the outer epoch counter:

- **L0** (``epoch < l0_warmup_epochs``): velocity-regression warmup on
  teacher-generated data. The teacher rolls out ``z0`` (no_grad), then the
  student velocity is regressed onto the teacher velocity on the off-policy
  data path ``x_t = (1-sigma) z0 + sigma eps`` with weight ``w(t)``.
- **L1** (``epoch >= l0_warmup_epochs``): on-policy transition matching. The
  student rolls out its own trajectory; per training timestep the closed-form
  Gaussian transition KL ``D_k`` (mean matching) drives the pathwise loss, with
  an optional REINFORCE trajectory term.

The teacher is a SEPARATE full transformer (not a LoRA snapshot), swapped in per
forward via ``adapter.use_teacher_transformer`` (whole-module swap, DDP-bypassed
for no_grad inference). Teacher and student use independent ``guidance_scale``.

Math helpers (``D_k``, reverse-cumulative, forward-kwarg plumbing, L0 weighting)
are copied into :mod:`flow_factory.trainers.xopd.common` so XOPD does not depend
on OPD internals.

Registry key: ``'xopd'`` -> :class:`XOPDTrainer`.
"""

import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ...hparams import XOPDTrainingArguments
from ...samples import BaseSample
from ...utils.base import create_generator, filter_kwargs, stitch_batch_metadata
from ...utils.dist import reduce_loss_info
from ...utils.logger_utils import setup_logger
from ...utils.noise_schedule import TimeSampler, flow_match_sigma
from ...utils.trajectory_collector import compute_trajectory_indices
from ..abc import BaseTrainer
from .common import (
    align_l0_inner_steps,
    build_forward_kwargs,
    cache_forward_signature,
    compute_per_step_kl,
    interleaved_source_iter,
    l0_loss_weight,
    reverse_cumulative,
    validate_l1_one_step_per_epoch,
    validate_source_ratio,
)

logger = setup_logger(__name__)


# Keys reused across student / teacher adapter.forward calls (mirror OPD).
_STUDENT_RETURN_KWARGS = ["log_prob", "next_latents_mean", "std_dev_t", "dt"]
_TEACHER_RETURN_KWARGS = ["next_latents_mean", "std_dev_t", "dt"]


class XOPDTrainer(BaseTrainer):
    """Cross-model on-policy distillation trainer (L0 velocity regression + L1)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: XOPDTrainingArguments
        ta = self.training_args

        self._is_ode = self.adapter.scheduler.dynamics_type == "ODE"
        self.pathwise_coef = ta.pathwise_coef
        self.reinforce_coef = ta.reinforce_coef
        self.reinforce_horizon = ta.reinforce_horizon
        self.reinforce_future_reduction = ta.reinforce_future_reduction
        self.normalize_d_k = ta.normalize_d_k
        self.teacher_gs = ta.teacher_guidance_scale
        self.student_gs = ta.student_guidance_scale

        # Cache adapter.forward signature once for cheap per-step kwarg filtering.
        self._forward_param_names, self._forward_accepts_var_kwargs = cache_forward_signature(
            self.adapter.forward
        )

        # Cross-model teacher requires the adapter's teacher-transformer hooks.
        if not (
            hasattr(self.adapter, "load_teacher_transformer")
            and hasattr(self.adapter, "use_teacher_transformer")
            and hasattr(self.adapter, "predict_velocity")
        ):
            raise TypeError(
                "XOPD requires a cross-model-capable adapter exposing "
                "`load_teacher_transformer`, `use_teacher_transformer`, and "
                f"`predict_velocity` (e.g. Flux2KleinAdapter); got "
                f"{type(self.adapter).__name__}."
            )

        if ta.teacher_param_device == "cpu":
            logger.warning(
                "teacher_param_device='cpu' is not supported for the XOPD teacher "
                "transformer (per-forward H2D is not implemented); loading the "
                "teacher on the compute device instead."
            )
        self.adapter.load_teacher_transformer(
            ta.teacher_model_name_or_path,
            device=self.accelerator.device,
            dtype=self.adapter._inference_dtype,
        )
        # Cross-model XOPD: the teacher transformer expects text embeddings from
        # its OWN encoder (its joint_attention_dim differs from the student's).
        # Those teacher embeddings are PRECOMPUTED during preprocessing and cached
        # (Flux2KleinAdapter.preprocess_func); the teacher text encoder is loaded
        # only for that offline pass and freed afterwards (see _init_dataloader),
        # so it never stays resident during training. This is what makes large
        # teachers viable (e.g. FLUX.2-dev's 48 GB Mistral3 text encoder).

        if (
            self.pathwise_coef == 0
            and self.reinforce_coef == 0
            and ta.kl_beta == 0
            and ta.l0_warmup_epochs == 0
        ):
            logger.warning(
                "XOPDTrainer received a zero-signal config: pathwise_coef="
                f"{self.pathwise_coef}, reinforce_coef={self.reinforce_coef}, "
                f"kl_beta={ta.kl_beta}, l0_warmup_epochs={ta.l0_warmup_epochs}. "
                "The student will not move; set at least one to a positive value."
            )

        # Gradient-accumulation geometry (see docs/opd/cross_opd_xopd.md):
        #   - L1 (on-policy): exactly one optimizer step per epoch, which requires
        #     GAS == num_batches_per_epoch * T and num_inner_epochs == 1.
        #   - L0 (off-policy): l0_inner_steps must be a multiple of T so each L0
        #     epoch ends on a gradient-sync boundary (no L0->L1 leakage); it is
        #     auto-rounded up if not, and L0 then runs l0_inner_steps // T steps/epoch.
        # `T` actually iterated per micro-batch; matches the GAS multiplier
        # (get_num_train_timesteps) unless num_sde_steps > len(sde_steps), in which
        # case the validation below trips with a clear message.
        self.num_train_timesteps = len(self._train_timestep_indices)
        validate_l1_one_step_per_epoch(
            num_batches_per_epoch=ta.num_batches_per_epoch,
            num_train_timesteps=self.num_train_timesteps,
            gradient_accumulation_steps=ta.gradient_accumulation_steps,
            num_inner_epochs=ta.num_inner_epochs,
        )

        self.l0_inner_steps = ta.l0_inner_steps
        if ta.l0_warmup_epochs > 0:
            aligned = align_l0_inner_steps(
                ta.num_batches_per_epoch,
                ta.gradient_accumulation_steps,
                ta.l0_inner_steps,
            )
            if aligned != ta.l0_inner_steps:
                logger.warning(
                    f"XOPD: l0_inner_steps {ta.l0_inner_steps} -> {aligned} (must be a "
                    f"multiple of num_train_timesteps={self.num_train_timesteps}); keeps "
                    "each L0 epoch on a gradient-sync boundary (no L0->L1 leakage)."
                )
            self.l0_inner_steps = aligned
            logger.info(
                "XOPD steps/epoch: L1=1, L0="
                f"{ta.num_batches_per_epoch * self.l0_inner_steps // ta.gradient_accumulation_steps} "
                f"(GAS={ta.gradient_accumulation_steps}, num_batches={ta.num_batches_per_epoch}, "
                f"T={self.num_train_timesteps})."
            )

        # Multi-source training (data.dataset_dirs): the base class builds
        # `self.train_dataloaders_by_source` (one DataLoader per source, each
        # independently preprocessed -- so the teacher text embeddings are
        # precomputed per source). XOPD uses a single teacher, so the source tag
        # only feeds eval/reward metadata; sampling block-cycles across sources
        # via `_make_train_iter`. Fail fast if `source_ratio` cannot tile the
        # per-epoch budget.
        validate_source_ratio(
            ta.source_ratio,
            ta.num_batches_per_epoch,
            self.train_dataloaders_by_source,
        )

    # ============================ Data iteration ==============================
    def _make_train_iter(self):
        """Unified train iterator: single dataloader or block-cycle over sources.

        Single-source (``data.dataset_dir``) yields from ``self.dataloader``.
        Multi-source (``data.dataset_dirs``) block-cycles across
        ``self.train_dataloaders_by_source`` honoring ``source_ratio`` (each
        batch tagged with ``__source__``; infinite cycle with auto-restart).
        """
        if self.train_dataloaders_by_source:
            return interleaved_source_iter(
                self.train_dataloaders_by_source,
                source_ratio=self.training_args.source_ratio,
            )
        if self.dataloader is None:
            raise RuntimeError(
                "XOPD requires training data: set data.dataset_dir (single source) "
                "or data.dataset_dirs (multi-source)."
            )
        return iter(self.dataloader)

    # ============================ Dataloader / preprocess =====================
    def _init_dataloader(self):
        """Load the teacher text encoder only for offline preprocessing, then free it.

        XOPD precomputes the teacher's text embeddings during preprocessing so the
        (potentially very large, e.g. 48 GB for FLUX.2-dev) teacher text encoder is
        never resident during training. The teacher text encoder is loaded right
        before the base preprocessing pass (which runs ``adapter.preprocess_func``
        over the train split) and unloaded immediately after, regardless of
        outcome. Runs inside ``super().__init__()`` (before the teacher transformer
        is loaded), so peak preprocessing memory is teacher TE + student TE + VAE.
        """
        ta = self.training_args
        if not self.config.data_args.enable_preprocess:
            raise ValueError(
                "XOPD requires data.enable_preprocess=True: the teacher text "
                "embeddings are precomputed during preprocessing and cached so the "
                "(large) teacher text encoder can be offloaded before training. "
                "Got enable_preprocess=False."
            )
        self.adapter.load_teacher_text_encoder(
            ta.teacher_model_name_or_path,
            device=self.accelerator.device,
            dtype=self.adapter._inference_dtype,
        )
        try:
            return super()._init_dataloader()
        finally:
            self.adapter.unload_teacher_text_encoder()

    # =============================== Properties ===============================
    @property
    def _train_timestep_indices(self):
        """Training timestep indices: all steps for ODE, scheduler-selected for SDE."""
        if self._is_ode:
            return list(range(self.training_args.num_inference_steps))
        return self.adapter.scheduler.train_timesteps

    @property
    def enable_kl_loss(self) -> bool:
        """KL anchor to the pre-trained base is enabled when ``kl_beta > 0``."""
        return self.training_args.kl_beta > 0.0

    def _student_return_kwargs_for_train(self) -> List[str]:
        keys = list(_STUDENT_RETURN_KWARGS)
        if self.enable_kl_loss and self.training_args.kl_type == "v-based":
            keys.append("noise_pred")
        return keys

    # =============================== Main loop ===============================
    def start(self):
        """Single-run training loop with L0 -> L1 stage switching on ``epoch``."""
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

            if self.epoch < self.training_args.l0_warmup_epochs:
                self._l0_epoch()
            else:
                samples = self.sample()
                self.prepare_feedback(samples)
                self.optimize(samples)

            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """XOPD supervises via teacher KL/velocity; no reward/advantage stage.

        Logs a handful of rollout samples for qualitative inspection (main
        process only), mirroring the GRPO/OPD convention.
        """
        if self.accelerator.is_main_process and samples:
            self.log_data({"train_samples": samples[:30]}, step=self.step)

    # ===================== Stage L0: velocity regression =====================
    def _l0_epoch(self) -> None:
        """L0 warmup: teacher-generated ``z0`` then student velocity regression."""
        ta = self.training_args
        device = self.accelerator.device
        data_iter = self._make_train_iter()
        loss_info: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for batch_idx in tqdm(
            range(ta.num_batches_per_epoch),
            desc=f"Epoch {self.epoch} L0 (velocity regression)",
            disable=not self.show_progress_bar,
        ):
            prompt_batch = next(data_iter)

            # XOPD cross-model: the teacher's own text embeddings are precomputed
            # offline and cached in the batch (teacher_* columns); the student
            # keeps its own precomputed embeddings. No teacher text encoder is
            # resident -- read both directly from the dataloader batch.
            teacher_prompt_embeds = prompt_batch["teacher_prompt_embeds"].to(device)
            teacher_text_ids = prompt_batch["teacher_text_ids"].to(device)
            teacher_neg_embeds = prompt_batch.get("teacher_negative_prompt_embeds")
            teacher_neg_text_ids = prompt_batch.get("teacher_negative_text_ids")
            if teacher_neg_embeds is not None:
                teacher_neg_embeds = teacher_neg_embeds.to(device)
                teacher_neg_text_ids = teacher_neg_text_ids.to(device)

            # 1. Teacher rollout (no_grad, rollout mode) -> clean latent z0.
            self.adapter.rollout()
            infer_kwargs = {
                **ta,
                **prompt_batch,
                "guidance_scale": self.teacher_gs,
                "num_inference_steps": ta.l0_num_inference_steps,
                "compute_log_prob": False,
            }
            # Route the cached teacher conditioning into the teacher rollout
            # (override the student embeds; teacher embeds carry no prompt_ids).
            infer_kwargs["prompt_embeds"] = teacher_prompt_embeds
            infer_kwargs["text_ids"] = teacher_text_ids
            infer_kwargs.pop("prompt_ids", None)
            if teacher_neg_embeds is not None:
                infer_kwargs["negative_prompt_embeds"] = teacher_neg_embeds
                infer_kwargs["negative_text_ids"] = teacher_neg_text_ids
                infer_kwargs.pop("negative_prompt_ids", None)
            else:
                for _k in ("negative_prompt_embeds", "negative_text_ids", "negative_prompt_ids"):
                    infer_kwargs.pop(_k, None)
            infer_kwargs = filter_kwargs(self.adapter.inference, **infer_kwargs)
            with torch.no_grad(), self.autocast(), self.adapter.use_teacher_transformer():
                teacher_samples = self.adapter.inference(**infer_kwargs)

            tb = BaseSample.stack([s.to(device) for s in teacher_samples])
            z0 = tb["all_latents"][:, -1]  # (B, seq_len, C) clean latent
            latent_ids = tb["latent_ids"]
            batch_size = z0.shape[0]
            # Student branch uses student embeds (from the dataloader batch, same
            # prompt order as z0).
            student_prompt_embeds = prompt_batch["prompt_embeds"].to(device)
            student_text_ids = prompt_batch["text_ids"].to(device)
            student_neg_embeds = prompt_batch.get("negative_prompt_embeds")
            student_neg_text_ids = prompt_batch.get("negative_text_ids")
            if student_neg_embeds is not None:
                student_neg_embeds = student_neg_embeds.to(device)
                student_neg_text_ids = student_neg_text_ids.to(device)

            # 2. Student velocity regression (train mode) over random continuous t.
            # self.l0_inner_steps is aligned to a multiple of num_train_timesteps so
            # each L0 epoch ends on a gradient-sync boundary (see __init__).
            self.adapter.train()
            for inner in range(self.l0_inner_steps):
                with self.accelerator.accumulate(*self.adapter.trainable_components):
                    t = self._sample_l0_timesteps(batch_size, device)
                    sigma = flow_match_sigma(t)  # (B,) in [0, 1]
                    sigma_b = sigma.view(batch_size, *([1] * (z0.ndim - 1)))
                    eps = torch.randn_like(z0)
                    z_t = (1.0 - sigma_b) * z0 + sigma_b * eps

                    with self.autocast():
                        with torch.no_grad(), self.adapter.use_teacher_transformer():
                            v_teacher = self.adapter.predict_velocity(
                                t=t,
                                latents=z_t,
                                latent_ids=latent_ids,
                                prompt_embeds=teacher_prompt_embeds,
                                text_ids=teacher_text_ids,
                                negative_prompt_embeds=teacher_neg_embeds,
                                negative_text_ids=teacher_neg_text_ids,
                                guidance_scale=self.teacher_gs,
                            )
                        v_student = self.adapter.predict_velocity(
                            t=t,
                            latents=z_t,
                            latent_ids=latent_ids,
                            prompt_embeds=student_prompt_embeds,
                            text_ids=student_text_ids,
                            negative_prompt_embeds=student_neg_embeds,
                            negative_text_ids=student_neg_text_ids,
                            guidance_scale=self.student_gs,
                        )

                    w = l0_loss_weight(sigma, ta.l0_weighting, ta.l0_snr_gamma)  # (B,)
                    per_sample_mse = (
                        (v_student.float() - v_teacher.float())
                        .pow(2)
                        .mean(dim=tuple(range(1, v_student.ndim)))
                    )  # (B,)
                    loss = (w * per_sample_mse).mean()

                    loss_info["l0_loss"].append(loss.detach())
                    loss_info["l0_t_mean"].append(t.float().mean().detach())

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.adapter.get_trainable_parameters(),
                            ta.max_grad_norm,
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

    def _sample_l0_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Per-sample scheduler-scale timesteps in [0, 1000] for L0 (shape ``(B,)``)."""
        if self.training_args.l0_time_sampling == "uniform":
            t = TimeSampler.uniform(
                batch_size=1,
                num_timesteps=batch_size,
                timestep_range=1.0,
                device=device,
            )
        else:
            t = TimeSampler.logit_normal_shifted(
                batch_size=1,
                num_timesteps=batch_size,
                timestep_range=1.0,
                device=device,
            )
        return t.squeeze(1).to(device)

    # ===================== Stage L1: on-policy distillation =====================
    def sample(self) -> List[BaseSample]:
        """Generate student rollouts (full trajectory + on-policy log-probs)."""
        self.adapter.rollout()
        samples: List[BaseSample] = []
        data_iter = self._make_train_iter()

        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self._train_timestep_indices,
            num_inference_steps=self.training_args.num_inference_steps,
        )

        with torch.no_grad(), self.autocast():
            for _ in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f"Epoch {self.epoch} Sampling (L1)",
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    # ODE rollouts (noise_level=0) collect no per-step log-probs
                    # (flux2_klein._inference gates collection on noise_level>0), and log-probs
                    # feed only the SDE-only REINFORCE term. Requesting them under ODE would
                    # stack an empty list; skip them.
                    "compute_log_prob": not self._is_ode,
                    "trajectory_indices": trajectory_indices,
                    **batch,
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                stitch_batch_metadata(batch, sample_batch)
                self._maybe_offload_samples_to_cpu(sample_batch)
                samples.extend(sample_batch)

        return samples

    def optimize(self, samples: List[BaseSample]) -> None:
        """L1 two-pass per-batch loss: pre-pass D_k/R_bar, then gradient main pass.

        Single pass over the rollout (``num_inner_epochs`` is enforced to 1 in
        ``__init__``): with ``GAS == num_batches * num_train_timesteps`` this performs
        exactly one on-policy optimizer step per epoch.
        """
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size

        self.adapter.train()

        perm_gen = create_generator(self.training_args.seed, self.epoch)
        perm = torch.randperm(len(samples), generator=perm_gen)
        shuffled_samples = [samples[i] for i in perm]

        loss_info: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for batch_idx in tqdm(
            range(num_batches),
            total=num_batches,
            desc=f"Epoch {self.epoch} Training (L1)",
            position=0,
            disable=not self.show_progress_bar,
        ):
            start = batch_idx * per_device_batch_size
            end = min(start + per_device_batch_size, len(shuffled_samples))
            batch_samples = [shuffled_samples[i].to(device) for i in range(start, end)]
            batch = BaseSample.stack(batch_samples)
            latents_index_map = batch["latent_index_map"]
            num_timesteps = batch["timesteps"].shape[1]

            d_list, mu_teacher_list = self._precompute_d_per_timestep(
                batch=batch,
                latents_index_map=latents_index_map,
                num_timesteps=num_timesteps,
            )

            if self.reinforce_coef > 0:
                r_per_k = reverse_cumulative(
                    d_list,
                    self.reinforce_horizon,
                    reduction=self.reinforce_future_reduction,
                )
            else:
                r_per_k = [torch.zeros_like(d) for d in d_list]

            loss_info = self._optimize_train_pass(
                batch=batch,
                latents_index_map=latents_index_map,
                num_timesteps=num_timesteps,
                mu_teacher_list=mu_teacher_list,
                r_per_k=r_per_k,
                loss_info=loss_info,
            )

    # =============================== L1 helpers ===============================
    def _build_forward_kwargs(
        self,
        batch: Dict[str, Any],
        t: torch.Tensor,
        t_next: torch.Tensor,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        compute_log_prob: bool,
        return_kwargs: List[str],
        guidance_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        forward_kwargs = build_forward_kwargs(
            training_args=self.training_args,
            batch=batch,
            t=t,
            t_next=t_next,
            latents=latents,
            next_latents=next_latents,
            compute_log_prob=compute_log_prob,
            noise_level=self.adapter.scheduler.noise_level,
            return_kwargs=return_kwargs,
            param_names=self._forward_param_names,
            accepts_var_kwargs=self._forward_accepts_var_kwargs,
        )
        if guidance_scale is not None:
            forward_kwargs["guidance_scale"] = guidance_scale
        return forward_kwargs

    def _teacher_next_latents_mean(
        self,
        forward_kwargs: Dict[str, Any],
        teacher_text_cond: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Teacher transition mean via a whole-module transformer swap (no_grad).

        For cross-model XOPD, ``teacher_text_cond`` (from ``encode_teacher_prompt``)
        replaces the student-encoded text conditioning so the teacher transformer
        receives embeddings of its own ``joint_attention_dim``.
        """
        teacher_kwargs = dict(forward_kwargs)
        teacher_kwargs["guidance_scale"] = self.teacher_gs
        if teacher_text_cond is not None:
            teacher_kwargs["prompt_embeds"] = teacher_text_cond["prompt_embeds"]
            teacher_kwargs["text_ids"] = teacher_text_cond["text_ids"]
            if "negative_prompt_embeds" in teacher_text_cond:
                teacher_kwargs["negative_prompt_embeds"] = teacher_text_cond[
                    "negative_prompt_embeds"
                ]
                teacher_kwargs["negative_text_ids"] = teacher_text_cond["negative_text_ids"]
            else:
                # teacher_gs == 1.0: no CFG; drop any student-dim negatives.
                teacher_kwargs.pop("negative_prompt_embeds", None)
                teacher_kwargs.pop("negative_text_ids", None)
        with self.adapter.use_teacher_transformer():
            out = self.adapter.forward(**teacher_kwargs)
        if out.next_latents_mean is None:
            raise RuntimeError(
                "Teacher forward did not return `next_latents_mean`; "
                f"check return_kwargs={teacher_kwargs.get('return_kwargs')!r}."
            )
        return out.next_latents_mean.detach()

    def _precompute_d_per_timestep(
        self,
        batch: Dict[str, Any],
        latents_index_map: torch.Tensor,
        num_timesteps: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """No-grad pre-pass: per-timestep ``D_k`` and cached teacher mean."""
        d_list: List[torch.Tensor] = []
        mu_teacher_list: List[torch.Tensor] = []
        device = self.accelerator.device

        with torch.no_grad(), self.autocast():
            # XOPD cross-model: the teacher's own text embeddings are precomputed
            # offline and carried on the rollout samples (teacher_* fields), so the
            # stacked batch already holds them (constant across timesteps); no
            # teacher text encoder is resident.
            teacher_text_cond = {
                "prompt_embeds": batch["teacher_prompt_embeds"],
                "text_ids": batch["teacher_text_ids"],
            }
            if self.teacher_gs > 1.0:
                teacher_text_cond["negative_prompt_embeds"] = batch[
                    "teacher_negative_prompt_embeds"
                ]
                teacher_text_cond["negative_text_ids"] = batch["teacher_negative_text_ids"]
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
                    guidance_scale=self.student_gs,
                )

                student_out = self.adapter.forward(**forward_kwargs)
                if student_out.next_latents_mean is None:
                    raise RuntimeError(
                        "Student forward did not return `next_latents_mean` during "
                        f"pre-pass; requested return_kwargs={_TEACHER_RETURN_KWARGS!r}."
                    )

                mu_teacher = self._teacher_next_latents_mean(
                    forward_kwargs, teacher_text_cond
                )

                d_k = compute_per_step_kl(
                    mu_student=student_out.next_latents_mean,
                    mu_teacher=mu_teacher,
                    std_dev_t=student_out.std_dev_t,
                    dt=student_out.dt,
                    normalize=self.normalize_d_k,
                )
                d_list.append(d_k.detach())
                mu_teacher_list.append(mu_teacher)

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
        """Gradient main pass: per-timestep student forward + loss + backward."""
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
                        # See sample(): no log-probs under ODE (unused without REINFORCE).
                        compute_log_prob=not self._is_ode,
                        return_kwargs=self._student_return_kwargs_for_train(),
                        guidance_scale=self.student_gs,
                    )

                    student_out = self.adapter.forward(**forward_kwargs)
                    if student_out.next_latents_mean is None:
                        raise RuntimeError(
                            "Student forward must return `next_latents_mean` for XOPD; got None."
                        )

                    mu_teacher = mu_teacher_list[k_idx]

                    d_k_grad = compute_per_step_kl(
                        mu_student=student_out.next_latents_mean,
                        mu_teacher=mu_teacher,
                        std_dev_t=student_out.std_dev_t,
                        dt=student_out.dt,
                        normalize=self.normalize_d_k,
                    )

                    pathwise_loss = d_k_grad.mean()

                    r_kp1 = r_per_k[k_idx].detach()
                    log_prob_new = student_out.log_prob
                    if self.reinforce_coef > 0 and log_prob_new is not None:
                        reinforce_loss = (r_kp1 * log_prob_new).mean()
                    else:
                        reinforce_loss = torch.zeros((), device=device)

                    loss = self.pathwise_coef * pathwise_loss + self.reinforce_coef * reinforce_loss

                    if self.enable_kl_loss:
                        kl_div, kl_loss = self._compute_kl_anchor(student_out, forward_kwargs)
                        loss = loss + kl_loss
                        loss_info["kl_div"].append(kl_div.detach())
                        loss_info["kl_loss"].append(kl_loss.detach())

                    loss_info["d_k"].append(pathwise_loss.detach())
                    loss_info["r_bar"].append(r_kp1.mean().detach())
                    if log_prob_new is not None:
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

    def _compute_kl_anchor(
        self,
        student_out: Any,
        forward_kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optional KL anchor to the pre-trained base (``kl_beta > 0``).

        ``x-based`` reuses the Gaussian transition KL (same scale as ``D_k``);
        ``v-based`` is an unscaled velocity MSE. The reference forward runs under
        ``use_ref_parameters`` (LoRA-off / base snapshot), no_grad.
        """
        kl_type = self.training_args.kl_type
        if kl_type == "v-based":
            ref_return_kwargs = ["noise_pred"]
        else:
            ref_return_kwargs = ["next_latents_mean", "std_dev_t", "dt"]

        with torch.no_grad(), self.adapter.use_ref_parameters():
            ref_kwargs = dict(forward_kwargs)
            ref_kwargs["compute_log_prob"] = False
            ref_kwargs["return_kwargs"] = ref_return_kwargs
            ref_kwargs["guidance_scale"] = self.student_gs
            ref_out = self.adapter.forward(**ref_kwargs)

        if kl_type == "v-based":
            if student_out.noise_pred is None or ref_out.noise_pred is None:
                raise RuntimeError(
                    "v-based KL requires `noise_pred` from both student and reference."
                )
            kl_div = torch.mean(
                (student_out.noise_pred - ref_out.noise_pred) ** 2,
                dim=tuple(range(1, student_out.noise_pred.ndim)),
            ).mean()
        else:
            if student_out.next_latents_mean is None or ref_out.next_latents_mean is None:
                raise RuntimeError(
                    "x-based KL requires `next_latents_mean` from both student and reference."
                )
            kl_div = compute_per_step_kl(
                mu_student=student_out.next_latents_mean,
                mu_teacher=ref_out.next_latents_mean,
                std_dev_t=student_out.std_dev_t,
                dt=student_out.dt,
                normalize=self.normalize_d_k,
            ).mean()

        kl_loss = self.training_args.kl_beta * kl_div
        return kl_div, kl_loss
