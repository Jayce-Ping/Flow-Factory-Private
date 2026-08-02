# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Trajectory distribution-matching trainers (Approach B / TDM).

Design:
  * docs/xopd/approach_b_trajectory_dm_design.md  (``xopd_dm``)
  * docs/xopd/tdm_cross_model_design.md           (``xtdm``)
  * docs/xopd/tdm_opd_dm_gradient_relation.tex

Shared base :class:`XTrajectoryDMTrainer` owns the fake LoRA + two-timescale loop.
Subclasses differ in ODE grid, τ-band, and generator loss metric.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from peft import PeftModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu

from ...scheduler import set_scheduler_timesteps
from ...utils.dist import reduce_loss_info
from ...utils.logger_utils import setup_logger
from .trainer import XOPDTrainer
from .traj_dm import (
    add_noise_rf,
    broadcast_int,
    dm_stopgrad_loss,
    flow_match_sigma,
    ode_euler_next,
    pseudo_huber_c_default,
    pseudo_huber_from_residual,
    self_normalized_dm_grad,
    uniform_scheduler_t,
    x0_from_velocity,
)

logger = setup_logger(__name__)


class XTrajectoryDMTrainer(XOPDTrainer):
    """Shared base for Approach B / TDM: online fake + ODE traj DM (not registered)."""

    _validates_l1_one_step = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from ...hparams.training_args import XTrajectoryDMTrainingArguments

        self.training_args: XTrajectoryDMTrainingArguments
        if getattr(self, "_cross_vae", False):
            raise NotImplementedError(
                f"{type(self).__name__} v1 supports same-arch (shared-VAE) teachers only; "
                "cross-VAE trajectory DM is out of scope."
            )
        self._tdm_setup_fake_adapter()

    # ------------------------------------------------------------------ setup ----
    def _tdm_setup_fake_adapter(self) -> None:
        """Add ``fake`` LoRA after ``accelerator.prepare`` + manual-DP AdamW (XDMD pattern)."""
        ta = self.training_args
        transformer = self.accelerator.unwrap_model(self.adapter.transformer)
        if not isinstance(transformer, PeftModel):
            raise TypeError(
                f"{type(self).__name__} requires finetune_type='lora' (PeftModel student); "
                f"got {type(transformer).__name__}."
            )
        if "default" not in transformer.peft_config:
            raise RuntimeError(
                f"{type(self).__name__}: expected default LoRA adapter; found "
                f"{list(transformer.peft_config)!r}."
            )
        if "fake" not in transformer.peft_config:
            transformer.add_adapter("fake", transformer.peft_config["default"])
            logger.info("%s: added 'fake' LoRA adapter.", type(self).__name__)
        self._tdm_transformer = transformer

        transformer.set_adapter("fake")
        fake_params = [p for p in transformer.parameters() if p.requires_grad]
        if not fake_params:
            raise RuntimeError(f"{type(self).__name__}: 'fake' adapter has no trainable params.")

        self._world_size = int(self.accelerator.num_processes)
        if self._world_size > 1 and dist.is_initialized():
            for p in fake_params:
                dist.broadcast(p.data, src=0)

        self._fake_params = fake_params
        self._opt_fake = torch.optim.AdamW(
            fake_params,
            lr=ta.tdm_fake_lr,
            betas=ta.adam_betas,
            weight_decay=ta.adam_weight_decay,
            eps=ta.adam_epsilon,
        )
        transformer.set_adapter("default")

        n = int(ta.num_batches_per_epoch)
        if n < 1:
            raise ValueError(f"{type(self).__name__} needs num_batches_per_epoch >= 1, got {n}.")
        if n != int(ta.tdm_fake_ratio):
            logger.warning(
                "%s: effective fake:gen = num_batches_per_epoch (%d) != tdm_fake_ratio (%d). "
                "Set unique_sample_num_per_epoch = tdm_fake_ratio * per_device_batch_size * "
                "num_processes to match.",
                type(self).__name__,
                n,
                ta.tdm_fake_ratio,
            )
        logger.info(
            "%s ready: fake (%d tensors) manual-DP AdamW(lr=%g); fake:gen=%d:1.",
            type(self).__name__,
            len(fake_params),
            ta.tdm_fake_lr,
            n,
        )

    @contextmanager
    def _tdm_unwrapped_transformer(self):
        prev = self.adapter.get_component("transformer")
        self.adapter.set_component("transformer", self._tdm_transformer)
        try:
            yield
        finally:
            self.adapter.set_component("transformer", prev)

    def _tdm_set_adapter(self, name: str) -> None:
        self._tdm_transformer.set_adapter(name)

    def _tdm_resolution(self) -> Tuple[int, int]:
        ta = self.training_args
        if ta.height and ta.width:
            return int(ta.height), int(ta.width)
        res = ta.resolution
        if isinstance(res, (list, tuple)):
            h = int(res[0])
            w = int(res[1]) if len(res) > 1 else h
            return h, w
        return int(res), int(res)

    def _tdm_num_ode_steps(self) -> int:
        raise NotImplementedError

    def _tdm_tau_frac_band(self, t_cur: torch.Tensor, t_next: torch.Tensor) -> Tuple[float, float]:
        """Return ``(lo_frac, hi_frac)`` in ``[0,1]`` for DM noise sampling."""
        raise NotImplementedError

    def _tdm_generator_metric(self, x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        ta = self.training_args
        metric = str(ta.tdm_loss_metric)
        if metric == "mse":
            return dm_stopgrad_loss(x, grad)
        if metric == "pseudo_huber":
            c = ta.tdm_pseudo_huber_c
            if c is None:
                c = pseudo_huber_c_default(int(x[0].numel()))
            residual = x.float() - (x.float() - grad.float()).detach()
            return pseudo_huber_from_residual(residual, float(c))
        raise ValueError(
            f"tdm_loss_metric must be 'mse' or 'pseudo_huber', got {metric!r}"
        )

    # -------------------------------------------------------------- main loop ----
    def start(self):
        if self.epoch == 0 and self.training_args.eval_teacher_at_start:
            self.evaluate_teacher_baseline()

        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)
            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir, str(self.log_args.run_name), "checkpoints"
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)
            if self.eval_args.eval_freq > 0 and self.epoch % self.eval_args.eval_freq == 0:
                self.evaluate()

            # Optional L0 warmup (often 0 for pure DM).
            if self.epoch < int(self.training_args.l0_warmup_epochs):
                self._l0_epoch()
            else:
                self._tdm_epoch()

            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    def prepare_feedback(self, samples):
        pass

    def _evaluate_validation_d_k(self) -> None:
        """No-op: trajectory DM is not an L1 Gaussian transition-KL objective."""
        return

    def _tdm_epoch(self) -> None:
        ta = self.training_args
        device = self.accelerator.device
        data_iter = self._make_train_iter()
        n = int(ta.num_batches_per_epoch)

        for b_idx in tqdm(
            range(n),
            desc=f"Epoch {self.epoch} {type(self).__name__}",
            disable=not self.show_progress_bar,
        ):
            prompt_batch = next(data_iter)
            info: Dict[str, Any] = {}
            if b_idx == 0:
                loss_dm, x0_fake_tgt, latent_ids, pe, text_ids = self._tdm_generator_step(
                    prompt_batch, device
                )
                info["loss_dm"] = loss_dm
            else:
                x0_fake_tgt, latent_ids, pe, text_ids = self._tdm_sample_for_fake(
                    prompt_batch, device
                )

            info.update(
                self._tdm_fake_step(x0_fake_tgt, latent_ids, pe, text_ids, device)
            )
            self.log_data({f"train/{k}": v for k, v in info.items()}, step=self.step)
            self.step += 1

    # -------------------------------------------------------------- ODE rollout --
    def _tdm_prepare_noise(
        self, prompt_batch: Dict[str, Any], device: torch.device, num_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pe = prompt_batch["prompt_embeds"].to(device)
        text_ids = prompt_batch["text_ids"].to(device)
        batch_size = pe.shape[0]
        height, width = self._tdm_resolution()
        num_channels_latents = self.adapter.pipeline.transformer.config.in_channels // 4
        z, latent_ids = self.adapter.pipeline.prepare_latents(
            batch_size=batch_size,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=pe.dtype,
            device=device,
            generator=None,
            latents=None,
        )
        z = self.adapter.cast_latents(z)
        mu = compute_empirical_mu(image_seq_len=z.shape[1], num_steps=num_steps)
        timesteps = set_scheduler_timesteps(
            scheduler=self.adapter.pipeline.scheduler,
            num_inference_steps=num_steps,
            device=device,
            mu=mu,
        )
        return z, latent_ids, pe, text_ids, timesteps

    def _tdm_t_pair(
        self, timesteps: torch.Tensor, sel: int, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = int(timesteps.shape[0])
        if not (0 <= sel < n):
            raise ValueError(f"sel={sel} out of range for num_steps={n}")
        t_cur = timesteps[sel].repeat(batch_size).to(device)
        if sel + 1 < n:
            t_next = timesteps[sel + 1].repeat(batch_size).to(device)
        else:
            t_next = torch.zeros(batch_size, device=device, dtype=timesteps.dtype)
        return t_cur, t_next

    def _tdm_ode_step(
        self,
        z: torch.Tensor,
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
        latent_ids: torch.Tensor,
        pe: torch.Tensor,
        text_ids: torch.Tensor,
        *,
        with_grad: bool,
    ) -> torch.Tensor:
        """One deterministic Euler step; optionally keeps the graph on ``z_next``."""
        gs = float(self.training_args.student_guidance_scale)
        grad_ctx = nullcontext() if with_grad else torch.no_grad()
        with grad_ctx, self.autocast():
            v = self.adapter.predict_velocity(
                t=t_cur,
                latents=z,
                latent_ids=latent_ids,
                prompt_embeds=pe,
                text_ids=text_ids,
                guidance_scale=gs,
            )
            z_next = ode_euler_next(z, v, t_cur, t_next)
        if not with_grad:
            z_next = z_next.detach()
        return z_next

    def _tdm_rollout_to_sel(
        self,
        prompt_batch: Dict[str, Any],
        device: torch.device,
        *,
        with_grad: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """ODE prefix (no_grad) + one step at ``sel`` (optional grad). Returns state after that step."""
        ta = self.training_args
        num_steps = self._tdm_num_ode_steps()
        self._tdm_set_adapter("default")
        if with_grad:
            self.adapter.train()
        else:
            self.adapter.rollout()

        z, latent_ids, pe, text_ids, timesteps = self._tdm_prepare_noise(
            prompt_batch, device, num_steps
        )
        batch_size = pe.shape[0]
        sel = int(torch.randint(0, num_steps, (1,), device=device).item())
        sel = broadcast_int(sel, device, self._world_size, src=0)

        with torch.no_grad():
            for i in range(sel):
                t_cur, t_next = self._tdm_t_pair(timesteps, i, batch_size, device)
                z = self._tdm_ode_step(
                    z, t_cur, t_next, latent_ids, pe, text_ids, with_grad=False
                )

        t_cur, t_next = self._tdm_t_pair(timesteps, sel, batch_size, device)
        z = self._tdm_ode_step(
            z, t_cur, t_next, latent_ids, pe, text_ids, with_grad=with_grad
        )
        return z, latent_ids, pe, text_ids, timesteps, sel

    def _tdm_sample_for_fake(
        self, prompt_batch: Dict[str, Any], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full no_grad ODE to data; use final state as FM clean target for fake."""
        num_steps = self._tdm_num_ode_steps()
        self._tdm_set_adapter("default")
        self.adapter.rollout()
        z, latent_ids, pe, text_ids, timesteps = self._tdm_prepare_noise(
            prompt_batch, device, num_steps
        )
        batch_size = pe.shape[0]
        with torch.no_grad():
            for i in range(num_steps):
                t_cur, t_next = self._tdm_t_pair(timesteps, i, batch_size, device)
                z = self._tdm_ode_step(
                    z, t_cur, t_next, latent_ids, pe, text_ids, with_grad=False
                )
        return z.detach(), latent_ids, pe, text_ids

    def _tdm_teacher_cond(
        self, prompt_batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        ta = self.training_args
        cond = {
            "prompt_embeds": prompt_batch["teacher_prompt_embeds"].to(device),
            "text_ids": prompt_batch["teacher_text_ids"].to(device),
        }
        if ta.tdm_real_guidance_scale > 1.0:
            if "teacher_negative_prompt_embeds" not in prompt_batch:
                raise KeyError(
                    "tdm_real_guidance_scale > 1 needs teacher_negative_prompt_embeds; "
                    "precompute them or keep tdm_real_guidance_scale: 1.0."
                )
            cond["negative_prompt_embeds"] = prompt_batch["teacher_negative_prompt_embeds"].to(
                device
            )
            cond["negative_text_ids"] = prompt_batch["teacher_negative_text_ids"].to(device)
        return cond

    def _tdm_generator_step(
        self, prompt_batch: Dict[str, Any], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ta = self.training_args
        teacher_cond = self._tdm_teacher_cond(prompt_batch, device)

        with self.accelerator.accumulate(*self.adapter.trainable_components):
            x_ti, latent_ids, pe, text_ids, timesteps, sel = self._tdm_rollout_to_sel(
                prompt_batch, device, with_grad=True
            )
            batch_size = x_ti.shape[0]
            t_cur, t_next = self._tdm_t_pair(timesteps, sel, batch_size, device)
            lo_frac, hi_frac = self._tdm_tau_frac_band(t_cur, t_next)

            with torch.no_grad():
                t_tau = uniform_scheduler_t(batch_size, device, lo_frac, hi_frac)
                # Treat trajectory state as clean for RF re-noise (DMD2 / TDM engineering).
                z_dm, _ = add_noise_rf(x_ti.detach(), t_tau)

                with self.autocast(), self.adapter.use_teacher_transformer():
                    v_real = self.adapter.predict_velocity(
                        t=t_tau,
                        latents=z_dm,
                        latent_ids=latent_ids,
                        guidance_scale=ta.tdm_real_guidance_scale,
                        **teacher_cond,
                    )
                x0_real = x0_from_velocity(z_dm, v_real, t_tau)

                self._tdm_set_adapter("fake")
                with self.autocast():
                    v_fake = self.adapter.predict_velocity(
                        t=t_tau,
                        latents=z_dm,
                        latent_ids=latent_ids,
                        prompt_embeds=pe,
                        text_ids=text_ids,
                        guidance_scale=1.0,
                    )
                self._tdm_set_adapter("default")
                x0_fake = x0_from_velocity(z_dm, v_fake, t_tau)

                # Score residual in x0 form relative to the matched state x_ti.
                p_real = (x_ti - x0_real).float()
                p_fake = (x_ti - x0_fake).float()
                grad = self_normalized_dm_grad(p_real, p_fake)

            loss_dm = self._tdm_generator_metric(x_ti, grad)

            self.accelerator.backward(loss_dm)
            grad_norm = None
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.adapter.get_trainable_parameters(), ta.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._opt_fake.zero_grad(set_to_none=True)

        info = reduce_loss_info(self.accelerator, {"loss_dm": [loss_dm.detach()]})
        self.log_data(
            {
                "train/gen_grad_norm": grad_norm if grad_norm is not None else 0.0,
                "train/tdm_grad_abs_mean": grad.abs().mean().detach(),
                "train/tdm_sel": float(sel),
            },
            step=self.step,
        )
        return info["loss_dm"], x_ti.detach(), latent_ids, pe, text_ids

    def _tdm_fake_step(
        self,
        x0: torch.Tensor,
        latent_ids: torch.Tensor,
        pe: torch.Tensor,
        text_ids: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, Any]:
        ta = self.training_args
        self._tdm_set_adapter("fake")
        self.adapter.train()

        batch_size = x0.shape[0]
        t = uniform_scheduler_t(batch_size, device, 0.0, 1.0)
        z_t, eps = add_noise_rf(x0.detach(), t)
        v_target = eps - x0.detach()

        with self._tdm_unwrapped_transformer(), self.autocast():
            v_fake = self.adapter.predict_velocity(
                t=t,
                latents=z_t,
                latent_ids=latent_ids,
                prompt_embeds=pe,
                text_ids=text_ids,
                guidance_scale=float(ta.tdm_fake_guidance_scale),
            )
        diff = v_fake.float() - v_target.float()
        loss = diff.pow(2).mean(dim=tuple(range(1, diff.ndim))).mean()

        self._opt_fake.zero_grad(set_to_none=True)
        loss.backward()
        if self._world_size > 1 and dist.is_initialized():
            for p in self._fake_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= self._world_size
        grad_norm = torch.nn.utils.clip_grad_norm_(self._fake_params, ta.tdm_grad_norm)
        self._opt_fake.step()
        self._opt_fake.zero_grad(set_to_none=True)
        self._tdm_set_adapter("default")

        info = reduce_loss_info(self.accelerator, {"loss_fake": [loss.detach()]})
        info["fake_grad_norm"] = grad_norm
        info["x0_std"] = x0.float().std()
        return info


class XOPDDMTrainer(XTrajectoryDMTrainer):
    """Approach B: OPD ODE grid + score-diff force on ``x_ti`` (``trainer_type: xopd_dm``)."""

    def _tdm_num_ode_steps(self) -> int:
        return int(self.training_args.num_inference_steps)

    def _tdm_tau_frac_band(
        self, t_cur: torch.Tensor, t_next: torch.Tensor
    ) -> Tuple[float, float]:
        ta = self.training_args
        return float(ta.tdm_t_min), float(ta.tdm_t_max)


class XTDMTrainer(XTrajectoryDMTrainer):
    """Paper TDM: K-step ODE, non-overlapping τ intervals, Pseudo-Huber (``xtdm``)."""

    def _tdm_num_ode_steps(self) -> int:
        from ...hparams.training_args import XTDMTrainingArguments

        ta: XTDMTrainingArguments = self.training_args  # type: ignore[assignment]
        return int(ta.tdm_sim_steps)

    def _tdm_tau_frac_band(
        self, t_cur: torch.Tensor, t_next: torch.Tensor
    ) -> Tuple[float, float]:
        """Non-overlapping band: σ ∈ (σ(t_next), σ(t_cur)] clamped to ``[tdm_t_min, tdm_t_max]``."""
        ta = self.training_args
        sig_hi = float(flow_match_sigma(t_cur[0]).item())
        sig_lo = float(flow_match_sigma(t_next[0]).item())
        if sig_lo > sig_hi:
            sig_lo, sig_hi = sig_hi, sig_lo
        lo = max(float(ta.tdm_t_min), sig_lo)
        hi = min(float(ta.tdm_t_max), sig_hi)
        if lo >= hi:
            # Degenerate last step / clamp: fall back to global band.
            lo, hi = float(ta.tdm_t_min), float(ta.tdm_t_max)
        # Ensure a tiny open interval for Uniform.
        if hi - lo < 1e-4:
            hi = min(1.0, lo + 1e-4)
        return lo, hi
