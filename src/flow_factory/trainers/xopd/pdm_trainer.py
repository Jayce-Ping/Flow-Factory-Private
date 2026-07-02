# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Pixel/latent-space one-step DENOISER MATCHING trainer.

Implements docs/mof/pixel_denoiser_matching.tex: an on-policy clean image ``x`` is
encoded / noised to level ``t`` / one-step-denoised (``x0 = z_t - sigma * v``) / (optionally)
decoded by EACH model's own VAE+denoiser; the student is trained to match the teacher's
one-step clean prediction. No latent transport, no L1 (a single L0-style epoch loop).

  * ``rollout_ratio``  : on-policy image source = student:teacher fraction (1=student, 0=teacher).
  * ``pdm_match_space``: 'latent' (SAME-VAE, shared-noise clean velocity matching, cheap) or
                         'pixel' (CROSS-VAE, decode both, MSE in [0,1]).

Subclasses :class:`XOPDTrainer` to reuse the cross-VAE teacher init, both VAEs, rollout
helpers, text conditioning and eval. Registry key ``'xpdm'``.
"""
import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ...utils.base import filter_kwargs
from ...utils.dist import reduce_loss_info
from ...utils.logger_utils import setup_logger
from ...utils.noise_schedule import flow_match_sigma
from .trainer import XOPDTrainer

logger = setup_logger(__name__)


class XPDMTrainer(XOPDTrainer):
    """One-step denoiser matching in latent (same-VAE) or pixel (cross-VAE) space."""

    # Custom (non-L1) optimize loop -> skip XOPD's L1 one-step-per-epoch GAS invariant.
    _validates_l1_one_step = False

    # ---------------------------- main loop (L0-only) ----------------------------
    def start(self):
        """Single denoiser-matching loop: no transport warm-up, no L0/L1 switch."""
        self._check_match_space()
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

            self._pdm_epoch()

            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    def prepare_feedback(self, samples: List[Any]) -> None:  # unused (no reward feedback)
        pass

    def _check_match_space(self) -> None:
        """pixel needs a cross-VAE teacher_adapter; latent needs an IDENTICAL VAE both sides.

        Same-arch teacher (``teacher_model_type`` unset -> shared VAE + transformer swap, e.g.
        dev-32B -> klein-4B) has no ``teacher_adapter`` and always matches in the shared latent.
        """
        if self.training_args.pdm_match_space == "pixel":
            if not self._cross_vae:
                raise ValueError(
                    "pdm_match_space='pixel' needs a cross-VAE teacher_adapter, but this is a "
                    "same-arch (shared-VAE) teacher. Use pdm_match_space='latent'."
                )
            return
        # latent
        if self._cross_vae:
            c_t = int(self.teacher_adapter.pipeline.vae.config.latent_channels)
            c_s = int(self.adapter.pipeline.vae.config.latent_channels)
            if c_t != c_s:
                raise ValueError(
                    "pdm_match_space='latent' needs the SAME VAE on both sides, but teacher VAE "
                    f"latent_channels={c_t} != student {c_s}. Use pdm_match_space='pixel'."
                )

    def _teacher_velocity(self, t, z_t, ids, teacher_cond):
        """Teacher one-step velocity. cross-VAE -> independent teacher_adapter; same-arch
        (shared VAE) -> temporarily swap the teacher transformer into the student pipeline."""
        kw = dict(teacher_cond)
        if ids is not None:
            kw["latent_ids"] = ids
        if self._cross_vae:
            return self.teacher_adapter.predict_velocity(
                t=t, latents=z_t, guidance_scale=self.teacher_gs, **kw
            )
        with self.adapter.use_teacher_transformer():
            return self.adapter.predict_velocity(
                t=t, latents=z_t, guidance_scale=self.teacher_gs, **kw
            )

    # ------------------------- on-policy image sourcing --------------------------
    def _student_rollout_images(self, prompt_batch: Dict[str, Any], n: Optional[int] = None) -> torch.Tensor:
        """Roll out the STUDENT (ODE) and return decoded images ``(n,3,H,W)`` in [0,1]."""
        ta = self.training_args
        self.adapter.rollout()
        infer_kwargs = filter_kwargs(self.adapter.inference, **{
            **ta,
            **prompt_batch,
            "guidance_scale": self.student_gs,
            "num_inference_steps": ta.pdm_num_inference_steps,
            "compute_log_prob": False,
        })
        with torch.no_grad(), self.autocast():
            samples = self.adapter.inference(**infer_kwargs)
        imgs = torch.stack([s.image for s in samples], dim=0)
        return imgs if n is None else imgs[:n]

    def _pdm_rollout_images(self, prompt_batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """On-policy clean images per ``rollout_ratio`` (student vs teacher rollout), detached."""
        ta = self.training_args
        ratio = float(ta.rollout_ratio)
        bsz = int(ta.per_device_batch_size)
        n_stu = int(round(ratio * bsz))
        n_tea = bsz - n_stu
        parts: List[torch.Tensor] = []
        if n_stu > 0:
            parts.append(self._student_rollout_images(prompt_batch, n_stu))
        if n_tea > 0:
            parts.append(self._teacher_rollout_images(prompt_batch)[:n_tea])
        x = torch.cat(parts, dim=0).to(device).float().clamp(0.0, 1.0)
        return x.detach()

    # ----------------------------- denoiser matching -----------------------------
    @staticmethod
    def _split_enc(enc: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """encode_pixels -> (latent, latent_ids|None): SD3.5 returns a tensor, FLUX.2 a tuple."""
        if isinstance(enc, (tuple, list)):
            return enc[0], enc[1]
        return enc, None

    def _sample_pdm_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        ta = self.training_args
        t = self._sample_l0_timesteps(batch_size, device)  # (B,) in [0,1000]
        return t.clamp(ta.pdm_sigma_min * 1000.0, ta.pdm_sigma_max * 1000.0)

    def _pdm_epoch(self) -> None:
        ta = self.training_args
        device = self.accelerator.device
        data_iter = self._make_train_iter()
        match_pixel = ta.pdm_match_space == "pixel"

        for b_idx in tqdm(
            range(ta.num_batches_per_epoch),
            desc=f"Epoch {self.epoch} PDM",
            disable=not self.show_progress_bar,
        ):
            prompt_batch = next(data_iter)
            x = self._pdm_rollout_images(prompt_batch, device)  # (B,3,H,W) [0,1] detached
            batch_size = x.shape[0]
            teacher_cond = self._build_teacher_text_cond(prompt_batch)
            student_prompt_embeds = prompt_batch["prompt_embeds"].to(device)
            student_extra = self._student_cond_from_batch(prompt_batch, device)

            self.adapter.train()
            viz_capture = None
            for inner in range(ta.pdm_inner_steps):
                with self.accelerator.accumulate(*self.adapter.trainable_components):
                    t = self._sample_pdm_timesteps(batch_size, device)
                    sigma = flow_match_sigma(t)

                    if match_pixel:
                        y_A, y_B = self._pixel_branches(
                            x, t, sigma, teacher_cond, student_prompt_embeds, student_extra
                        )
                    else:
                        y_A, y_B = self._latent_branches(
                            x, t, sigma, teacher_cond, student_prompt_embeds, student_extra
                        )

                    # Snapshot this step's ACTUAL sigma + one-step predictions for the recon viz
                    # (logged after the inner loop), so the gallery reflects the real training t.
                    if b_idx == 0 and inner == 0:
                        viz_capture = (sigma.detach(), y_A.detach(), y_B.detach())

                    diff = y_B.float() - y_A.float().detach()
                    loss = diff.pow(2).mean(dim=tuple(range(1, diff.ndim))).mean()

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.adapter.get_trainable_parameters(), ta.max_grad_norm
                        )
                        info = reduce_loss_info(
                            self.accelerator, {"loss": [loss.detach()]}
                        )
                        info["grad_norm"] = grad_norm
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.log_data(
                            {f"train/{k}": v for k, v in info.items()}, step=self.step
                        )
                        self.step += 1

            if viz_capture is not None:
                self._pdm_log_recon(x, *viz_capture, match_pixel)

    def _latent_branches(self, x, t, sigma, teacher_cond, student_prompt_embeds, student_extra):
        """SAME-VAE: encode once, SHARED noise -> compare one-step x0 in the shared latent."""
        z0, ids = self._split_enc(self.adapter.encode_pixels(x))
        sig = sigma.reshape(-1, *([1] * (z0.ndim - 1)))
        z_t = (1.0 - sig) * z0 + sig * torch.randn_like(z0)  # shared noised latent
        with torch.no_grad(), self.autocast():
            v_t = self._teacher_velocity(t, z_t, ids, teacher_cond)  # cross-VAE adapter or same-arch swap
            y_A = z_t - sig * v_t
        with self.autocast():
            kw = dict(student_extra)
            if ids is not None:
                kw["latent_ids"] = ids
            v_s = self.adapter.predict_velocity(
                t=t, latents=z_t, prompt_embeds=student_prompt_embeds,
                guidance_scale=self.student_gs, **kw,
            )
            y_B = z_t - sig * v_s
        return y_A, y_B

    def _pixel_branches(self, x, t, sigma, teacher_cond, student_prompt_embeds, student_extra):
        """CROSS-VAE: each model its OWN VAE + independent noise; compare decoded pixels [0,1]."""
        with torch.no_grad(), self.autocast():
            p_t, ids_t = self._split_enc(self.teacher_adapter.encode_pixels(x))
            s_t = sigma.reshape(-1, *([1] * (p_t.ndim - 1)))
            z_t_t = (1.0 - s_t) * p_t + s_t * torch.randn_like(p_t)
            v_t = self.teacher_adapter.predict_velocity(
                t=t, latents=z_t_t, latent_ids=ids_t, guidance_scale=self.teacher_gs, **teacher_cond
            )
            y_A = self.teacher_adapter.decode_latents(z_t_t - s_t * v_t, ids_t, output_type="pt")
        with self.autocast():
            z0_s, ids_s = self._split_enc(self.adapter.encode_pixels(x))
            s_s = sigma.reshape(-1, *([1] * (z0_s.ndim - 1)))
            z_t_s = (1.0 - s_s) * z0_s + s_s * torch.randn_like(z0_s)
            kw = dict(student_extra)
            if ids_s is not None:
                kw["latent_ids"] = ids_s
            v_s = self.adapter.predict_velocity(
                t=t, latents=z_t_s, prompt_embeds=student_prompt_embeds,
                guidance_scale=self.student_gs, **kw,
            )
            x0_s = z_t_s - s_s * v_s
            if ids_s is not None:
                y_B = self.adapter.decode_latents(x0_s, ids_s, output_type="pt")
            else:
                y_B = self.adapter.decode_latents(x0_s, output_type="pt")
        return y_A, y_B

    @torch.no_grad()
    def _pdm_log_recon(self, x, sigma, y_A, y_B, match_pixel, n=4):
        """Log the on-policy image + teacher/student one-step reconstructions at the ACTUAL
        training sigma of this step. ``y_A``/``y_B`` are that step's one-step x0 predictions
        (pixel: already-decoded images; latent: shared-VAE latents to decode here). The caption
        shows the real per-sample sigma so the gallery reflects the true training noise level."""
        from ...logger.formatting import LogImage
        from PIL import Image as _Image

        n = min(n, x.shape[0])
        xn = x[:n]
        if match_pixel:
            rec_A, rec_B = y_A[:n], y_B[:n]  # already decoded images in [0,1]
        else:
            with self.autocast():
                _, ids = self._split_enc(self.adapter.encode_pixels(xn))
                dec = (lambda z: self.adapter.decode_latents(z, ids, output_type="pt")) if ids is not None \
                    else (lambda z: self.adapter.decode_latents(z, output_type="pt"))
                rec_A, rec_B = dec(y_A[:n]), dec(y_B[:n])
        sig_list = sigma.reshape(-1)[:n].tolist()

        def to_pil(t3):
            a = (t3.float().clamp(0, 1) * 255).round().byte().cpu().permute(1, 2, 0).numpy()
            return _Image.fromarray(a)

        tiles = []
        for i in range(n):
            rows = [to_pil(xn[i]), to_pil(rec_A[i]), to_pil(rec_B[i])]
            w, h = rows[0].size
            tile = _Image.new("RGB", (w, 3 * h), "white")
            for r, im in enumerate(rows):
                tile.paste(im.resize((w, h)), (0, r * h))
            tiles.append(LogImage(tile, caption=f"s{i}: x / teacher_x0 / student_x0 (sigma={sig_list[i]:.3f})"))
        self.log_data({"pdm/recon": tiles}, step=self.step)
