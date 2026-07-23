# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Cross-model Distribution Matching Distillation (DMD) trainer.

Design: docs/xopd/dmd_cross_model_design.md. Reference: tianweiy/DMD2 (strictly aligned;
see docs/xopd/progress/2026-07-23-dmd-implementation-report.md for the DMD2 mapping).

Roles (pure DMD, no MSE flow-anchor, no GAN head in v1):
  * ``real``    = the frozen 32B flux2-dev teacher (XOPD's ``use_teacher_transformer`` swap).
                  flux2-dev is guidance-DISTILLED and the transformer takes ``guidance=None``, so
                  ``dmd_real_guidance_scale=1.0`` means a SINGLE conditional pass (no CFG, no
                  negatives): the student fits the teacher's base distribution, NOT a CFG-amplified
                  one. ``guidance_scale`` still flows through the clean ``predict_velocity``
                  interface, so >1 (CFG double-pass) is available if teacher negatives are provided.
  * ``fake``    = a SECOND LoRA adapter on the SAME frozen klein-4B base -- an ONLINE score of the
                  student distribution ``p_theta``. Trained with a manual data-parallel optimizer
                  (plain AdamW + ``all_reduce(AVG)``; the cold-start pattern) so there is exactly ONE
                  DeepSpeed/DDP engine (the ``fake`` adapter is added AFTER ``accelerator.prepare``
                  -> NOT engine-managed; ``.grad`` filled by a raw ``backward`` + hand-reduced).
  * ``student`` = the default LoRA adapter (the few-step generator we keep). Trains through the main
                  engine, EXACTLY ONCE per epoch (``gradient_step_per_epoch=1``).

Batch geometry (32-GPU, per_device_batch_size=1 target): one epoch = ``num_batches_per_epoch``
micro-steps; the generator is updated EXACTLY ONCE per epoch (at ``b_idx==0``) and the fake score
EVERY micro-step -> fake:gen = ``num_batches_per_epoch`` : 1 (set ``unique_sample_num_per_epoch`` so
this equals ``dmd_fake_ratio``, default 5). This is the epoch-mapped DMD2 two-timescale cycle
(DMD2: fake every step, gen every ``dfake_gen_update_ratio`` steps), with the main (generator)
optimizer stepping once/epoch per the experiment-batch-geometry rule (GAS=1: DMD2 asserts no accum).

Generator gradient (DMD2, adapted to rectified flow / velocity):
  Only ONE student forward carries grad -- the final one-step prediction. The multi-step backward
  simulation that produces its input is ``no_grad`` (predict x0, re-noise to the next level with
  FRESH noise -- DMD2 ``sample_backward``). The DM loss samples a diffusion step ``t``, noises
  ``x0_G``, evaluates BOTH scores (real teacher + fake) under ``no_grad``, forms the self-normalized
  DMD gradient ``grad = (p_real - p_fake) / mean|p_real|`` (``p = x0_G - x0_pred``; ``nan_to_num``),
  and applies the stop-grad identity ``loss_dm = 0.5*||x0_G - sg(x0_G - grad)||^2`` so
  ``d loss/d x0_G = grad`` and ``d loss/d theta = grad . d x0_G/d theta``.
"""
import os
from contextlib import nullcontext
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
from ...utils.noise_schedule import flow_match_sigma
from .trainer import XOPDTrainer

logger = setup_logger(__name__)

TIMESTEP_MAX = 1000.0  # scheduler timestep scale (sigma = t / TIMESTEP_MAX, see flow_match_sigma)


class XDMDTrainer(XOPDTrainer):
    """Cross-model DMD: 32B-dev ``real`` -> klein-4B ``student`` (+ online ``fake`` score)."""

    # DMD has no L1 latent-transport trajectory, so the one-step-per-epoch L1 validation
    # (XOPDTrainer.__init__) does not apply. The generator still steps EXACTLY ONCE per epoch
    # (gradient_step_per_epoch=1); DMD configs pin gradient_accumulation_steps: 1 (DMD2 asserts
    # no gradient accumulation for the generator).
    _validates_l1_one_step = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from ...hparams.training_args import XDMDTrainingArguments

        self.training_args: XDMDTrainingArguments
        self._dmd_setup_fake_adapter()

    # ------------------------------------------------------------------ setup ----
    def _dmd_setup_fake_adapter(self) -> None:
        """Add the ``fake`` LoRA adapter on the (already-prepared) klein base + manual-DP optimizer.

        Runs AFTER ``super().__init__`` (hence after ``accelerator.prepare``): the ``fake`` params
        are added to the UNWRAPPED PeftModel, so they are NOT registered with the DeepSpeed/DDP
        engine (no ZeRO grad hooks) -> their ``.grad`` is filled by a plain ``backward`` and reduced
        by hand. The ``default`` (student) adapter -- already engine-managed via ``self.optimizer``
        -- is untouched.
        """
        ta = self.training_args
        transformer = self.accelerator.unwrap_model(self.adapter.transformer)
        if not isinstance(transformer, PeftModel):
            raise TypeError(
                "XDMD requires a LoRA student (finetune_type='lora'): expected the student "
                f"transformer to be a PeftModel, got {type(transformer).__name__}. Set "
                "model.finetune_type: lora."
            )
        if "default" not in transformer.peft_config:
            raise RuntimeError(
                "XDMD expected the default ('student') LoRA adapter on the transformer; found "
                f"adapters {list(transformer.peft_config)!r}."
            )
        if "fake" not in transformer.peft_config:
            transformer.add_adapter("fake", transformer.peft_config["default"])
            logger.info("XDMD: added 'fake' LoRA adapter (peer of 'student'/'default').")
        self._dmd_transformer = transformer

        # Collect the fake-adapter params (set_adapter flips requires_grad to the active adapter).
        transformer.set_adapter("fake")
        fake_params = [p for p in transformer.parameters() if p.requires_grad]
        if not fake_params:
            raise RuntimeError("XDMD: 'fake' adapter has no trainable params after set_adapter.")

        # Identical fake replicas across ranks: broadcast the (per-rank random gaussian) init from
        # rank 0. Combined with all_reduce(AVG) grads, every rank's fake stays bit-identical.
        self._world_size = int(self.accelerator.num_processes)
        if self._world_size > 1 and dist.is_initialized():
            for p in fake_params:
                dist.broadcast(p.data, src=0)

        self._fake_params = fake_params
        self._opt_fake = torch.optim.AdamW(
            fake_params,
            lr=ta.dmd_fake_lr,
            betas=ta.adam_betas,
            weight_decay=ta.adam_weight_decay,
            eps=ta.adam_epsilon,
        )
        # Restore the student adapter as the active/trainable one (default engine state).
        transformer.set_adapter("default")

        # fake:gen ratio is the number of micro-steps per epoch (gen once/epoch at b_idx==0).
        n = int(ta.num_batches_per_epoch)
        if n < 1:
            raise ValueError(f"XDMD needs num_batches_per_epoch >= 1, got {n}.")
        if n != int(ta.dmd_fake_ratio):
            logger.warning(
                "XDMD: effective fake:gen ratio = num_batches_per_epoch (%d), which differs from "
                "dmd_fake_ratio (%d). Set unique_sample_num_per_epoch = dmd_fake_ratio * "
                "per_device_batch_size * num_processes to make them match.",
                n, ta.dmd_fake_ratio,
            )
        # Few-step generator: eval SHOULD sample at the SAME step budget the generator is trained
        # for (dmd_sim_steps), otherwise the eval curve reflects a mismatched inference schedule.
        eval_steps = getattr(self.eval_args, "num_inference_steps", None)
        if eval_steps is not None and int(eval_steps) != int(ta.dmd_sim_steps):
            logger.warning(
                "XDMD: eval.num_inference_steps (%s) != dmd_sim_steps (%d). DMD yields a few-step "
                "generator; set eval.num_inference_steps = dmd_sim_steps so eval matches the "
                "deployed inference budget.",
                eval_steps, ta.dmd_sim_steps,
            )
        logger.info(
            "XDMD ready: fake adapter (%d tensors) manual-DP AdamW(lr=%g); student active. "
            "world_size=%d, sim_steps=%d, real_gs=%g, fake:gen=%d:1 (1 gen update/epoch).",
            len(fake_params), ta.dmd_fake_lr, self._world_size, ta.dmd_sim_steps,
            ta.dmd_real_guidance_scale, n,
        )

    def _dmd_set_adapter(self, name: str) -> None:
        """Toggle the active adapter on the (unwrapped) PeftModel; PEFT flips requires_grad too."""
        self._dmd_transformer.set_adapter(name)

    @staticmethod
    def _dmd_uniform_t(batch_size: int, device: torch.device, lo_frac: float, hi_frac: float) -> torch.Tensor:
        """Per-sample UNIFORM scheduler timesteps in ``[lo_frac, hi_frac] * TIMESTEP_MAX`` (DMD2
        samples timesteps uniformly: randint(min_step, max_step) for the DM loss, randint(0, T) for
        the fake loss)."""
        return torch.rand(batch_size, device=device) * ((hi_frac - lo_frac) * TIMESTEP_MAX) + lo_frac * TIMESTEP_MAX

    # -------------------------------------------------------------- main loop ----
    def start(self):
        """DMD loop: per epoch, ONE generator DMD update (b_idx==0) + a fake-score update EVERY
        micro-step (fake:gen = num_batches_per_epoch : 1)."""
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

            self._dmd_epoch()

            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    def prepare_feedback(self, samples):  # unused (no reward feedback)
        pass

    def _dmd_epoch(self) -> None:
        """One epoch = ``num_batches_per_epoch`` micro-steps. Generator DMD update EXACTLY ONCE
        (b_idx==0) -> gradient_step_per_epoch=1; fake-score update EVERY micro-step."""
        ta = self.training_args
        device = self.accelerator.device
        data_iter = self._make_train_iter()
        n = int(ta.num_batches_per_epoch)

        for b_idx in tqdm(
            range(n), desc=f"Epoch {self.epoch} DMD", disable=not self.show_progress_bar
        ):
            prompt_batch = next(data_iter)
            info: Dict[str, Any] = {}
            if b_idx == 0:
                # The single generator DMD update of this epoch (DMD2-style single generation).
                loss_dm, x0_G, latent_ids, pe, text_ids = self._dmd_generator_step(prompt_batch, device)
                info["loss_dm"] = loss_dm
            else:
                # Fake-only turn: produce an on-policy DETACHED sample to train the fake score.
                x0_G, latent_ids, pe, text_ids = self._dmd_sample_x0(prompt_batch, device, with_grad=False)

            # Fake-score update EVERY step on the (detached) student sample.
            info.update(self._dmd_fake_step(x0_G, latent_ids, pe, text_ids, device))
            self.log_data({f"train/{k}": v for k, v in info.items()}, step=self.step)
            self.step += 1

    # -------------------------------------------------------------- generation ---
    def _dmd_sample_x0(
        self, prompt_batch: Dict[str, Any], device: torch.device, with_grad: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """DMD2 ``sample_backward`` + one-step prediction, in rectified-flow form.

        A random denoising index ``sel in [0, sim_steps)`` is drawn on rank 0 and BROADCAST (so all
        ranks run the same number of transformer calls -> collective-safe). ``sel`` NO-GRAD steps
        run the DMD2 backward simulation from fresh noise (predict x0, re-noise to the next level
        with FRESH noise); a SINGLE student step at ``timesteps[sel]`` then gives
        ``x0_G = z - sigma*v_s`` (guidance-free). ``with_grad`` keeps the graph for the generator
        update; otherwise the whole thing is ``no_grad`` (feeds the fake update)."""
        ta = self.training_args
        self._dmd_set_adapter("default")
        if with_grad:
            self.adapter.train()
        else:
            self.adapter.rollout()

        pe = prompt_batch["prompt_embeds"].to(device)
        text_ids = prompt_batch["text_ids"].to(device)
        batch_size = pe.shape[0]
        height, width = self._dmd_resolution()

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
        z = self.adapter.cast_latents(z)  # pure noise (sigma = 1)
        mu = compute_empirical_mu(image_seq_len=z.shape[1], num_steps=ta.dmd_sim_steps)
        timesteps = set_scheduler_timesteps(
            scheduler=self.adapter.pipeline.scheduler,
            num_inference_steps=ta.dmd_sim_steps,
            device=device,
            mu=mu,
        )

        # Random start index, shared across ranks (collective safety).
        sel = torch.randint(0, ta.dmd_sim_steps, (1,), device=device)
        if self._world_size > 1 and dist.is_initialized():
            dist.broadcast(sel, src=0)
        sel = int(sel.item())

        def _sig(t_scalar: torch.Tensor) -> torch.Tensor:
            t = t_scalar.repeat(batch_size).to(device)
            return t, flow_match_sigma(t).reshape(-1, *([1] * (z.ndim - 1)))

        # DMD2 backward simulation (no_grad): predict x0, re-noise to the next level with FRESH noise.
        with torch.no_grad(), self.autocast():
            for i in range(sel):
                t_i, sig_i = _sig(timesteps[i])
                v = self.adapter.predict_velocity(
                    t=t_i, latents=z, latent_ids=latent_ids,
                    prompt_embeds=pe, text_ids=text_ids, guidance_scale=1.0,
                )
                x0 = z - sig_i * v
                _, sig_next = _sig(timesteps[i + 1])
                z = (1.0 - sig_next) * x0 + sig_next * torch.randn_like(x0)  # re-noise (fresh)

        # Single (optionally differentiable) student step at timesteps[sel] -> one-step x0.
        t_sel, sig_sel = _sig(timesteps[sel])
        grad_ctx = nullcontext() if with_grad else torch.no_grad()
        with grad_ctx, self.autocast():
            v_s = self.adapter.predict_velocity(
                t=t_sel, latents=z, latent_ids=latent_ids,
                prompt_embeds=pe, text_ids=text_ids, guidance_scale=1.0,
            )
            x0_G = z - sig_sel * v_s
        if not with_grad:
            x0_G = x0_G.detach()
        return x0_G, latent_ids, pe, text_ids

    def _dmd_resolution(self) -> Tuple[int, int]:
        ta = self.training_args
        if ta.height and ta.width:
            return int(ta.height), int(ta.width)
        res = ta.resolution
        if isinstance(res, (list, tuple)):
            h = int(res[0])
            w = int(res[1]) if len(res) > 1 else h
            return h, w
        return int(res), int(res)

    # -------------------------------------------------------- generator (DMD) ----
    def _dmd_teacher_cond(self, prompt_batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
        """Same-arch teacher text conditioning (precomputed ``teacher_*`` fields). flux2-dev is
        guidance-distilled with ``guidance=None``, so ``dmd_real_guidance_scale=1.0`` is a single
        conditional pass (no negatives). Negatives are added ONLY if the real CFG > 1 (available via
        the same clean interface). Cross-VAE is unsupported (shared-VAE assumed)."""
        if self._cross_vae:
            raise NotImplementedError("XDMD assumes a same-arch (shared-VAE) teacher; cross-VAE is unsupported.")
        cond = {
            "prompt_embeds": prompt_batch["teacher_prompt_embeds"].to(device),
            "text_ids": prompt_batch["teacher_text_ids"].to(device),
        }
        if self.training_args.dmd_real_guidance_scale > 1.0:
            if "teacher_negative_prompt_embeds" not in prompt_batch:
                raise KeyError(
                    "dmd_real_guidance_scale > 1 needs teacher negatives (teacher_negative_prompt_embeds); "
                    "precompute them or keep dmd_real_guidance_scale: 1.0 (flux2-dev is guidance-distilled)."
                )
            cond["negative_prompt_embeds"] = prompt_batch["teacher_negative_prompt_embeds"].to(device)
            cond["negative_text_ids"] = prompt_batch["teacher_negative_text_ids"].to(device)
        return cond

    def _dmd_generator_step(
        self, prompt_batch: Dict[str, Any], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One generator DMD update. Returns ``(loss_dm, x0_G_detached, latent_ids, pe, text_ids)``.

        Runs inside ``accelerator.accumulate`` (GAS=1 -> one optimizer step, DMD2-style). Both scores
        run under ``no_grad``; only ``x0_G`` carries grad, via the stop-grad DMD identity."""
        ta = self.training_args
        teacher_cond = self._dmd_teacher_cond(prompt_batch, device)

        with self.accelerator.accumulate(*self.adapter.trainable_components):
            x0_G, latent_ids, pe, text_ids = self._dmd_sample_x0(prompt_batch, device, with_grad=True)
            spatial = tuple(range(1, x0_G.ndim))

            with torch.no_grad():
                # DM diffusion step ~ Uniform[t_min, t_max] (DMD2 randint(min_step, max_step)).
                t = self._dmd_uniform_t(x0_G.shape[0], device, ta.dmd_t_min, ta.dmd_t_max)
                sig = flow_match_sigma(t).reshape(-1, *([1] * (x0_G.ndim - 1)))
                eps = torch.randn_like(x0_G)
                z_dm = (1.0 - sig) * x0_G + sig * eps  # value only (no_grad)

                # real score: 32B teacher (CFG = dmd_real_guidance_scale; =1 -> single pass), swapped in.
                with self.autocast(), self.adapter.use_teacher_transformer():
                    v_real = self.adapter.predict_velocity(
                        t=t, latents=z_dm, latent_ids=latent_ids,
                        guidance_scale=ta.dmd_real_guidance_scale, **teacher_cond,
                    )
                x0_real = z_dm - sig * v_real

                # fake score: fake adapter (no CFG). Toggle adapter then restore student.
                self._dmd_set_adapter("fake")
                with self.autocast():
                    v_fake = self.adapter.predict_velocity(
                        t=t, latents=z_dm, latent_ids=latent_ids,
                        prompt_embeds=pe, text_ids=text_ids, guidance_scale=1.0,
                    )
                self._dmd_set_adapter("default")
                x0_fake = z_dm - sig * v_fake

                # Self-normalized DMD gradient (DMD2): p = x0_G - x0_pred; grad = (p_real - p_fake)/mean|p_real|.
                p_real = (x0_G - x0_real).float()
                p_fake = (x0_G - x0_fake).float()
                norm = p_real.abs().mean(dim=spatial, keepdim=True)
                grad = torch.nan_to_num((p_real - p_fake) / norm)

            # Stop-grad identity: d loss/d x0_G = grad (DMD2 0.5*mse(x0_G, sg(x0_G - grad))).
            target = (x0_G.float() - grad).detach()
            loss_dm = 0.5 * (x0_G.float() - target).pow(2).mean(dim=spatial).mean()

            self.accelerator.backward(loss_dm)
            grad_norm = None
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.adapter.get_trainable_parameters(), ta.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._opt_fake.zero_grad(set_to_none=True)  # zero BOTH (DMD2)

        info = reduce_loss_info(self.accelerator, {"loss_dm": [loss_dm.detach()]})
        self.log_data(
            {
                "train/gen_grad_norm": grad_norm if grad_norm is not None else 0.0,
                "train/dmd_grad_abs_mean": grad.abs().mean().detach(),
            },
            step=self.step,
        )
        return info["loss_dm"], x0_G.detach(), latent_ids, pe, text_ids

    # ----------------------------------------------------------- fake training ---
    def _dmd_fake_step(
        self,
        x0_G: torch.Tensor,
        latent_ids: torch.Tensor,
        pe: torch.Tensor,
        text_ids: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Train the ``fake`` score on the (detached) student samples: flow-matching velocity MSE at
        a random ``t ~ Uniform[0, T]`` (DMD2 fake loss uses the FULL timestep range) with fresh
        noise. Manual data-parallel optimizer step (no engine)."""
        ta = self.training_args
        self._dmd_set_adapter("fake")
        self.adapter.train()

        batch_size = x0_G.shape[0]
        t = self._dmd_uniform_t(batch_size, device, 0.0, 1.0)  # full range (DMD2 compute_loss_fake)
        sig = flow_match_sigma(t).reshape(-1, *([1] * (x0_G.ndim - 1)))
        eps = torch.randn_like(x0_G)
        z_t = (1.0 - sig) * x0_G + sig * eps
        v_target = eps - x0_G  # rectified-flow velocity target: z_t = (1-sig)*x0 + sig*eps

        with self.autocast():
            v_fake = self.adapter.predict_velocity(
                t=t,
                latents=z_t,
                latent_ids=latent_ids,
                prompt_embeds=pe,
                text_ids=text_ids,
                guidance_scale=ta.dmd_fake_guidance_scale,  # == 1.0 (no CFG for fake)
            )
        diff = v_fake.float() - v_target.float()
        loss = diff.pow(2).mean(dim=tuple(range(1, diff.ndim))).mean()

        self._opt_fake.zero_grad(set_to_none=True)
        loss.backward()
        # Manual data-parallel: average the fake grads across ranks (identical replicas).
        if self._world_size > 1 and dist.is_initialized():
            for p in self._fake_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= self._world_size
        grad_norm = torch.nn.utils.clip_grad_norm_(self._fake_params, ta.dmd_grad_norm)
        self._opt_fake.step()
        self._opt_fake.zero_grad(set_to_none=True)
        # Restore the student adapter as active (default engine state).
        self._dmd_set_adapter("default")

        info = reduce_loss_info(self.accelerator, {"loss_fake": [loss.detach()]})
        info["fake_grad_norm"] = grad_norm
        info["x0G_std"] = x0_G.float().std()
        return info
