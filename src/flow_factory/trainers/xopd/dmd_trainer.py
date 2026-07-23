# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Cross-model Distribution Matching Distillation (DMD) trainer.

Design: docs/xopd/dmd_cross_model_design.md. Reference: tianweiy/DMD2.

Roles (pure DMD, no MSE flow-anchor, no GAN head in v1):
  * ``real``    = the frozen 32B flux2-dev teacher (XOPD's ``use_teacher_transformer`` swap).
  * ``fake``    = a SECOND LoRA adapter on the SAME frozen klein-4B base -- an ONLINE score of
                  the student distribution ``p_theta``. Trained with a manual data-parallel
                  optimizer (plain AdamW + ``all_reduce(AVG)``; the cold-start pattern), so there
                  is exactly ONE DeepSpeed/DDP engine (the ``fake`` adapter is added AFTER
                  ``accelerator.prepare`` -> it is NOT engine-managed; its ``.grad`` is filled by a
                  raw ``backward`` and reduced by hand).
  * ``student`` = the default LoRA adapter (the few-step generator we keep). Trains through the
                  main engine (Stage 3; not stepped in Stage 1+2).

THIS FILE IMPLEMENTS STAGE 1+2 of the staged rollout (scaffold + fake loop):
  1. build the ``fake`` adapter + its manual-DP optimizer (identical replicas via broadcast init).
  2. each micro-step: (a) generate a few-step on-policy clean latent ``x0_G`` with the ``student``
     adapter (no grad), (b) train ``fake`` to the flow-matching velocity of ``x0_G`` (random t,
     fresh noise). Verifies ``loss_fake`` decreases; replicas stay in sync BY CONSTRUCTION.

Stage 3 (generator DMD update via ``(x0_real - x0_fake)`` self-normalized grad) is intentionally
NOT here yet -- see the design doc.
"""
import os
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


class XDMDTrainer(XOPDTrainer):
    """Cross-model DMD (Stage 1+2: fake-score loop on a second LoRA adapter)."""

    # DMD has no L1 latent-transport trajectory, so the one-step-per-epoch L1 validation
    # (XOPDTrainer.__init__) does not apply. Auto-GAS stays well-defined via
    # XDMDTrainingArguments.get_num_train_timesteps() == 1.
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
        logger.info(
            "XDMD Stage 1+2 ready: fake adapter (%d tensors) with manual-DP AdamW(lr=%g), "
            "student adapter active. world_size=%d, sim_steps=%d, fake_ratio=%d.",
            len(fake_params), ta.dmd_fake_lr, self._world_size, ta.dmd_sim_steps, ta.dmd_fake_ratio,
        )

    def _dmd_set_adapter(self, name: str) -> None:
        """Toggle the active adapter on the (unwrapped) PeftModel; PEFT flips requires_grad too."""
        self._dmd_transformer.set_adapter(name)

    # -------------------------------------------------------------- main loop ----
    def start(self):
        """Stage 1+2 loop: per epoch, roll out x0_G (student) and train the fake score."""
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

            self._dmd_fake_epoch()

            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    def prepare_feedback(self, samples):  # unused (no reward feedback)
        pass

    def _dmd_fake_epoch(self) -> None:
        ta = self.training_args
        device = self.accelerator.device
        data_iter = self._make_train_iter()

        for b_idx in tqdm(
            range(ta.num_batches_per_epoch),
            desc=f"Epoch {self.epoch} DMD-fake",
            disable=not self.show_progress_bar,
        ):
            prompt_batch = next(data_iter)
            # (a) on-policy few-step generation with the student adapter (no grad in Stage 1+2).
            x0_G, latent_ids, pe, text_ids = self._dmd_generate_latents(prompt_batch, device)
            # (b) train the fake score to the flow-matching velocity of x0_G (manual-DP step).
            info = self._dmd_fake_step(x0_G, latent_ids, pe, text_ids, device)
            self.log_data({f"train/{k}": v for k, v in info.items()}, step=self.step)
            self.step += 1

    # -------------------------------------------------------------- generation ---
    @torch.no_grad()
    def _dmd_generate_latents(
        self, prompt_batch: Dict[str, Any], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Few-step (``dmd_sim_steps``) deterministic ODE rollout from noise with the STUDENT
        adapter -> clean latent ``x0_G`` (detached), plus the latent/text conditioning used.

        Uses the scheduler's Euler *mean* transition (``next_latents_mean``) at each step (no SDE
        noise), guidance-free (DMD generators run without CFG). Fresh initial noise every call
        (reproducibility intentionally dropped)."""
        ta = self.training_args
        self._dmd_set_adapter("default")
        self.adapter.rollout()

        pe = prompt_batch["prompt_embeds"].to(device)
        text_ids = prompt_batch["text_ids"].to(device)
        batch_size = pe.shape[0]
        height, width = self._dmd_resolution()

        num_channels_latents = self.adapter.pipeline.transformer.config.in_channels // 4
        latents, latent_ids = self.adapter.pipeline.prepare_latents(
            batch_size=batch_size,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=pe.dtype,
            device=device,
            generator=None,
            latents=None,
        )

        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=ta.dmd_sim_steps)
        timesteps = set_scheduler_timesteps(
            scheduler=self.adapter.pipeline.scheduler,
            num_inference_steps=ta.dmd_sim_steps,
            device=device,
            mu=mu,
        )
        latents = self.adapter.cast_latents(latents)
        with self.autocast():
            for i, t in enumerate(timesteps):
                t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
                out = self.adapter.forward(
                    t=t,
                    t_next=t_next,
                    latents=latents,
                    latent_ids=latent_ids,
                    prompt_embeds=pe,
                    text_ids=text_ids,
                    guidance_scale=1.0,
                    compute_log_prob=False,
                    return_kwargs=["next_latents_mean"],
                )
                latents = self.adapter.cast_latents(out.next_latents_mean)
        return latents.detach(), latent_ids, pe, text_ids

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

    # ----------------------------------------------------------- fake training ---
    def _dmd_fake_step(
        self,
        x0_G: torch.Tensor,
        latent_ids: torch.Tensor,
        pe: torch.Tensor,
        text_ids: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Train the ``fake`` score on the (detached) student samples: flow-matching velocity MSE
        at a random ``t`` with fresh noise. Manual data-parallel optimizer step (no engine)."""
        ta = self.training_args
        self._dmd_set_adapter("fake")
        self.adapter.train()

        batch_size = x0_G.shape[0]
        t = self._sample_l0_timesteps(batch_size, device).clamp(
            ta.dmd_t_min * 1000.0, ta.dmd_t_max * 1000.0
        )
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
