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

import numpy as np
import torch
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ...hparams import XOPDTrainingArguments
from ...rewards import RewardBuffer
from ...samples import BaseSample
from ...utils.base import (
    create_generator,
    create_generator_by_prompt,
    filter_kwargs,
    stitch_batch_metadata,
)
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
from .transport import build_transport

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

        # Fail fast: the REINFORCE trajectory term is well-defined only under a
        # stochastic (SDE) transition. Under ODE the transition is deterministic
        # (std_dev_t == 0), so the scheduler returns log_prob == 0 / None and the
        # term `reinforce_coef * (R_bar * log_prob)` silently contributes nothing
        # to the loss. Rather than let `reinforce_coef > 0` look effective while
        # being a no-op, reject the combination outright (use a Flow-SDE /
        # Dance-SDE / CPS scheduler with noise_level > 0 to enable REINFORCE).
        if self._is_ode and self.reinforce_coef > 0:
            raise ValueError(
                "XOPD: reinforce_coef > 0 requires a stochastic scheduler "
                "(dynamics_type in {'Flow-SDE', 'Dance-SDE', 'CPS'} with "
                "noise_level > 0). Under ODE the transition is deterministic, so "
                "the REINFORCE term's log_prob is identically zero and the term is "
                f"a no-op. Got dynamics_type='ODE' and reinforce_coef="
                f"{self.reinforce_coef}. Set reinforce_coef=0 for ODE, or switch "
                "to an SDE scheduler to use REINFORCE."
            )

        # Cache adapter.forward signature once for cheap per-step kwarg filtering.
        self._forward_param_names, self._forward_accepts_var_kwargs = cache_forward_signature(
            self.adapter.forward
        )
        # Cache the student velocity signature too (cross-VAE L0 forwards extra
        # conditioning, e.g. pooled embeds for SD3.5 vs latent_ids for FLUX.2).
        if hasattr(self.adapter, "predict_velocity"):
            self._velocity_param_names, self._velocity_accepts_var_kwargs = (
                cache_forward_signature(self.adapter.predict_velocity)
            )
        else:
            self._velocity_param_names, self._velocity_accepts_var_kwargs = frozenset(), False

        # Teacher backend: same-architecture (swap transformer into the student
        # pipeline, shared VAE) vs cross-VAE (independent frozen teacher adapter +
        # a latent-space transport). Shared-VAE is the `vae_transport='identity'`
        # special case of the generalized cross-VAE path.
        self._cross_vae = ta.teacher_model_type is not None
        self.teacher_adapter = None  # set in cross-VAE mode
        self.transport = None        # set below
        if self._cross_vae:
            self._init_cross_vae_teacher()
        else:
            self._init_same_arch_teacher()

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

        # Teacher-baseline eval: cache of the (constant) teacher reward scalars,
        # keyed by wandb metric name. Populated once by evaluate_teacher_baseline()
        # at epoch 0 and re-emitted at every evaluate() so the teacher renders as
        # a flat reference line spanning the student's x-axis (single-chart
        # overlay). Empty until the baseline runs.
        self._teacher_baseline_scalars: Dict[str, float] = {}

    # ============================ Teacher backends ============================
    def _init_same_arch_teacher(self) -> None:
        """Same-architecture teacher: swap the teacher transformer into the student
        pipeline (shared VAE / scheduler / latent space). Identity transport."""
        ta = self.training_args
        if not (
            hasattr(self.adapter, "load_teacher_transformer")
            and hasattr(self.adapter, "use_teacher_transformer")
            and hasattr(self.adapter, "predict_velocity")
        ):
            raise TypeError(
                "Same-architecture XOPD requires a cross-model-capable adapter "
                "exposing `load_teacher_transformer`, `use_teacher_transformer`, "
                f"and `predict_velocity` (e.g. Flux2KleinAdapter); got "
                f"{type(self.adapter).__name__}. For a different teacher "
                "architecture set teacher_model_type + vae_transport."
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
        # Same-architecture cross-model XOPD: the teacher transformer expects text
        # embeddings from its OWN encoder; those are precomputed during
        # preprocessing and cached (teacher_* columns), and the teacher text
        # encoder is freed before training (see _init_dataloader). Identity
        # transport: teacher velocities already live in the student latent space.
        self.transport = build_transport("identity")

    def _init_cross_vae_teacher(self) -> None:
        """Cross-VAE teacher: build an INDEPENDENT frozen teacher adapter (its own
        VAE / scheduler / latent layout) and a latent-space transport into the
        student space.

        The teacher adapter is constructed from a cloned config whose model_args
        point at the teacher (model_type=teacher_model_type, path=teacher path). It
        is frozen, eval, and NOT ``accelerator.prepare``d (own data_ptr; not DDP-
        wrapped), so the CLAUDE.md autocast-cache / DDP-bypass weight-swap
        invariants do not apply (those guard ``.data.copy_()`` swaps, not a fully
        separate module). See docs/mof/xopd_vae_space_align.tex.

        VRAM: the teacher transformer + teacher VAE are resident throughout
        training (L1 is on-policy: the teacher is queried at student-visited states
        every step). The teacher text encoder is precomputed offline and offloaded
        (see _init_dataloader) — only the teacher transformer + VAE stay.
        """
        import copy

        from ...models.loader import load_model

        ta = self.training_args
        device = self.accelerator.device

        # Clone the config and point model_args at the teacher.
        teacher_config = copy.deepcopy(self.config)
        teacher_config.model_args.model_type = ta.teacher_model_type
        teacher_config.model_args.model_name_or_path = ta.teacher_model_name_or_path
        teacher_config.model_args.finetune_type = "full"  # teacher is frozen, no LoRA
        teacher_config.model_args.resume_path = None
        teacher_config.model_args.resume_type = None

        logger.info(
            f"Cross-VAE XOPD: building independent teacher adapter "
            f"(model_type={ta.teacher_model_type!r}, path={ta.teacher_model_name_or_path!r}); "
            f"transport={ta.vae_transport!r}."
        )
        self.teacher_adapter = load_model(teacher_config, self.accelerator)
        # Freeze + eval + on-device. The teacher is NOT accelerator.prepare()d.
        for name in self.teacher_adapter._resolve_component_names():
            comp = self.teacher_adapter.get_component(name)
            if comp is not None and hasattr(comp, "requires_grad_"):
                comp.requires_grad_(False)
        self.teacher_adapter.eval()
        self.teacher_adapter.on_load(device=device)

        # Build the transport. Adapter latent-layout converters (native <-> BCHW)
        # let an affine transport accept/return native latents; identity for an
        # already-BCHW adapter (SD3.5), unpack/pack for FLUX.2.
        if ta.vae_transport == "pixel":
            self.transport = build_transport(
                "pixel",
                teacher_adapter=self.teacher_adapter,
                student_adapter=self.adapter,
            )
        elif ta.vae_transport == "linear":
            self.transport = build_transport(
                "linear",
                teacher_to_spatial=self._teacher_to_spatial,
                teacher_from_spatial=self._teacher_from_spatial,
                student_to_spatial=self._student_to_spatial,
                student_from_spatial=self._student_from_spatial,
            )
        else:  # "mlp" -> placeholder (raises in build_transport/constructor)
            self.transport = build_transport(ta.vae_transport)

    # -- Canonical latent-layout converters (native <-> (B,C,H,W)) --
    @staticmethod
    def _adapter_to_spatial(adapter, z, **ctx):
        """Native latent -> (B,C,H,W). Uses adapter.to_spatial_latent if provided,
        else assumes the latent is already BCHW (e.g. SD3.5)."""
        if hasattr(adapter, "to_spatial_latent"):
            return adapter.to_spatial_latent(z, **ctx)
        return z

    @staticmethod
    def _adapter_from_spatial(adapter, z_spatial, **ctx):
        if hasattr(adapter, "from_spatial_latent"):
            return adapter.from_spatial_latent(z_spatial, **ctx)
        return z_spatial

    def _teacher_to_spatial(self, z, **ctx):
        return self._adapter_to_spatial(self.teacher_adapter, z, **ctx)

    def _teacher_from_spatial(self, z, **ctx):
        return self._adapter_from_spatial(self.teacher_adapter, z, **ctx)

    def _student_to_spatial(self, z, **ctx):
        return self._adapter_to_spatial(self.adapter, z, **ctx)

    def _student_from_spatial(self, z, **ctx):
        return self._adapter_from_spatial(self.adapter, z, **ctx)

    # ============================ Data iteration ============================
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
        # Teacher text embeddings are precomputed offline (teacher_* columns) and
        # the teacher text encoder is freed before training. The encode+cache hook
        # lives on the STUDENT adapter (Flux2KleinAdapter.preprocess_func writes
        # teacher_* when a teacher TE is loaded). This requires the student adapter
        # to expose load_teacher_text_encoder / unload_teacher_text_encoder.
        #
        # Cross-VAE caveat: when the student adapter lacks these hooks (e.g.
        # SD3_5Adapter), teacher text conditioning via precompute+offload is not
        # yet wired on that adapter. Fail fast with a clear message rather than
        # silently producing no teacher_* columns.
        if not hasattr(self.adapter, "load_teacher_text_encoder"):
            raise NotImplementedError(
                "Cross-VAE XOPD teacher text conditioning (precompute+offload) "
                f"requires the student adapter ({type(self.adapter).__name__}) to "
                "expose load_teacher_text_encoder / unload_teacher_text_encoder and "
                "a preprocess_func that caches teacher_* columns. This hook is "
                "currently implemented on Flux2KleinAdapter only; port it to the "
                "student adapter to enable a cross-architecture teacher. "
                "See docs/mof/xopd_vae_space_align.tex and the .scratch plan."
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
        if self.epoch == 0 and self.training_args.eval_teacher_at_start:
            self.evaluate_teacher_baseline()

        # Cross-VAE 'linear' transport: fit the affine map once before training,
        # then freeze (warm-up cold start). 'pixel'/'identity' need no warm-up.
        if (
            self._cross_vae
            and getattr(self.transport, "requires_warmup", False)
            and not getattr(self.transport, "is_fitted", True)
        ):
            self._warmup_linear_transport()

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

    # ===================== Teacher baseline evaluation =====================
    def evaluate(self) -> None:
        """Student eval + re-emit the constant teacher baseline for overlay.

        Runs the standard student evaluation, then (main process) re-logs the
        cached teacher-baseline scalars at the current step so the teacher shows
        as a flat reference line spanning the student's x-axis in the SAME wandb
        chart (both live under ``eval/{test_set}/...``). No-op for the teacher
        part until :meth:`evaluate_teacher_baseline` has populated the cache.
        """
        super().evaluate()
        if self.accelerator.is_main_process and self._teacher_baseline_scalars:
            self.log_data(dict(self._teacher_baseline_scalars), step=self.step)

    def evaluate_teacher_baseline(self) -> None:
        """Evaluate the frozen teacher on every test set (fair student protocol).

        For each test set the teacher is scored with the SAME prompts, seed,
        num_inference_steps and per-test-set ``guidance_scale`` as the student
        eval, swapping in the teacher transformer (``use_teacher_transformer``)
        and routing the teacher's own cached text embeddings (``teacher_*``,
        precomputed for the test split during preprocessing). Results are logged
        under ``eval/{test_set}/teacher/...`` and cached in
        ``self._teacher_baseline_scalars`` so :meth:`evaluate` can re-emit them
        as a constant reference line.

        ``use_teacher_transformer`` swaps the WHOLE transformer module (a
        distinct ``data_ptr``), which already bypasses the DDP/ZeRO wrapper and
        disables the autocast cache, so no extra weight-swap guards are needed
        (unlike the ``use_named_parameters`` path).
        """
        if not self.test_dataloaders:
            logger.info("XOPD teacher baseline: no test dataloaders; skipping.")
            return

        self.adapter.eval()
        logger.info("XOPD: evaluating teacher baseline on all test sets")
        for test_set_name in sorted(self.test_dataloaders.keys()):
            self._evaluate_teacher_test_set(test_set_name)
        self.accelerator.wait_for_everyone()

    def _evaluate_teacher_test_set(self, test_set_name: str) -> None:
        """Score the teacher on one test set and cache the reward scalars."""
        self.eval_reward_buffer = RewardBuffer(
            self._eval_reward_processor_for_test_set(test_set_name),
            self.training_args.group_size,
        )
        merged_eval = self._merged_eval_args_for_test_set_name(test_set_name)
        eval_seed = merged_eval.seed if merged_eval.seed is not None else self.training_args.seed
        # Teacher metrics share the student's ``eval/{test_set}/`` section so they
        # can be overlaid in one chart; teacher curves get a ``/teacher`` suffix.
        log_pfx = f"{self._eval_log_prefix(test_set_name)}/teacher"

        with torch.no_grad(), self.autocast(), self.adapter.use_teacher_transformer():
            all_samples = self._run_teacher_eval_inference_batches(
                test_set_name, merged_eval, eval_seed
            )
            gathered_rewards = self._gather_eval_rewards()
            gathered_tags = self._gather_eval_tags(all_samples)
            if self.accelerator.is_main_process:
                self._log_eval_reward_metrics(
                    gathered_rewards, log_pfx, all_samples, gathered_tags=gathered_tags
                )
                # Cache scalar reward means/stds so evaluate() can re-emit a
                # constant reference line at every subsequent eval step.
                for key, value in gathered_rewards.items():
                    self._teacher_baseline_scalars[f"{log_pfx}/reward_{key}_mean"] = float(
                        np.mean(value)
                    )
                    self._teacher_baseline_scalars[f"{log_pfx}/reward_{key}_std"] = float(
                        np.std(value)
                    )
        self.accelerator.wait_for_everyone()

    def _run_teacher_eval_inference_batches(
        self,
        test_set_name: str,
        merged_eval: Any,
        eval_seed: int,
    ) -> List[BaseSample]:
        """Like ``_run_eval_inference_batches`` but routes teacher conditioning.

        Overrides the student text embeddings in each batch with the teacher's
        own cached ``teacher_*`` embeddings and applies the teacher's CFG
        (== the test set's eval guidance_scale, the same value the teacher
        negatives were cached under). Must be called inside
        ``use_teacher_transformer``.
        """
        all_samples: List[BaseSample] = []
        for batch in tqdm(
            self.test_dataloaders[test_set_name],
            desc=f"Teacher eval [{test_set_name}]",
            disable=not self.show_progress_bar,
        ):
            if "teacher_prompt_embeds" not in batch:
                raise RuntimeError(
                    "XOPD teacher baseline requires cached teacher text embeddings "
                    f"for test set {test_set_name!r} (key 'teacher_prompt_embeds'), "
                    "but none were found. Re-preprocess with the teacher text "
                    "encoder loaded (data.enable_preprocess=True) so teacher_* "
                    "columns are cached for the test split."
                )
            generator = create_generator_by_prompt(batch["prompt"], eval_seed)
            inference_kwargs = {
                "compute_log_prob": False,
                "generator": generator,
                "trajectory_indices": None,
                **merged_eval,
                **batch,
            }
            # Route teacher conditioning (teacher embeds carry no prompt_ids).
            inference_kwargs["prompt_embeds"] = batch["teacher_prompt_embeds"]
            inference_kwargs["text_ids"] = batch["teacher_text_ids"]
            inference_kwargs.pop("prompt_ids", None)
            if "teacher_negative_prompt_embeds" in batch:
                inference_kwargs["negative_prompt_embeds"] = batch[
                    "teacher_negative_prompt_embeds"
                ]
                inference_kwargs["negative_text_ids"] = batch["teacher_negative_text_ids"]
                inference_kwargs.pop("negative_prompt_ids", None)
            else:
                for _k in (
                    "negative_prompt_embeds",
                    "negative_text_ids",
                    "negative_prompt_ids",
                ):
                    inference_kwargs.pop(_k, None)
            inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
            samples = self.adapter.inference(**inference_kwargs)

            # Stitch dataset metadata onto generated samples for reward routing.
            stitch_batch_metadata(batch, samples)

            all_samples.extend(samples)
            self.eval_reward_buffer.add_samples(samples)
        return all_samples

    # ===================== Stage L0: velocity regression =====================
    def _l0_epoch(self) -> None:
        """L0 warmup: teacher-generated ``z0`` then student velocity regression.

        Same-architecture: teacher and student velocities live in one latent
        space; regress the student velocity onto the teacher velocity at the same
        ``z_t`` (the original XOPD L0).

        Cross-VAE (``_cross_vae``): the teacher's velocity is NOT comparable in the
        student space, so L0 degrades to sample transport + analytic flow matching
        (theory doc, Insight 1): teacher rolls out a clean image, the pixel bridge
        maps it to a clean student latent ``z0_S``, and the student regresses the
        analytic FM target ``(eps - z0_S)`` — no teacher velocity, no transport
        training. Dispatched to :meth:`_l0_epoch_cross_vae`.
        """
        if self._cross_vae:
            return self._l0_epoch_cross_vae()
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

    # ----- Cross-VAE L0 (sample transport + analytic flow matching) -----
    def _teacher_rollout_samples(self, prompt_batch: Dict[str, Any]) -> List[BaseSample]:
        """Cross-VAE: roll out the independent teacher adapter (teacher space).

        Returns the raw teacher samples; each carries its decoded ``image``
        ``(3,H,W)`` tensor (the teacher adapter decodes with ``output_type="pt"``
        internally) and its trajectory ``all_latents`` (teacher-native clean
        latent at the final step). Runs entirely in teacher space.
        """
        ta = self.training_args
        self.teacher_adapter.eval()
        infer_kwargs = {
            **ta,
            "prompt": prompt_batch.get("prompt"),
            "guidance_scale": self.teacher_gs,
            "num_inference_steps": ta.l0_num_inference_steps,
            "compute_log_prob": False,
        }
        infer_kwargs = filter_kwargs(self.teacher_adapter.inference, **infer_kwargs)
        with torch.no_grad(), self.autocast():
            return self.teacher_adapter.inference(**infer_kwargs)

    def _teacher_rollout_images(self, prompt_batch: Dict[str, Any]) -> torch.Tensor:
        """Cross-VAE: teacher images ``(B,3,H,W)`` for the pixel bridge (L0)."""
        teacher_samples = self._teacher_rollout_samples(prompt_batch)
        return torch.stack([s.image for s in teacher_samples], dim=0)

    def _warmup_linear_transport(self) -> None:
        """Fit the linear (affine) transport once, then freeze (cold start).

        Collects ``transport_warmup_batches`` of paired teacher/student clean
        latents for the SAME images — teacher native latent ``z0_T`` from the
        teacher rollout, student latent ``z0_S = encode_pixels(teacher image)``
        via the pixel bridge — and least-squares fits ``A, b`` (theory doc M2/M4).
        Main process fits; the fitted ``A, b`` are broadcast to all ranks.
        """
        ta = self.training_args
        device = self.accelerator.device
        data_iter = self._make_train_iter()
        z_T_list: List[torch.Tensor] = []
        z_S_list: List[torch.Tensor] = []

        logger.info(
            f"Cross-VAE linear transport warm-up: collecting "
            f"{ta.transport_warmup_batches} paired-latent batches."
        )
        for _ in tqdm(
            range(ta.transport_warmup_batches),
            desc="Linear transport warm-up (paired latents)",
            disable=not self.show_progress_bar,
        ):
            prompt_batch = next(data_iter)
            teacher_samples = self._teacher_rollout_samples(prompt_batch)
            # Teacher native clean latent (last trajectory step).
            z0_T = torch.stack(
                [s.all_latents[-1] for s in teacher_samples], dim=0
            ).to(device)
            images = torch.stack([s.image for s in teacher_samples], dim=0).to(device)
            with torch.no_grad():
                z0_S = self.adapter.encode_pixels(images)
            z_T_list.append(z0_T.float().cpu())
            z_S_list.append(z0_S.float().cpu())

        # Fit on the main process, broadcast A, b to all ranks for determinism.
        self.transport.fit(z_T_list, z_S_list)
        if self.accelerator.num_processes > 1:
            import torch.distributed as dist

            A = self.transport.A.to(device)
            b = self.transport.b.to(device)
            dist.broadcast(A, src=0)
            dist.broadcast(b, src=0)
            self.transport.A = A
            self.transport.b = b
        logger.info(
            f"Linear transport fitted: A {tuple(self.transport.A.shape)}, "
            f"b {tuple(self.transport.b.shape)} (frozen)."
        )

    def _l0_epoch_cross_vae(self) -> None:
        """Cross-VAE L0: pixel-bridge sample transport + analytic FM regression.

        Per batch: teacher rolls out images (teacher space), the pixel bridge maps
        them to clean student latents ``z0_S`` (``student.encode_pixels``), then the
        student regresses the analytic flow-matching target ``(eps - z0_S)`` at
        random ``t``. No teacher velocity, no transport training (Insight 1).
        """
        ta = self.training_args
        device = self.accelerator.device
        data_iter = self._make_train_iter()
        loss_info: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for _ in tqdm(
            range(ta.num_batches_per_epoch),
            desc=f"Epoch {self.epoch} L0 (cross-VAE: pixel transport + FM)",
            disable=not self.show_progress_bar,
        ):
            prompt_batch = next(data_iter)

            # 1. Teacher rollout -> images -> student clean latent via pixel bridge.
            images = self._teacher_rollout_images(prompt_batch)
            with torch.no_grad():
                z0 = self.adapter.encode_pixels(images.to(device)).float()
            batch_size = z0.shape[0]

            # Student text conditioning (its own precomputed embeds).
            student_prompt_embeds = prompt_batch["prompt_embeds"].to(device)
            student_extra = self._student_cond_from_batch(prompt_batch, device)

            # 2. Student velocity regression onto the ANALYTIC FM target.
            self.adapter.train()
            for _inner in range(self.l0_inner_steps):
                with self.accelerator.accumulate(*self.adapter.trainable_components):
                    t = self._sample_l0_timesteps(batch_size, device)
                    sigma = flow_match_sigma(t)
                    sigma_b = sigma.view(batch_size, *([1] * (z0.ndim - 1)))
                    eps = torch.randn_like(z0)
                    z_t = (1.0 - sigma_b) * z0 + sigma_b * eps
                    fm_target = eps - z0  # analytic flow-matching velocity target

                    with self.autocast():
                        v_student = self.adapter.predict_velocity(
                            t=t,
                            latents=z_t,
                            prompt_embeds=student_prompt_embeds,
                            guidance_scale=self.student_gs,
                            **student_extra,
                        )

                    w = l0_loss_weight(sigma, ta.l0_weighting, ta.l0_snr_gamma)
                    per_sample_mse = (
                        (v_student.float() - fm_target.float())
                        .pow(2)
                        .mean(dim=tuple(range(1, v_student.ndim)))
                    )
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

    def _student_cond_from_batch(
        self, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """Collect the student adapter's extra conditioning kwargs from a batch.

        Adapter-agnostic: forwards whatever predict_velocity / forward accepts
        among the common student conditioning keys (pooled embeds for SD3.5,
        latent_ids for FLUX.2, negatives for CFG). Filtered to the student's
        velocity signature so unused keys are dropped.
        """
        candidates = {
            "pooled_prompt_embeds": batch.get("pooled_prompt_embeds"),
            "negative_prompt_embeds": batch.get("negative_prompt_embeds"),
            "negative_pooled_prompt_embeds": batch.get("negative_pooled_prompt_embeds"),
            "text_ids": batch.get("text_ids"),
            "latent_ids": batch.get("latent_ids"),
            "negative_text_ids": batch.get("negative_text_ids"),
        }
        out = {}
        for k, v in candidates.items():
            if v is None:
                continue
            if k in self._velocity_param_names or self._velocity_accepts_var_kwargs:
                out[k] = v.to(device) if torch.is_tensor(v) else v
        return out

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

    def _teacher_mean_dispatch(
        self,
        forward_kwargs: Dict[str, Any],
        teacher_text_cond: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Route the L1 teacher transition mean to the same-arch or cross-VAE path."""
        if self._cross_vae:
            return self._teacher_next_latents_mean_cross_vae(forward_kwargs, teacher_text_cond)
        return self._teacher_next_latents_mean(forward_kwargs, teacher_text_cond)

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

    def _teacher_next_latents_mean_cross_vae(
        self,
        forward_kwargs: Dict[str, Any],
        teacher_text_cond: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Cross-VAE teacher transition mean, mapped into the student space.

        The student transition was computed at student state ``x_S`` (the
        ``latents`` in ``forward_kwargs``) for ``(t, t_next)``. The transport maps
        ``x_S`` back to the teacher space, where the INDEPENDENT teacher adapter
        runs its own transition step (its VAE/scheduler/forward), then maps the
        teacher mean back to the student space:
        ``mu_S = transport.transition_mean_to_student(x_S, query_teacher_mean)``.
        For an affine transport this is exact and cheap (Prop. 3); for the pixel
        bridge it decodes/encodes through pixels (lossy, slow). All under no_grad.
        """
        x_S = forward_kwargs["latents"]
        t = forward_kwargs.get("t")
        t_next = forward_kwargs.get("t_next")

        def query_teacher_mean(x_T: torch.Tensor) -> torch.Tensor:
            tkw = {
                "t": t,
                "t_next": t_next,
                "latents": x_T,
                "next_latents": None,
                "compute_log_prob": False,
                "noise_level": self.teacher_adapter.scheduler.noise_level,
                "guidance_scale": self.teacher_gs,
                "return_kwargs": ["next_latents_mean", "std_dev_t", "dt"],
                **teacher_text_cond,
            }
            tkw = filter_kwargs(self.teacher_adapter.forward, **tkw)
            out = self.teacher_adapter.forward(**tkw)
            if out.next_latents_mean is None:
                raise RuntimeError(
                    "Cross-VAE teacher forward did not return `next_latents_mean`."
                )
            return out.next_latents_mean.detach()

        mu_S = self.transport.transition_mean_to_student(x_S, query_teacher_mean)
        return mu_S.detach()

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

                mu_teacher = self._teacher_mean_dispatch(
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
