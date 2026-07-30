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
  Gaussian transition KL ``D_k`` (mean matching) is the loss, optionally plus a
  KL anchor to the reference model. ``xopd_target_mode='p_opd'`` instead applies
  a detached Gaussian-mixture teacher responsibility to the covariance-normalized
  transition KL.

The teacher is a SEPARATE full transformer (not a LoRA snapshot), swapped in per
forward via ``adapter.use_teacher_transformer`` (whole-module swap, DDP-bypassed
for no_grad inference). Teacher and student use independent ``guidance_scale``.

Math helpers (``D_k``, forward-kwarg plumbing, L0 weighting) are copied into
:mod:`flow_factory.trainers.xopd.common` so XOPD does not depend on OPD
internals.

Registry key: ``'xopd'`` -> :class:`XOPDTrainer`.
"""

import os
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from ...utils.trajectory_collector import SCHEDULER_TRAIN_INDICES, compute_trajectory_indices
from ..abc import BaseTrainer
from .common import (
    POPDResponsibility,
    align_l0_inner_steps,
    build_forward_kwargs,
    cache_forward_signature,
    compute_per_step_kl,
    compute_popd_diagnostics,
    compute_popd_gaussian_mean_kl,
    compute_popd_quantiles,
    compute_popd_responsibility,
    compute_transition_variance,
    extract_i2i_condition_kwargs,
    extract_popd_behavior_transition,
    interleaved_source_iter,
    l0_loss_weight,
    validate_l1_one_step_per_epoch,
    validate_popd_configuration,
    validate_source_ratio,
)
from .router_coldstart import (
    alignment_metrics,
    balanced_spherical_kmeans,
    coldstart_router,
    pool_prompt_embeds,
)
from .transport import AdaLNTransport, build_transport

logger = setup_logger(__name__)


# Keys reused across student / teacher adapter.forward calls (mirror OPD).
_STUDENT_RETURN_KWARGS = ["next_latents_mean", "std_dev_t", "dt"]
_TEACHER_RETURN_KWARGS = ["next_latents_mean", "std_dev_t", "dt"]


@dataclass(frozen=True)
class _POPDStepCache:
    """Detached behavior-transition state for one P-OPD training timestep."""

    next_latents: torch.Tensor
    mu_old: torch.Tensor
    transition_variance: torch.Tensor
    dt: torch.Tensor
    responsibility: POPDResponsibility


class XOPDTrainer(BaseTrainer):
    """Cross-model on-policy distillation trainer (L0 velocity regression + L1)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: XOPDTrainingArguments
        ta = self.training_args

        self._is_ode = self.adapter.scheduler.dynamics_type == "ODE"
        self.pathwise_coef = ta.pathwise_coef
        self.normalize_d_k = ta.normalize_d_k
        self.xopd_dk_space = ta.xopd_dk_space
        self.teacher_gs = ta.teacher_guidance_scale
        self.student_gs = ta.student_guidance_scale
        self.xopd_target_mode = ta.xopd_target_mode
        self._is_popd = self.xopd_target_mode == "p_opd"
        self.popd_alpha = float(ta.popd_alpha) if self._is_popd else 0.5
        self.popd_temperature = float(ta.popd_temperature) if self._is_popd else 1.0
        self.popd_verbose_diagnostics = bool(ta.popd_verbose_diagnostics)

        # 'v' (raw velocity), 'x0' (clean-latent) and 'x0_norm' (self-normalized x0) d_k all recover
        # v from the ODE Euler mean (mu = x_t + v*dt); that identity only holds under ODE, so require
        # it. 'xt' (transition mean) works under any dynamics. See
        # docs/xopd/per_timestep_loss_dominance_theory.tex.
        if self.xopd_dk_space in ("v", "x0", "x0_norm") and not self._is_ode:
            raise ValueError(
                f"XOPD: xopd_dk_space={self.xopd_dk_space!r} requires an ODE scheduler "
                "(it recovers v via mu = x_t + v*dt). Got dynamics_type="
                f"{self.adapter.scheduler.dynamics_type!r}. Use 'xt' for SDE, or set "
                "scheduler.dynamics_type='ODE'."
            )

        # Cache adapter.forward signature once for cheap per-step kwarg filtering.
        self._forward_param_names, self._forward_accepts_var_kwargs = cache_forward_signature(
            self.adapter.forward
        )
        # Cache the student velocity signature too (cross-VAE L0 forwards extra
        # conditioning, e.g. pooled embeds for SD3.5 vs latent_ids for FLUX.2).
        if hasattr(self.adapter, "predict_velocity"):
            self._velocity_param_names, self._velocity_accepts_var_kwargs = cache_forward_signature(
                self.adapter.predict_velocity
            )
        else:
            self._velocity_param_names, self._velocity_accepts_var_kwargs = frozenset(), False

        # Teacher backend: same-architecture (swap transformer into the student
        # pipeline, shared VAE) vs cross-VAE (independent frozen teacher adapter +
        # a latent-space transport). Shared-VAE is the `vae_transport='identity'`
        # special case of the generalized cross-VAE path.
        self._cross_vae = ta.teacher_model_type is not None
        self.teacher_adapter = None  # set in cross-VAE mode
        self.transport = None  # set below
        # Cross-VAE FLUX teacher packed-latent layout (position ids + spatial size),
        # captured at warm-up and injected into the affine-transport converters.
        self._teacher_latent_ids = None
        self._teacher_spatial_hw = None
        # Transport/HSCT flags: default here so BOTH teacher backends (cross-VAE and
        # same-arch) always define them. _init_cross_vae_teacher refines _is_hsct/_pixel_loss
        # per transport; the same-arch (shared-VAE, identity-transport) path keeps these
        # defaults (no HSCT capture, no pixel-space L1).
        self._is_hsct = ta.vae_transport in ("hsct", "flow")
        self._pixel_loss = bool(ta.xopd_pixel_loss)
        self._hsct_hidden: Dict[int, torch.Tensor] = {}
        self._hsct_capture = False
        validate_popd_configuration(
            target_mode=self.xopd_target_mode,
            dynamics_type=self.adapter.scheduler.dynamics_type,
            noise_level=self.adapter.scheduler.noise_level,
            xopd_dk_space=self.xopd_dk_space,
            normalize_d_k=self.normalize_d_k,
            is_cross_vae=self._cross_vae,
            pixel_loss=self._pixel_loss,
        )
        if self._cross_vae:
            self._init_cross_vae_teacher()
        else:
            self._init_same_arch_teacher()

        if self.pathwise_coef == 0 and ta.kl_beta == 0 and ta.l0_warmup_epochs == 0:
            logger.warning(
                "XOPDTrainer received a zero-signal config: pathwise_coef="
                f"{self.pathwise_coef}, kl_beta={ta.kl_beta}, "
                f"l0_warmup_epochs={ta.l0_warmup_epochs}. "
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
        # case the validation below trips with a clear message. With
        # xopd_train_steps/num_xopd_steps the per-epoch subset is random but its SIZE
        # is FIXED, so use get_num_train_timesteps (the canonical fixed count) rather
        # than len() of one (possibly random) draw, keeping GAS consistent across epochs.
        self.num_train_timesteps = ta.get_num_train_timesteps(self.config)
        # Subclasses with a custom (non-L1) optimize loop (e.g. XPDM denoiser matching)
        # disable this L1-only one-step-per-epoch GAS invariant.
        if getattr(self, "_validates_l1_one_step", True):
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
        pipeline (shared VAE / scheduler / latent space). Identity transport.

        Default (ZeRO-2/DDP) path loads the teacher here. The FSDP full-shard path
        (``fsdp_shard_teacher``) loads it EARLIER in ``_initialization`` (before
        ``accelerator.prepare``, so it can be bundled with the student); this then only
        sets the transport."""
        if not getattr(self, "_teacher_loaded_early", False):
            self._load_same_arch_teacher_transformer()
        # Same-architecture cross-model XOPD: the teacher transformer expects text
        # embeddings from its OWN encoder; those are precomputed during
        # preprocessing and cached (teacher_* columns), and the teacher text
        # encoder is freed before training (see _init_dataloader). Identity
        # transport: teacher velocities already live in the student latent space.
        self.transport = build_transport("identity")

    def _load_same_arch_teacher_transformer(self, device=None) -> None:
        """Load the frozen same-arch teacher transformer into the student adapter
        (validation + device/dtype). Idempotent via ``_teacher_loaded_early``.

        ``device`` overrides where the teacher is loaded. The FSDP-bundle path passes
        ``device='cpu'`` so the full teacher is NOT materialized on every rank's GPU before
        ``accelerator.prepare`` shards it (that pre-shard spike OOMed the 8-expert runs);
        FSDP then shards the CPU teacher onto GPU incrementally."""
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
        load_device = device if device is not None else self.accelerator.device
        if device is None and ta.teacher_param_device == "cpu":
            logger.warning(
                "teacher_param_device='cpu' is not supported for the XOPD teacher "
                "transformer (per-forward H2D is not implemented); loading the "
                "teacher on the compute device instead."
            )
        self.adapter.load_teacher_transformer(
            ta.teacher_model_name_or_path,
            device=load_device,
            dtype=self.adapter._inference_dtype,
        )
        self._teacher_loaded_early = True

    def _clip_grad_norm_ep_aware(self, params, max_norm):
        """Grad-norm clipping tolerant of the expert-parallel parameter mix. With EP the trainable
        params split into FSDP-sharded backbone/router (DTensor grads) and EP-owned local experts
        (plain Tensor grads); torch's fused ``_foreach_norm`` cannot mix DTensor and Tensor in one
        call (``aten._foreach_norm got mixed torch.Tensor and DTensor``), so clip the two groups with
        separate calls -- each to ``max_norm``. This is a per-group heuristic, not a single global
        norm, which is fine for the stability role of clipping. Returns the FSDP (DTensor) group's
        grad norm for logging. Without EP (or no clipping), defers to accelerate's clip."""
        params = [p for p in params if p is not None]
        from ...utils.ep import ep_enabled

        if not ep_enabled() or not max_norm:
            return self.accelerator.clip_grad_norm_(params, max_norm)
        try:
            from torch.distributed.tensor import DTensor
        except ImportError:
            from torch.distributed._tensor import DTensor

        with_grad = [p for p in params if p.grad is not None]
        dparams = [p for p in with_grad if isinstance(p.grad, DTensor)]
        lparams = [p for p in with_grad if not isinstance(p.grad, DTensor)]
        grad_norm = None
        if dparams:
            grad_norm = self.accelerator.clip_grad_norm_(dparams, max_norm)
        if lparams:
            local_norm = torch.nn.utils.clip_grad_norm_(lparams, max_norm)
            grad_norm = local_norm if grad_norm is None else grad_norm
        return grad_norm

    def _initialization(self):
        """XOPD initialization. Default path defers to the base (student-only prepare;
        teacher loaded later in ``_init_same_arch_teacher``). The FSDP full-shard OOM
        path (``fsdp_shard_teacher``, same-arch only) loads the teacher and wraps
        student+teacher into ONE ModelBundle BEFORE ``accelerator.prepare`` (single FSDP
        root shards both), then installs routing proxies AFTER prepare."""
        ta = self.training_args
        use_bundle = (
            getattr(ta, "fsdp_shard_teacher", False)
            and ta.teacher_model_type is None
            and hasattr(self.adapter, "build_xopd_transformer_bundle")
        )
        if use_bundle:
            logger.info(
                "[FSDP-bundle] fsdp_shard_teacher=True (same-arch): load teacher on CPU + bundle "
                "with the (CPU) student BEFORE prepare, so one FSDP root shards both onto GPU "
                "incrementally (no pre-shard full-teacher GPU spike)."
            )
            self._load_same_arch_teacher_transformer(device="cpu")
            self.adapter.build_xopd_transformer_bundle()
        super()._initialization()
        if use_bundle:
            self.adapter.install_xopd_bundle_proxies(self.adapter.get_component("transformer"))

    def _init_cross_vae_teacher(self) -> None:
        """Cross-VAE teacher: build an INDEPENDENT frozen teacher adapter (its own
        VAE / scheduler / latent layout) and a latent-space transport into the
        student space.

        The teacher adapter is constructed from a cloned config whose model_args
        point at the teacher (model_type=teacher_model_type, path=teacher path). It
        is frozen, eval, and NOT ``accelerator.prepare``d (own data_ptr; not DDP-
        wrapped), so the CLAUDE.md autocast-cache / DDP-bypass weight-swap
        invariants do not apply (those guard ``.data.copy_()`` swaps, not a fully
        separate module). See docs/xopd/xopd_vae_space_align.tex.

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
        # The teacher VAE: some teacher repos ship NO vae/ subfolder (FLUX.2-klein-
        # base-9B is a transformer-only release sharing the 4B VAE). Resolve a VAE
        # source so the teacher pipeline can be built; auto-fall back to the klein
        # 4B VAE for a klein teacher whose repo lacks one.
        teacher_config.model_args.vae_name_or_path = self._resolve_teacher_vae_path()

        logger.info(
            f"Cross-VAE XOPD: building independent teacher adapter "
            f"(model_type={ta.teacher_model_type!r}, path={ta.teacher_model_name_or_path!r}, "
            f"vae={teacher_config.model_args.vae_name_or_path!r}); "
            f"transport={ta.vae_transport!r}."
        )
        self.teacher_adapter = load_model(teacher_config, self.accelerator)
        # One-time note: the adaLN transport conditions on the scheduler-agnostic
        # noise fraction sigma=t/num_train_timesteps, so a teacher/student timestep-base
        # mismatch is absorbed (warm-up and L1 both express the noise level as sigma).
        if ta.vae_transport == "adaln":
            t_sched = getattr(self.teacher_adapter, "scheduler", None)
            s_sched = getattr(self.adapter, "scheduler", None)
            t_N = getattr(t_sched, "num_train_timesteps", None) if t_sched is not None else None
            s_N = getattr(s_sched, "num_train_timesteps", None) if s_sched is not None else None
            if t_N is not None and s_N is not None and t_N != s_N:
                logger.info(
                    f"Cross-VAE adaLN transport: teacher num_train_timesteps={t_N} != "
                    f"student={s_N}; conditioning on sigma=t/num_train_timesteps "
                    "normalizes this difference."
                )
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
        # HSCT-family (hsct + flow): both are hidden-state-conditioned, cold-started
        # transports whose L1 query needs the captured student hidden states h_S. This
        # one flag gates the shared machinery (hooks, _hsct_h_list, cold-start, L1 threading).
        self._is_hsct = ta.vae_transport in ("hsct", "flow")
        self._pixel_loss = bool(ta.xopd_pixel_loss)  # cross-VAE L1 in decoded pixel space
        self._hsct_hidden: Dict[int, torch.Tensor] = {}
        self._hsct_capture = False
        if ta.vae_transport == "pixel":
            self.transport = build_transport(
                "pixel",
                teacher_adapter=self.teacher_adapter,
                student_adapter=self.adapter,
            )
        elif ta.vae_transport in ("linear", "whitening"):
            # Both are affine transports fit on paired latents during warm-up;
            # 'whitening' (M7) is the diagonal/AdaLN special case, 'linear' (M2)
            # the full channel affine. Same converter-based construction.
            self.transport = build_transport(
                ta.vae_transport,
                teacher_to_spatial=self._teacher_to_spatial,
                teacher_from_spatial=self._teacher_from_spatial,
                student_to_spatial=self._student_to_spatial,
                student_from_spatial=self._student_from_spatial,
            )
        elif ta.vae_transport == "adaln":
            # Learnable AdaLN affine: a 2nd module trained ONLY during warm-up on a
            # latent-reconstruction objective, then frozen (see _warmup_transport).
            # Channel counts come from each VAE's latent channels; spatial grids are
            # inferred from data at warm-up (init_from_moments).
            self.transport = build_transport(
                "adaln",
                teacher_to_spatial=self._teacher_to_spatial,
                teacher_from_spatial=self._teacher_from_spatial,
                student_to_spatial=self._student_to_spatial,
                student_from_spatial=self._student_from_spatial,
                teacher_channels=self._adapter_latent_channels(self.teacher_adapter),
                student_channels=self._adapter_latent_channels(self.adapter),
            )
        elif ta.vae_transport in ("conv", "conv_linear", "m5", "conv_nl", "nonlinear"):
            # conv (== conv_linear): STRICTLY-LINEAR conv transport — a learned
            # (PixelShuffle) upsample + conv residual on top of the frozen closed-form
            # base affine (do-no-harm), plus a paired linear inverse net. Adds a spatial
            # receptive field the per-pixel affine/adaln lack, while keeping the L1
            # pushforward EXACT (still linear).
            # m5 (== conv_nl/nonlinear): the SAME scaffolding but NON-LINEAR residual
            # nets (activations) + a learned (non-analytic) inverse — higher clean
            # fidelity at the cost of an only-APPROXIMATE L1 pushforward (the doc's M5
            # cycle-consistent two-network transport). Both are trained ONLY during
            # warm-up (forward+inverse+cycle recon), then frozen for L1.
            self.transport = build_transport(
                ta.vae_transport,
                teacher_to_spatial=self._teacher_to_spatial,
                teacher_from_spatial=self._teacher_from_spatial,
                student_to_spatial=self._student_to_spatial,
                student_from_spatial=self._student_from_spatial,
                teacher_channels=self._adapter_latent_channels(self.teacher_adapter),
                student_channels=self._adapter_latent_channels(self.adapter),
            )
        elif ta.vae_transport == "hsct":
            # M8: hidden-state-conditioned transport. Linear P (raw teacher->student,
            # closed-form, exact pushforward) + deepstack Q (raw student + SD3.5 hidden
            # h_S -> raw teacher). Operates on RAW VAE latents; HSCT converts the SCALED
            # rollout latent <-> raw and the teacher raw <-> packed internally.
            s_vae_cfg = self.adapter.pipeline.vae.config
            t_vae_cfg = self.teacher_adapter.pipeline.vae.config
            self.transport = build_transport(
                "hsct",
                teacher_adapter=self.teacher_adapter,
                student_to_spatial=self._student_to_spatial,
                student_from_spatial=self._student_from_spatial,
                c_T=int(t_vae_cfg.latent_channels),
                c_S=int(s_vae_cfg.latent_channels),
                student_scaling=float(s_vae_cfg.scaling_factor),
                student_shift=float(getattr(s_vae_cfg, "shift_factor", 0.0) or 0.0),
                h_proj=ta.hsct_h_proj,
                q_arch=ta.hsct_q_arch,
                q_inject=ta.hsct_q_inject,
                q_hidden=ta.hsct_q_hidden,
                q_depth=ta.hsct_q_depth,
                q_dit_dim=ta.hsct_dit_dim,
                q_dit_heads=ta.hsct_dit_heads,
                n_blocks=len(ta.hsct_hidden_blocks),
            )
            self._register_hsct_hooks()
        elif ta.vae_transport == "flow":
            # M9: conditional-flow inverse. Linear P (raw teacher->student, closed-form,
            # exact pushforward) + a conditional coupling flow Q (NLL-trained, conditioned
            # on SD3.5 hidden h_S) that samples ON-manifold teacher latents. Reuses the HSCT
            # raw<->packed bridge + cold-start; only the inverse Q differs.
            s_vae_cfg = self.adapter.pipeline.vae.config
            t_vae_cfg = self.teacher_adapter.pipeline.vae.config
            self.transport = build_transport(
                "flow",
                teacher_adapter=self.teacher_adapter,
                student_to_spatial=self._student_to_spatial,
                student_from_spatial=self._student_from_spatial,
                c_T=int(t_vae_cfg.latent_channels),
                c_S=int(s_vae_cfg.latent_channels),
                student_scaling=float(s_vae_cfg.scaling_factor),
                student_shift=float(getattr(s_vae_cfg, "shift_factor", 0.0) or 0.0),
                n_blocks=len(ta.hsct_hidden_blocks),
                cond_proj=ta.flow_cond_proj,
                flow_n_coupling_blocks=ta.flow_n_coupling_blocks,
                flow_hidden=ta.flow_hidden,
                flow_query_mode=ta.flow_query_mode,
                flow_num_samples=ta.flow_num_samples,
            )
            self._register_hsct_hooks()
        elif ta.vae_transport == "aligned":
            # M3 Stage-2 needs teacher_adapter + checkpoint_path, which this trainer does not
            # construct. Fail fast with an actionable message instead of a cryptic ctor error.
            raise NotImplementedError(
                "vae_transport='aligned' is not wired in the XOPD trainer (needs "
                "teacher_adapter + checkpoint_path). Use 'hsct', or wire AlignedTransport here."
            )
        else:  # "mlp" -> placeholder (raises in build_transport/constructor)
            self.transport = build_transport(ta.vae_transport)

    # -- Checkpointing: persist the (frozen) transport alongside the model --
    def save_checkpoint(self, save_directory: str, epoch: Optional[int] = None):
        """Save the student checkpoint plus the cross-VAE transport state.

        The transport (linear/whitening A,b or AdaLN params + grids) is frozen
        after warm-up but must be persisted so a resumed run skips re-warm-up and
        keeps an identical teacher->student mapping. Written as ``transport.pt``
        next to the model checkpoint (main process only). No-op for identity.
        """
        super().save_checkpoint(save_directory, epoch=epoch)
        if self._cross_vae and self.transport is not None and self.accelerator.is_main_process:
            ckpt_dir = save_directory
            if epoch is not None:
                ckpt_dir = os.path.join(save_directory, f"checkpoint-{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                {
                    "vae_transport": self.training_args.vae_transport,
                    "state": self.transport.state_dict(),
                },
                os.path.join(ckpt_dir, "transport.pt"),
            )
        self.accelerator.wait_for_everyone()

    def load_checkpoint(self, path: str, resume_type=None):
        """Load the student checkpoint plus the cross-VAE transport state (if present)."""
        super().load_checkpoint(path, resume_type=resume_type)
        if self._cross_vae and self.transport is not None:
            transport_path = os.path.join(path, "transport.pt")
            if os.path.isfile(transport_path):
                blob = torch.load(transport_path, map_location="cpu")
                if blob.get("vae_transport") != self.training_args.vae_transport:
                    logger.warning(
                        f"Checkpoint transport type {blob.get('vae_transport')!r} != "
                        f"config {self.training_args.vae_transport!r}; loading anyway."
                    )
                self.transport.load_state_dict(blob["state"])
                if isinstance(self.transport, torch.nn.Module):
                    self.transport.to(self.accelerator.device)
                logger.info(f"Loaded cross-VAE transport state from {transport_path}.")
            else:
                logger.warning(
                    f"No transport.pt under {path}; the transport will be re-warmed "
                    "up at start() (requires_warmup)."
                )
        self.accelerator.wait_for_everyone()

    # -- Canonical latent-layout converters (native <-> (B,C,H,W)) --
    @staticmethod
    def _adapter_latent_channels(adapter) -> int:
        """Number of VAE latent channels for an adapter (canonical BCHW channels).

        Prefers the VAE's ``latent_channels`` config; falls back to the VAE
        decoder ``in_channels`` (== latent channels). Used to size the AdaLN
        transport's per-channel parameters.
        """
        # The transport operates on the canonical patchified spatial latent that
        # `to_spatial_latent` produces, whose channel count equals the transformer
        # `in_channels` (FLUX.2: 128 = vae_latent_channels*4 after patchify; SD3.5:
        # 16 = raw VAE latent_channels, to_spatial is identity). Prefer that over
        # the VAE's raw `latent_channels` (which is pre-patchify, 32 for klein and
        # would mis-size the AdaLN params).
        transformer = getattr(adapter.pipeline, "transformer", None)
        if transformer is not None and hasattr(transformer, "config"):
            tcfg = transformer.config
            if hasattr(tcfg, "in_channels"):
                return int(tcfg.in_channels)
        vae = getattr(adapter.pipeline, "vae", None)
        if vae is not None and hasattr(vae, "config"):
            cfg = vae.config
            for key in ("latent_channels", "in_channels"):
                if hasattr(cfg, key):
                    return int(getattr(cfg, key))
        raise RuntimeError(
            f"Could not infer latent channels for {type(adapter).__name__}; "
            "AdaLN transport needs teacher/student channel counts."
        )

    @staticmethod
    def _noise_fraction(adapter, t):
        """Map a scheduler timestep to the scheduler-AGNOSTIC noise fraction sigma in [0,1].

        Flow-matching convention ``sigma = t / num_train_timesteps`` (see
        ``common.py``: ``x_t = (1-sigma) x0 + sigma eps``). ``num_train_timesteps`` is
        read from the adapter's scheduler (or its config), defaulting to 1000.
        Expressing BOTH the warm-up (teacher trajectory) and L1 (student state) noise
        levels as ``sigma`` decouples the adaLN modulation from each scheduler's
        timestep scaling/shift, so a teacher/student mismatch can no longer skew it.
        Returns a tensor when ``t`` is a tensor, else a float; both clamped to [0,1].
        """
        sched = getattr(adapter, "scheduler", None)
        N = getattr(sched, "num_train_timesteps", None) if sched is not None else None
        if N is None and sched is not None and hasattr(sched, "config"):
            N = getattr(sched.config, "num_train_timesteps", None)
        N = float(N) if N else 1000.0
        if torch.is_tensor(t):
            return (t.float() / N).clamp(0.0, 1.0)
        return float(min(max(float(t) / N, 0.0), 1.0))

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
        # FLUX.2 to_spatial_latent needs the packed latents' position ids. They are
        # fixed for a given resolution (all geneval prompts share one), so we cache
        # them at warm-up (_capture_teacher_layout) and inject here. SD3.5 student is
        # already BCHW (no to_spatial_latent) so the student path ignores this.
        if "latent_ids" not in ctx and self._teacher_latent_ids is not None:
            ids = self._teacher_latent_ids
            # _unpack_latents_with_ids zips x (batch) with x_ids (batch): broadcast
            # the single cached id-row to z's batch size.
            if ids.dim() == 3 and ids.shape[0] == 1 and z.dim() == 3 and z.shape[0] > 1:
                ids = ids.expand(z.shape[0], -1, -1)
            ctx = {**ctx, "latent_ids": ids.to(z.device)}
            if self._teacher_spatial_hw is not None:
                ctx.setdefault("height", self._teacher_spatial_hw[0])
                ctx.setdefault("width", self._teacher_spatial_hw[1])
        sp = self._adapter_to_spatial(self.teacher_adapter, z, **ctx)
        if getattr(self.training_args, "transport_teacher_unshuffle", False):
            # 128ch@32x32 (2x2 patchify of the 32ch VAE latent) -> 32ch@64x64. Lossless
            # reshape: aligns the teacher latent to the student 64x64 grid (no bilinear)
            # and cuts the channel ratio 128:16 -> 32:16. Paired with PixelUnshuffle(2) in
            # _teacher_from_spatial so t_from(t2s(x)) == x (the teacher query stays exact).
            sp = torch.nn.functional.pixel_shuffle(sp, 2)
        return sp

    def _teacher_from_spatial(self, z, **ctx):
        if getattr(self.training_args, "transport_teacher_unshuffle", False):
            z = torch.nn.functional.pixel_unshuffle(z, 2)
        return self._adapter_from_spatial(self.teacher_adapter, z, **ctx)

    def _student_to_spatial(self, z, **ctx):
        return self._adapter_to_spatial(self.adapter, z, **ctx)

    def _student_from_spatial(self, z, **ctx):
        return self._adapter_from_spatial(self.adapter, z, **ctx)

    # ----- M8 HSCT: student-transformer hidden-state capture --------------------
    def _register_hsct_hooks(self) -> None:
        """Register forward hooks on the student transformer blocks to capture h_S.

        The SD3 JointTransformerBlock returns ``(encoder_hidden_states, hidden_states)``;
        we grab the image stream ``[1]``. Capture is gated by ``self._hsct_capture`` so
        only the L1 pre-pass / cold-start forwards pay the (tiny) cost; sampling/eval
        forwards skip it. Hooks live on the underlying block modules so they survive
        accelerate/deepspeed wrapping.
        """
        transformer = getattr(self.adapter.pipeline, "transformer", None)
        if transformer is None or not hasattr(transformer, "transformer_blocks"):
            raise RuntimeError(
                "HSCT needs self.adapter.pipeline.transformer.transformer_blocks to hook "
                f"hidden states; got {type(self.adapter).__name__}."
            )
        blocks = transformer.transformer_blocks
        for b in self.training_args.hsct_hidden_blocks:
            if not (0 <= b < len(blocks)):
                raise ValueError(f"hsct_hidden_blocks entry {b} out of range [0,{len(blocks)})")

            def _mk(bi):
                def _hook(_m, _i, out):
                    if self._hsct_capture:
                        self._hsct_hidden[bi] = out[1] if isinstance(out, (tuple, list)) else out

                return _hook

            blocks[b].register_forward_hook(_mk(b))

    def _hsct_h_list(self, latent_bs: int) -> List[torch.Tensor]:
        """Reshape captured per-block hidden ``(B|2B, N, D)`` -> list of ``(B, D, s, s)``.

        Under CFG the transformer runs on ``cat[uncond, cond]`` (2B); take the cond
        half so h_S matches the (positive-prompt) conditioning the inverse needs.
        """
        out = []
        for b in self.training_args.hsct_hidden_blocks:
            if b not in self._hsct_hidden:
                raise RuntimeError(
                    f"HSCT hidden for block {b} not captured; was _hsct_capture set?"
                )
            h = self._hsct_hidden[b]
            if h.shape[0] == 2 * latent_bs:
                h = h[latent_bs:]
            B, N, D = h.shape
            s = int(round(N**0.5))
            if s * s != N:
                raise ValueError(f"non-square token count N={N} at block {b}")
            out.append(h.reshape(B, s, s, D).permute(0, 3, 1, 2).contiguous())
        return out

    def _resolve_teacher_vae_path(self) -> Optional[str]:
        """Resolve where to load the cross-VAE teacher's VAE from.

        Explicit ``teacher_vae_name_or_path`` wins. Otherwise, if the teacher repo
        ships no ``vae/`` subfolder (FLUX.2-klein-base-9B is a transformer-only
        release that shares the 4B VAE), auto-fall back to the klein 4B VAE. Returns
        None when the teacher repo has its own VAE (e.g. FLUX.2-dev) — load as usual.
        """
        ta = self.training_args
        if ta.teacher_vae_name_or_path:
            return ta.teacher_vae_name_or_path
        if (ta.teacher_model_type or "").startswith("flux2-klein"):
            # klein-9B / klein-base teacher repos are transformer-only; use 4B VAE.
            try:
                from huggingface_hub import snapshot_download

                snap = snapshot_download(ta.teacher_model_name_or_path, local_files_only=True)
                if os.path.isdir(os.path.join(snap, "vae")):
                    return None  # teacher repo has its own VAE
            except Exception:
                pass
            return "black-forest-labs/FLUX.2-klein-base-4B"
        return None

    def _capture_teacher_layout(self, latent_ids: torch.Tensor, hw=None) -> None:
        """Cache the teacher's packed-latent position ids / spatial size.

        FLUX.2 ``to_spatial_latent`` needs the position ids to scatter packed tokens
        onto their (h, w) grid. They are constant for a fixed resolution (geneval),
        so we snapshot them once (from a teacher rollout during warm-up) and reuse
        them for every affine-transport conversion in L1.
        """
        if latent_ids is None:
            return
        ids = latent_ids.detach()
        # Keep a single (1, seq, K) row; the converter broadcasts over the batch via
        # the per-sample zip in _unpack_latents_with_ids (it iterates rows).
        if ids.dim() == 3 and ids.shape[0] > 1:
            ids = ids[:1]
        self._teacher_latent_ids = ids.to(self.accelerator.device)
        if hw is not None:
            self._teacher_spatial_hw = (int(hw[0]), int(hw[1]))

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
        # the teacher text encoder is freed before training. The encode+cache hooks
        # (load_teacher_text_encoder / unload_teacher_text_encoder /
        # encode_teacher_prompt + the preprocess_func teacher branch) live on
        # BaseAdapter, so any student adapter (SD3.5, FLUX.2-klein, ...) supports
        # this cross-architecture precompute+offload.
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
    def _base_train_timestep_indices(self):
        """Base per-rollout training-step indices BEFORE xopd_train_steps selection.

        All steps for ODE, scheduler-selected (post-SDE) for SDE. This is the pool
        that ``xopd_train_steps`` indexes into.
        """
        if self._is_ode:
            return list(range(self.training_args.num_inference_steps))
        ts = self.adapter.scheduler.train_timesteps
        return ts.tolist() if hasattr(ts, "tolist") else list(ts)

    @property
    def _train_timestep_indices(self):
        """L1 training-timestep indices (subset of the base list).

        ``xopd_train_steps`` (config) selects k_idx positions of the base list; with
        ``num_xopd_steps`` a fixed-size random subset is drawn PER EPOCH (deterministic
        via epoch+seed, parallel to scheduler.current_sde_steps). The selection is
        CACHED per epoch so every consumer in one epoch (sample / D_k pre-pass /
        optimize) sees the SAME steps; the subset SIZE is fixed (= num_xopd_steps or
        len(xopd_train_steps)) so GAS / one-step-per-epoch stays consistent. null/null
        => the full base list (legacy behavior).
        """
        ta = self.training_args
        base = self._base_train_timestep_indices

        # Candidate pool = xopd_train_steps positions into base, else the whole base.
        if ta.xopd_train_steps:
            if any(i >= len(base) for i in ta.xopd_train_steps):
                raise ValueError(
                    f"xopd_train_steps={ta.xopd_train_steps} has an index >= the number of "
                    f"base training steps ({len(base)} for "
                    f"{'ODE' if self._is_ode else 'SDE'}). Indices are 0-based positions "
                    "into the per-rollout training-step list."
                )
            pool = [base[i] for i in ta.xopd_train_steps]
        else:
            pool = list(base)

        # No random subsampling -> use the whole pool.
        if ta.num_xopd_steps is None or ta.num_xopd_steps >= len(pool):
            return pool

        # Per-epoch deterministic random subset (cached so all consumers agree).
        epoch = getattr(self, "epoch", 0)
        if getattr(self, "_xopd_tts_cache_epoch", None) != epoch:
            g = torch.Generator().manual_seed(int(epoch) + int(ta.seed))
            if ta.xopd_step_sampling == "stratified":
                sel = self._stratified_pool_indices(len(pool), ta.num_xopd_steps, g)
            else:
                sel = sorted(
                    torch.randperm(len(pool), generator=g)[: ta.num_xopd_steps].tolist()
                )  # ascending trajectory order
            self._xopd_tts_cache = [pool[i] for i in sel]
            self._xopd_tts_cache_epoch = epoch
        return self._xopd_tts_cache

    @staticmethod
    def _stratified_pool_indices(n_pool: int, k: int, generator) -> List[int]:
        """Split ``range(n_pool)`` into k contiguous equal segments and pick one random
        index per segment (ascending). Guarantees one step from each k-quantile of the
        trajectory (e.g. n_pool=28, k=4 -> one index from each of [0,7) [7,14) [14,21)
        [21,28)). Boundaries via ``round(i*n_pool/k)`` so uneven splits differ by <=1.
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"stratified sampling: k must be a positive int, got {k!r}.")
        if k > n_pool:
            raise ValueError(
                f"stratified sampling: k={k} cannot exceed pool size n_pool={n_pool} "
                "(need at least one candidate step per segment)."
            )
        bounds = [round(i * n_pool / k) for i in range(k + 1)]
        out: List[int] = []
        for j in range(k):
            lo, hi = bounds[j], bounds[j + 1]
            if hi <= lo:
                raise ValueError(
                    f"stratified sampling: empty segment {j} ([{lo},{hi})) for "
                    f"n_pool={n_pool}, k={k}."
                )
            out.append(int(torch.randint(lo, hi, (1,), generator=generator).item()))
        return out  # already ascending (segments are contiguous & ascending)

    @property
    def _candidate_train_timestep_indices(self) -> List[int]:
        """Full L1 candidate pool (base indices), BEFORE any num_xopd_steps subsampling.

        For ``xopd_resample_steps_per_batch`` the rollout must store latents for EVERY
        candidate step (any per-batch draw is served from these), and each micro-batch
        draws its k steps from this pool. = ``xopd_train_steps`` positions into the base
        list, else the whole base list.
        """
        ta = self.training_args
        base = self._base_train_timestep_indices
        if ta.xopd_train_steps:
            return [base[i] for i in ta.xopd_train_steps]
        return list(base)

    def _draw_batch_steps(self, batch_idx: int) -> List[int]:
        """Per-micro-batch random subset of ``num_xopd_steps`` candidate steps (sorted).

        Deterministic in ``(epoch, batch_idx, seed)`` so every rank draws the SAME steps
        for a given ``batch_idx`` (each rank applies them to its own samples). The count is
        fixed = ``num_xopd_steps``, so every rank makes exactly ``num_batches * k`` accumulate
        calls and the GAS / one-optimizer-step-per-epoch boundary stays aligned across ranks.
        """
        ta = self.training_args
        pool = self._candidate_train_timestep_indices
        k = ta.num_xopd_steps
        if k is None or k >= len(pool):
            return list(pool)
        epoch = int(getattr(self, "epoch", 0))
        g = torch.Generator().manual_seed(int(ta.seed) + 100003 * epoch + 97 * int(batch_idx))
        if ta.xopd_step_sampling == "stratified":
            sel = self._stratified_pool_indices(len(pool), k, g)
        else:
            sel = sorted(torch.randperm(len(pool), generator=g)[:k].tolist())
        return [pool[i] for i in sel]

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
    def _maybe_coldstart_mof_router(self) -> None:
        """Cluster-based one-hot cold-start of the MoF-V global router (data-parallel).

        UNSUPERVISED: cluster the train ``prompt_embeds`` into ``mof_num_experts``
        groups (balanced spherical k-means) and cold-start the router (router-only
        CE to the cluster labels) so each expert binds a distinct prompt cluster
        from the start. Dataset source labels are NEVER used as targets/features
        (only post-hoc ARI/purity). Bypasses the DeepSpeed/DDP engine (plain torch
        AdamW on the router params + manual grad all_reduce); cleans grads after.
        """
        import json

        import torch.distributed as dist

        if not getattr(self.model_args, "mof_router_coldstart", False):
            return
        mof = self.adapter._unwrap(self.adapter.transformer)
        if hasattr(mof, "_orig_mod"):  # torch.compile
            mof = mof._orig_mod
        if not hasattr(mof, "router") or getattr(mof.config, "router_type", None) != "global":
            raise RuntimeError(
                "mof_router_coldstart=True but the student is not a MoF-V with a 'global' router "
                f"(got {type(mof).__name__}, router_type={getattr(mof.config, 'router_type', None)!r})."
            )

        acc = self.accelerator
        device = acc.device
        world_size = acc.num_processes
        rank = acc.process_index
        is_main = acc.is_main_process
        ma = self.model_args
        n_clusters = int(mof.config.num_experts)
        logger.info(
            "[mof-coldstart] start: experts=%d balanced=%s max_samples=%d steps=%d (world=%d)",
            n_clusters, ma.mof_cluster_balanced, ma.mof_cluster_max_samples,
            ma.mof_router_coldstart_steps, world_size,
        )

        # --- (a) gather this rank's shard of train prompts from the cache ---
        if self.train_dataloaders_by_source:
            datasets = {src: dl.dataset for src, dl in self.train_dataloaders_by_source.items()}
        elif self.dataloader is not None:
            src0 = os.path.basename(str(self.config.data_args.dataset_dir))
            datasets = {src0: self.dataloader.dataset}
        else:
            raise RuntimeError("mof_router_coldstart requires a train dataloader/dataset.")

        per_rank = max(1, ma.mof_cluster_max_samples // world_size)
        entries: List[Tuple[str, Any, int]] = []
        for src, ds in datasets.items():
            for i in range(rank, len(ds), world_size):
                entries.append((src, ds, i))
        eg = torch.Generator().manual_seed(int(self.training_args.seed) + 7 * rank)
        sel = torch.randperm(len(entries), generator=eg)[:per_rank].tolist()

        pad_id = getattr(self.adapter.pipeline.tokenizer, "pad_token_id", None)
        seq_list: List[torch.Tensor] = []
        pooled_list: List[torch.Tensor] = []
        sources_local: List[str] = []
        for j in sel:
            src, ds, i = entries[j]
            sample = ds[i]
            pe = sample["prompt_embeds"]  # (L, D)
            if not isinstance(pe, torch.Tensor):
                raise TypeError(f"prompt_embeds for source {src!r} idx {i} is {type(pe).__name__}, expected Tensor.")
            pe = pe.detach()
            pids = sample.get("prompt_ids", None)
            mask = None
            if pad_id is not None and isinstance(pids, torch.Tensor):
                mask = (pids != pad_id).unsqueeze(0)  # (1, L)
            pooled = pool_prompt_embeds(pe.unsqueeze(0).float(), mask)[0]  # (D,)
            seq_list.append(pe.to(torch.float16).cpu())
            pooled_list.append(pooled.cpu())
            sources_local.append(src)
        if not pooled_list:
            raise RuntimeError(f"[mof-coldstart] rank {rank} gathered 0 prompts; check the train cache.")
        prompt_seq_local = torch.stack(seq_list, dim=0)   # (m, L, D) fp16 CPU
        pooled_local = torch.stack(pooled_list, dim=0)    # (m, D) fp32 CPU

        # --- gather pooled embeds (+ sources) across ranks ---
        if world_size > 1:
            gathered_pooled: List[Any] = [None] * world_size
            gathered_src: List[Any] = [None] * world_size
            dist.all_gather_object(gathered_pooled, pooled_local)
            dist.all_gather_object(gathered_src, sources_local)
            counts = [int(g.shape[0]) for g in gathered_pooled]
            X_all = torch.cat([g.to(device) for g in gathered_pooled], dim=0)
            sources_all = [s for chunk in gathered_src for s in chunk]
            offset = sum(counts[:rank])
        else:
            counts = [pooled_local.shape[0]]
            X_all = pooled_local.to(device)
            sources_all = list(sources_local)
            offset = 0
        total_m = X_all.shape[0]

        # --- (b) rank0 clusters -> labels_all; broadcast; each rank slices its block ---
        use_soft = getattr(ma, "mof_router_coldstart_label", "hard") == "soft"
        soft_T = float(getattr(ma, "mof_router_coldstart_temperature", 0.5))
        labels_all = torch.zeros(total_m, dtype=torch.long, device=device)
        soft_all = torch.zeros(total_m, n_clusters, dtype=torch.float32, device=device) if use_soft else None
        metrics = None
        if is_main:
            lab, _, sim = balanced_spherical_kmeans(
                X_all, n_clusters,
                balanced=ma.mof_cluster_balanced,
                pca_dim=ma.mof_cluster_pca_dim,
                seed=int(self.training_args.seed),
            )
            labels_all = lab.to(device)
            if use_soft:
                # soft cluster responsibilities as the cold-start target (softmax over cosine sims / T)
                soft_all = torch.softmax(sim.to(device).float() / soft_T, dim=-1)
            metrics = alignment_metrics(labels_all, sources_all)
            logger.info(
                "[mof-coldstart] clustering done: sizes=%s ARI=%.4f purity=%.4f (vs sources %s) "
                "| target=%s%s -- source labels are DIAGNOSTIC ONLY (not used for clustering/CE)",
                metrics["cluster_sizes"], metrics["ari"], metrics["purity"], metrics["sources"],
                ("soft(T=%.3g)" % soft_T) if use_soft else "hard(one-hot)", "",
            )
        if world_size > 1:
            dist.broadcast(labels_all, src=0)
            if use_soft:
                dist.broadcast(soft_all, src=0)
        m_local = prompt_seq_local.shape[0]
        labels_local = labels_all[offset:offset + m_local].to("cpu")
        soft_local = soft_all[offset:offset + m_local].to("cpu") if use_soft else None

        # --- (c) data-parallel router cold-start (bypasses DeepSpeed/DDP engine) ---
        router = mof.router
        router_time_embed = mof.router_time_embed
        params = [p for p in list(router.parameters()) + list(router_time_embed.parameters())]
        for p in params:
            p.requires_grad_(True)
        in_ch = int(mof.config.in_channels)
        guidance_embeds = bool(getattr(mof.config, "guidance_embeds", False))
        student_gs = float(self.student_gs)
        tgen = torch.Generator(device=device).manual_seed(int(self.training_args.seed) + 13 * rank)

        def router_forward_fn(pe_batch: torch.Tensor) -> torch.Tensor:
            b = pe_batch.shape[0]
            # Resolve the dtype from the LIVE parameters on every call: under FSDP mixed precision
            # the sharded parameter is bf16 outside an unshard context but summon_full_params
            # materializes the fp32 master, so a dtype captured before the context would feed bf16
            # inputs into fp32 weights.
            rdtype = next(router.parameters()).dtype
            pe_batch = pe_batch.to(device=device, dtype=rdtype)
            t = torch.rand(b, device=device, generator=tgen)
            ts = (t * 1000).to(rdtype)
            g = (
                torch.full((b,), student_gs * 1000, device=device, dtype=rdtype)
                if guidance_embeds else None
            )
            temb = router_time_embed(ts, g)
            dummy_latent = torch.zeros(b, 1, in_ch, device=device, dtype=rdtype)
            # MoFGlobalRouter.forward returns RAW logits -> softmax here so cold-start CE trains a
            # proper distribution (classify prompt into a cluster), independent of the train-time
            # gate_fn (softmax/sigmoid).
            logits = router(pe_batch, dummy_latent, temb)  # (b, N) raw logits
            return torch.softmax(logits.float(), dim=-1)   # (b, N) probs

        log_f = None
        default_path = os.path.join(
            os.path.expanduser(str(self.log_args.save_dir)),
            f"mof_router_coldstart_{self.log_args.run_name}.jsonl",
        )
        log_path = ma.mof_router_coldstart_log_path or default_path
        if is_main:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            log_f = open(log_path, "a")

        def log_cb(step: int, ce: float, acc_v: float, maxprob: float) -> None:
            if not is_main:
                return
            logger.info(
                "[mof-coldstart] step=%d ce=%.4f acc=%.4f maxprob=%.4f", step, ce, acc_v, maxprob
            )
            log_f.write(json.dumps({"step": step, "ce": ce, "acc": acc_v, "maxprob": maxprob}) + "\n")
            log_f.flush()

        # Under FSDP the router is not its own wrap unit (TRANSFORMER_BASED_WRAP only wraps the
        # blocks), so it lives in the ROOT unit's flat parameter: outside an unshard context each
        # rank holds just its slice and the router forward dies on a size mismatch. Materialize the
        # root unit for the cold-start -- recurse=False keeps that to the small non-block remainder
        # rather than the 8B of expert blocks -- and let writeback scatter the result back into the
        # shards. The replica broadcast below must also happen INSIDE the context: on full
        # parameters it is the intended no-op safety sync, whereas on shards it would overwrite
        # every rank with rank 0's slice.
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        prepared = self.adapter.transformer
        unshard = (
            FSDP.summon_full_params(prepared, recurse=False, writeback=True, with_grads=True)
            if isinstance(prepared, FSDP) else nullcontext()
        )
        with unshard:
            coldstart_router(
                router_forward_fn, params, prompt_seq_local, labels_local,
                steps=ma.mof_router_coldstart_steps, lr=ma.mof_router_coldstart_lr,
                batch_size=ma.mof_router_coldstart_batch, world_size=world_size, device=device,
                log_every=ma.mof_router_coldstart_log_every, log_cb=log_cb,
                seed=int(self.training_args.seed), soft_targets=soft_local,
            )

            # --- (d) sync router replicas ---
            if world_size > 1:
                for p in params:
                    dist.broadcast(p.data, src=0)

        # --- clear grads, log summary ---
        for p in self.adapter.transformer.parameters():
            p.grad = None
        if is_main and log_f is not None:
            summary = {"event": "summary", "steps": ma.mof_router_coldstart_steps}
            if metrics is not None:
                summary.update({
                    "ari": metrics["ari"], "purity": metrics["purity"],
                    "cluster_sizes": metrics["cluster_sizes"], "sources": metrics["sources"],
                })
            log_f.write(json.dumps(summary) + "\n")
            log_f.close()
        acc.wait_for_everyone()
        logger.info("[mof-coldstart] done; router cold-started to %d prompt clusters.", n_clusters)

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
            if self._is_hsct:
                self._coldstart_hsct()  # M8: train linear-P + deepstack-Q on corpus h_S
            else:
                self._warmup_transport()

        # Cluster-based one-hot cold-start of the MoF-V router (once, epoch 0).
        if self.epoch == 0:
            self._maybe_coldstart_mof_router()

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
        # Batch the whole student eval + teacher re-emit into ONE log call so all
        # test sets land on the SAME wandb step (previously each test set logged
        # separately, spreading eval metrics across stepless auto-incremented steps).
        self._eval_log_sink = {}
        # Cache of the trajectory-storing student rollouts done by the reward-eval
        # pass on the gs=1.0 sets (populated in _run_eval_inference_batches), so the
        # validation-D_k pass reuses them instead of re-rolling. Cleared at the end.
        self._eval_rollout_cache = {}
        super().evaluate()
        # Validation D_k on the gs=1.0 sets (held-out transition-matching loss),
        # written into the same sink so it shares the student/teacher eval step.
        self._evaluate_validation_d_k()
        if self.accelerator.is_main_process:
            if self._teacher_baseline_scalars:
                self._eval_log_sink.update(self._teacher_baseline_scalars)
            if self._eval_log_sink:
                self.log_data(self._eval_log_sink, step=self.step)
        self._eval_log_sink = None
        self._eval_rollout_cache = None

    def _run_eval_inference_batches(
        self,
        test_set_name: str,
        merged_eval: Any,
        eval_seed: int,
    ) -> List[BaseSample]:
        """XOPD override of the base student eval rollout.

        For the no-CFG (gs=1.0) eval sets we roll out WITH ``trajectory_indices``
        so the FULL trajectory is stored, and cache the resulting samples
        (CPU-offloaded) on ``self._eval_rollout_cache`` so
        :meth:`_evaluate_validation_d_k` reuses this SINGLE rollout for the
        per-timestep D_k instead of running an identical second rollout. The
        final image is unchanged by trajectory storage, so the reward metrics are
        identical to the base path. gs!=1.0 sets keep the base behavior exactly
        (``trajectory_indices=None``, final image only, no cache).

        The cached samples are moved to CPU on purpose: with 3 gs=1.0 sets scored
        before the D_k pass runs, keeping every set's full trajectory resident on
        GPU would spike VRAM; the D_k pass re-uploads one micro-batch at a time
        (identical to the reward path under ``offload_samples_to_cpu``).
        """
        gs = float(getattr(merged_eval, "guidance_scale", 1.0))
        store_traj = abs(gs - 1.0) <= 1e-6
        trajectory_indices = None
        if store_traj:
            eval_steps = int(
                getattr(merged_eval, "num_inference_steps", self.training_args.num_inference_steps)
            )
            trajectory_indices = compute_trajectory_indices(
                train_timestep_indices=list(range(eval_steps)),
                num_inference_steps=eval_steps,
                include_initial=True,
            )

        all_samples: List[BaseSample] = []
        for batch in tqdm(
            self.test_dataloaders[test_set_name],
            desc=self._eval_progress_desc(test_set_name),
            disable=not self.show_progress_bar,
        ):
            generator = create_generator_by_prompt(batch["prompt"], eval_seed)
            inference_kwargs = {
                "compute_log_prob": False,
                "generator": generator,
                "trajectory_indices": trajectory_indices,
                **merged_eval,
            }
            inference_kwargs.update(**batch)
            inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
            samples = self.adapter.inference(**inference_kwargs)

            # Stitch dataset metadata onto generated samples for reward routing.
            stitch_batch_metadata(batch, samples)

            all_samples.extend(samples)
            self.eval_reward_buffer.add_samples(samples)

        # Cache the trajectory-storing rollout for the validation-D_k reuse. Only
        # for gs=1.0 sets and only when evaluate() initialized the cache. CPU-offload
        # to bound VRAM; the reward path (already returned below) and the D_k pass
        # both move tensors back to device as needed.
        cache = getattr(self, "_eval_rollout_cache", None)
        if store_traj and cache is not None:
            for sample in all_samples:
                sample.to("cpu")
            cache[test_set_name] = all_samples
        return all_samples

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

        self._eval_debug_gloo_barrier("teacher_baseline:begin")
        self.adapter.eval()
        # Only baseline the teacher on test sets flagged eval_teacher=True (default).
        # OOD/off-domain sets (e.g. geneval/ocr/pickscore during an I2I gedit run) set
        # eval_teacher=false to skip the expensive 32B-teacher generation there; the
        # STUDENT is still evaluated on them every eval to track OOD drift. All ranks
        # derive the same set from config, so the collective structure stays aligned.
        teacher_test_sets = [
            name
            for name in sorted(self.test_dataloaders.keys())
            if getattr(self._test_sets_by_name.get(name), "eval_teacher", True)
        ]
        skipped = sorted(set(self.test_dataloaders.keys()) - set(teacher_test_sets))
        logger.info(
            "XOPD: evaluating teacher baseline on %s (eval_teacher=true); "
            "skipping %s (eval_teacher=false)",
            teacher_test_sets,
            skipped or "none",
        )
        # Batch all test sets into ONE log call so the teacher baseline lands on a
        # single wandb step (step 0), not spread across stepless auto-steps.
        self._eval_log_sink = {}
        for test_set_name in teacher_test_sets:
            self._eval_debug_gloo_barrier(f"teacher_test_set:{test_set_name}:begin")
            self._log_eval_dist_debug(test_set_name, "teacher_test_set:begin")
            self._evaluate_teacher_test_set(test_set_name)
            self._log_eval_dist_debug(test_set_name, "teacher_test_set:end")
        if self.accelerator.is_main_process and self._eval_log_sink:
            self._log_eval_dist_debug(None, "teacher_wandb_log:begin")
            self.log_data(self._eval_log_sink, step=self.step)
            self._log_eval_dist_debug(None, "teacher_wandb_log:end")
        self._eval_log_sink = None
        self._log_eval_dist_debug(None, "teacher_baseline_barrier:begin")
        self.accelerator.wait_for_everyone()
        self._log_eval_dist_debug(None, "teacher_baseline_barrier:end")

    def _eval_debug_gloo_barrier(self, stage: str) -> None:
        """Debug-only CPU barrier that cannot consume the NCCL collective stream."""
        if os.environ.get("FLOW_FACTORY_EVAL_GLOO_BARRIER") != "1":
            return
        if self.accelerator.num_processes <= 1:
            return

        from datetime import timedelta

        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                "FLOW_FACTORY_EVAL_GLOO_BARRIER=1 requires an initialized "
                "torch.distributed process group, but dist.is_initialized() is False."
            )
        if not hasattr(self, "_eval_debug_gloo_group"):
            self._log_eval_dist_debug(None, "gloo_group:create_begin", target_stage=stage)
            self._eval_debug_gloo_group = dist.new_group(backend="gloo")
            self._log_eval_dist_debug(None, "gloo_group:create_end", target_stage=stage)

        self._log_eval_dist_debug(None, "gloo_barrier:begin", target_stage=stage)
        dist.monitored_barrier(
            group=self._eval_debug_gloo_group,
            timeout=timedelta(minutes=15),
            wait_all_ranks=True,
        )
        self._log_eval_dist_debug(None, "gloo_barrier:end", target_stage=stage)

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

        # Cross-VAE: the teacher is an independent adapter (its own VAE/TE), so it
        # rolls out directly from prompts in ITS OWN latent space — no whole-module
        # swap (use_teacher_transformer is klein-only) and no cached teacher_* on the
        # heterogeneous student batch. Same-arch: swap the teacher transformer into
        # the student pipeline and route cached teacher embeddings.
        if self._cross_vae:
            with torch.no_grad(), self.autocast():
                self._log_eval_dist_debug(test_set_name, "teacher_inference:begin")
                all_samples = self._run_teacher_eval_inference_batches_cross_vae(
                    test_set_name, merged_eval, eval_seed
                )
                self._log_eval_dist_debug(
                    test_set_name,
                    "teacher_inference:end",
                    local_samples=len(all_samples),
                )
                gathered_rewards = self._gather_eval_rewards(test_set_name)
                gathered_tags = self._gather_eval_tags(all_samples, test_set_name)
                if self.accelerator.is_main_process:
                    self._log_eval_dist_debug(test_set_name, "teacher_metric_log:begin")
                    self._log_eval_reward_metrics(
                        gathered_rewards, log_pfx, all_samples, gathered_tags=gathered_tags
                    )
                    for key, value in gathered_rewards.items():
                        self._teacher_baseline_scalars[f"{log_pfx}/reward_{key}_mean"] = float(
                            np.mean(value)
                        )
                        self._teacher_baseline_scalars[f"{log_pfx}/reward_{key}_std"] = float(
                            np.std(value)
                        )
                    self._log_eval_dist_debug(test_set_name, "teacher_metric_log:end")
            self._log_eval_dist_debug(test_set_name, "teacher_test_set_barrier:begin")
            self.accelerator.wait_for_everyone()
            self._log_eval_dist_debug(test_set_name, "teacher_test_set_barrier:end")
            return

        with torch.no_grad(), self.autocast(), self.adapter.use_teacher_transformer():
            self._log_eval_dist_debug(test_set_name, "teacher_inference:begin")
            all_samples = self._run_teacher_eval_inference_batches(
                test_set_name, merged_eval, eval_seed
            )
            self._log_eval_dist_debug(
                test_set_name,
                "teacher_inference:end",
                local_samples=len(all_samples),
            )
            gathered_rewards = self._gather_eval_rewards(test_set_name)
            gathered_tags = self._gather_eval_tags(all_samples, test_set_name)
            if self.accelerator.is_main_process:
                self._log_eval_dist_debug(test_set_name, "teacher_metric_log:begin")
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
                self._log_eval_dist_debug(test_set_name, "teacher_metric_log:end")
        self._log_eval_dist_debug(test_set_name, "teacher_test_set_barrier:begin")
        self.accelerator.wait_for_everyone()
        self._log_eval_dist_debug(test_set_name, "teacher_test_set_barrier:end")

    def _run_teacher_eval_inference_batches_cross_vae(
        self,
        test_set_name: str,
        merged_eval: Any,
        eval_seed: int,
    ) -> List[BaseSample]:
        """Cross-VAE teacher baseline: roll out the independent teacher adapter.

        The teacher is a separate frozen adapter with its own VAE/scheduler/text
        encoder, so it generates from the raw prompt strings in its own latent
        space and decodes to images with its own VAE. Images are scored by the same
        reward as the student (reward models consume decoded images, not latents),
        giving a fair teacher reference. No ``teacher_*`` cache and no transformer
        swap are involved.
        """
        all_samples: List[BaseSample] = []
        teacher_gs = float(getattr(merged_eval, "guidance_scale", self.teacher_gs))
        for batch in tqdm(
            self.test_dataloaders[test_set_name],
            desc=f"Teacher eval (cross-VAE) [{test_set_name}]",
            disable=not self.show_progress_bar,
        ):
            generator = create_generator_by_prompt(batch["prompt"], eval_seed)
            infer_kwargs = {
                "prompt": batch["prompt"],
                "negative_prompt": batch.get("negative_prompt"),
                "guidance_scale": teacher_gs,
                "num_inference_steps": getattr(
                    merged_eval, "num_inference_steps", self.training_args.num_inference_steps
                ),
                "compute_log_prob": False,
                "generator": generator,
                "trajectory_indices": None,
                # I2I: pass pixel condition so the independent teacher VAE
                # re-encodes. T2I batches omit these keys (safe no-op).
                **extract_i2i_condition_kwargs(batch, prefer_latents=False),
            }
            if getattr(merged_eval, "condition_image_size", None) is not None:
                infer_kwargs["condition_image_size"] = merged_eval.condition_image_size
            infer_kwargs = filter_kwargs(self.teacher_adapter.inference, **infer_kwargs)
            samples = self.teacher_adapter.inference(**infer_kwargs)
            stitch_batch_metadata(batch, samples)
            all_samples.extend(samples)
            self.eval_reward_buffer.add_samples(samples)
        return all_samples

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
                inference_kwargs["negative_prompt_embeds"] = batch["teacher_negative_prompt_embeds"]
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

                    # Shared-VAE I2I: reuse student-encoded image_latents for both
                    # teacher and student velocity (plan: assume_shared_vae). T2I
                    # batches yield an empty dict.
                    i2i_kwargs = extract_i2i_condition_kwargs(
                        prompt_batch, prefer_latents=True, device=device
                    )
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
                                **{
                                    k: v
                                    for k, v in i2i_kwargs.items()
                                    if k in ("image_latents", "image_latent_ids")
                                },
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
                            **{
                                k: v
                                for k, v in i2i_kwargs.items()
                                if k in ("image_latents", "image_latent_ids")
                            },
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
                        grad_norm = self._clip_grad_norm_ep_aware(
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
            # I2I: independent teacher must see the condition image. Prefer
            # pixels (re-encode with teacher VAE); T2I batches skip cleanly.
            **extract_i2i_condition_kwargs(prompt_batch, prefer_latents=False),
        }
        infer_kwargs = filter_kwargs(self.teacher_adapter.inference, **infer_kwargs)
        with torch.no_grad(), self.autocast():
            return self.teacher_adapter.inference(**infer_kwargs)

    def _teacher_rollout_images(self, prompt_batch: Dict[str, Any]) -> torch.Tensor:
        """Cross-VAE: teacher images ``(B,3,H,W)`` for the pixel bridge (L0)."""
        teacher_samples = self._teacher_rollout_samples(prompt_batch)
        return torch.stack([s.image for s in teacher_samples], dim=0)

    def _collect_warmup_pairs(self):
        """Roll out ``transport_warmup_batches`` FRESH paired (z_T, z_S) latents.

        Called ONCE PER WARM-UP EPOCH (online schedule), so transport_warmup_batches
        is the per-epoch rollout count. For each teacher rollout we pair teacher
        native latents with student-space latents via the pixel bridge
        (``z_S = encode_pixels(decode(z_T))``).

        ``transport_warmup_trajectory`` (default True): pair EVERY denoising step's
        latent ``z_t^T`` (decode -> student encode), so the transport sees all noise
        levels (matching L1's use on noisy student states). False: only the final
        clean latent ``z0`` (legacy). CPU-resident lists to bound VRAM; each list
        entry is one (B, ...) pair tensor (one per (batch, timestep) in trajectory mode).

        Returns ``(z_T_list, z_S_list, sigma_list, img_list)`` where ``sigma_list[i]``
        is the scheduler-agnostic NOISE FRACTION ``sigma = t / num_train_timesteps in
        [0, 1]`` of the ``i``-th pair (teacher trajectory) — used to condition the
        adaLN-Zero modulation consistently with the L1 student noise level.
        """
        ta = self.training_args
        device = self.accelerator.device
        use_traj = getattr(ta, "transport_warmup_trajectory", False)
        data_iter = self._make_train_iter()
        z_T_list, z_S_list, sigma_list, img_list = [], [], [], []
        for _ in tqdm(
            range(ta.transport_warmup_batches),
            desc="Transport warm-up (paired latents)",
            disable=not self.show_progress_bar,
        ):
            prompt_batch = next(data_iter)
            teacher_samples = self._teacher_rollout_samples(prompt_batch)
            # Snapshot the teacher packed-latent position ids once (fixed resolution)
            # so the affine-transport converters can unpack (B,seq,C)->(B,C,H,W).
            if self._teacher_latent_ids is None:
                t_ids = getattr(teacher_samples[0], "latent_ids", None)
                if t_ids is not None:
                    self._capture_teacher_layout(t_ids if t_ids.dim() == 3 else t_ids.unsqueeze(0))

            if not use_traj:
                # Legacy: final clean latent only. z0_S from the teacher's final image.
                z0_T = torch.stack([s.all_latents[-1] for s in teacher_samples], dim=0).to(device)
                images = torch.stack([s.image for s in teacher_samples], dim=0).to(device)
                with torch.no_grad():
                    z0_S, _ = self._split_student_encode(self.adapter.encode_pixels(images))
                z_T_list.append(z0_T.float().cpu())
                z_S_list.append(z0_S.float().cpu())
                sigma_list.append(0.0)  # clean latent -> sigma 0 (lowest noise level)
                img_list.append(images.float().cpu())
                continue

            # Trajectory mode: pair every denoising step. all_latents is
            # (num_steps, seq, C); decode each step in teacher space -> image ->
            # student encode_pixels. One (B,...) pair per timestep. The per-step
            # timestep value (the noise level of THAT latent, on the shared [0,1000]
            # flow-matching axis) conditions the adaLN-Zero modulation; map each
            # collected position to its scheduler timestep via latent_index_map.
            num_steps = teacher_samples[0].all_latents.shape[0]
            ts_vals = getattr(teacher_samples[0], "timesteps", None)
            idx_map = getattr(teacher_samples[0], "latent_index_map", None)
            T_len = int(ts_vals.shape[0]) if ts_vals is not None else num_steps
            # Match the teacher VAE dtype for decode (rollout latents may be fp16
            # while the VAE/bias is bf16 -> dtype-mismatch error otherwise).
            vae_dtype = self.teacher_adapter.pipeline.vae.dtype
            for t in range(num_steps):
                z_t_T = torch.stack([s.all_latents[t] for s in teacher_samples], dim=0).to(device)
                t_ids = torch.stack([s.latent_ids for s in teacher_samples], dim=0).to(device)
                with torch.no_grad(), self.autocast():
                    imgs_t = self.teacher_adapter.decode_latents(
                        z_t_T.to(vae_dtype), latent_ids=t_ids, output_type="pt"
                    )
                    if isinstance(imgs_t, list):
                        imgs_t = torch.stack(imgs_t, dim=0)
                    imgs_t = imgs_t.to(device)
                    z_t_S, _ = self._split_student_encode(self.adapter.encode_pixels(imgs_t))
                # noise level of all_latents[t]: timesteps[step_idx] if pre-clean else 0.
                step_idx = int(idx_map[t]) if idx_map is not None else t
                t_val = (
                    float(ts_vals[step_idx]) if (ts_vals is not None and step_idx < T_len) else 0.0
                )
                # Convert the teacher timestep to the scheduler-agnostic sigma in [0,1].
                sigma_val = self._noise_fraction(self.teacher_adapter, t_val)
                z_T_list.append(z_t_T.float().cpu())
                z_S_list.append(z_t_S.float().cpu())
                sigma_list.append(sigma_val)
                # images not retained in trajectory mode (downstream uses only pairs).
        return z_T_list, z_S_list, sigma_list, img_list

    @staticmethod
    def _split_student_encode(enc):
        """Normalize a student ``encode_pixels`` return to a bare latent tensor.

        SD3.5 returns a tensor; FLUX.2 returns ``(packed, ids)``. The student here is
        always BCHW (SD3.5) but stay robust.
        """
        if isinstance(enc, tuple):
            return enc[0], enc[1]
        return enc, None

    def _warmup_comparison_images(
        self, z_T_list, z_S_list, sigma_list=None, max_pairs: Optional[int] = 64
    ):
        """Build a LIST of per-pair 2-row comparison images for warm-up viz.

        ``z_T_list`` / ``z_S_list`` are LISTS of batched tensors (one per rolled-out
        warm-up batch this epoch). Each individual (sample) pair becomes its OWN small
        2-row image (NOT one wide concatenated strip):

        Top    = student-space TARGET latent decoded (``decode_latents(z_S)``).
        Bottom = teacher latent carried through the transport then decoded
                 (``decode_latents(T(z_T))``).

        Returns a list of ``LogImage`` (logged as a wandb gallery under one key);
        ``max_pairs`` caps how many pairs are emitted. A good transport makes the two
        rows of each tile match. Only CLEAN pairs (``sigma`` ~ 0) are emitted: noisy-
        latent pairs decode to garbage (both rows pure noise) and are skipped, so the
        gallery stays interpretable. This affects ONLY the visualization; data
        collection and the transport fit are unchanged (still cover all noise levels).
        """
        from PIL import Image as _PILImage

        from ...logger.formatting import LogImage

        try:
            dev = self.accelerator.device
            if not isinstance(z_T_list, (list, tuple)):
                z_T_list, z_S_list = [z_T_list], [z_S_list]
            if sigma_list is None:
                sigma_list = [None] * len(z_T_list)
            tiles = []
            # Only CLEAN pairs (sigma ~ 0) are visualized: a noisy-latent pair decodes to
            # meaningless garbage (BOTH the target and transported rows are pure noise),
            # so it pollutes the gallery without conveying any transport quality. sigma is
            # None for the legacy clean-only collection -> always kept.
            clean_eps = 1e-3
            with torch.no_grad():
                for bi, (z_T, z_S, s_b) in enumerate(zip(z_T_list, z_S_list, sigma_list)):
                    if s_b is not None and float(s_b) > clean_eps:
                        continue
                    z_S_b = z_S.to(dev)
                    z_T_b = z_T.to(dev)
                    # transport at this pair's noise level (adaln uses sigma; affine ignores)
                    z_Thad = self.transport.transport_sample(z_T_b, sigma=s_b)  # -> student-space
                    top = self.adapter.decode_latents(z_S_b, output_type="pil")
                    bot = self.adapter.decode_latents(z_Thad, output_type="pil")
                    top = top if isinstance(top, (list, tuple)) else [top]
                    bot = bot if isinstance(bot, (list, tuple)) else [bot]
                    for si in range(min(len(top), len(bot))):
                        w, h = top[si].size
                        tile = _PILImage.new("RGB", (w, 2 * h), "white")
                        tile.paste(top[si], (0, 0))
                        tile.paste(bot[si].resize((w, h)), (0, h))
                        tiles.append(
                            LogImage(tile, caption=f"target(top)/transported(bot) b{bi}s{si}")
                        )
                        if max_pairs is not None and len(tiles) >= max_pairs:
                            return tiles
            return tiles or None
        except Exception as e:  # viz must never break training
            logger.warning(f"warm-up comparison images failed: {type(e).__name__}: {e}")
            return None

    def _warmup_transport(self) -> None:
        """Warm-up the transport ONLINE, then freeze (before L1).

        Online schedule (avoids overfitting the transport to a fixed pair set):
        each of ``transport_warmup_epochs`` epochs rolls out FRESH paired latents
        (``transport_warmup_batches`` teacher rollouts across the full denoising
        trajectory), then performs ONE transport update on that fresh batch:
        - closed-form (linear/whitening): accumulate sufficient statistics across
          epochs and re-solve on ALL data seen so far (DPO-style online iteration);
        - learnable (adaln): one Adam step (grad-accumulated over the batch).

        adaln two-phase (``transport_base_warmup_epochs > 0``): the first K epochs
        update ONLY the closed-form base affine, then the base is frozen and the
        remaining epochs train ONLY the adaLN modulation MLP against that stable
        target (avoids chasing a moving base). K=0 keeps the legacy joint update.

        Decoupling: the transport uses its OWN optimizer / pure-stat accumulation,
        never ``accelerator.accumulate``, so it does not interact with the XOPD L1
        gradient-accumulation (GAS) loop. After warm-up the transport is frozen and
        broadcast from rank 0 for cross-rank determinism.
        """
        ta = self.training_args
        device = self.accelerator.device
        n_epochs = max(1, ta.transport_warmup_epochs)
        # Gradient-trained transports (adaln modulation MLP / conv residual nets) are
        # nn.Modules with a dedicated online optimizer; closed-form ones (linear/
        # whitening) are not. The nn.Module path also gets the two-phase schedule and
        # the param+buffer broadcast for cross-rank determinism.
        is_nn = isinstance(self.transport, torch.nn.Module)
        is_adaln = isinstance(self.transport, AdaLNTransport)
        # Two-phase schedule (gradient transports): the first `base_epochs` epochs
        # update ONLY the closed-form base affine; the rest freeze the base and train
        # ONLY the learnable part (adaLN MLP / conv residual) against that stable
        # target. 0 -> legacy joint update.
        base_epochs = int(getattr(ta, "transport_base_warmup_epochs", 0)) if is_nn else 0
        if is_nn:
            self.transport.to(device)
            self.transport.set_online_lr(ta.transport_lr)
            if base_epochs >= n_epochs:
                logger.warning(
                    f"transport_base_warmup_epochs ({base_epochs}) >= "
                    f"transport_warmup_epochs ({n_epochs}): the learnable part will "
                    "not be trained; the transport degrades to the closed-form base "
                    "affine (equivalent to 'linear')."
                )
        logger.info(
            f"Transport ({ta.vae_transport}) ONLINE warm-up: {n_epochs} epoch(s), "
            f"{ta.transport_warmup_batches} freshly-rolled-out batch(es)/epoch "
            f"({'gradient step' if is_nn else 'closed-form on accumulated stats'} "
            "per epoch), then frozen."
            + (
                f" Base/MLP split: first {base_epochs} epoch(s) base-only, then "
                "MLP-only on a frozen base."
                if base_epochs > 0
                else ""
            )
        )
        clean_fit_only = bool(getattr(ta, "transport_clean_fit_only", True))
        for ep in range(n_epochs):
            # 1. Fresh rollout for THIS epoch (full denoising trajectory per batch).
            z_T_list, z_S_list, sigma_list, _img = self._collect_warmup_pairs()
            # Clean-only fit (default): the NOISY-step student target is built by the
            # pixel bridge (decode a NOISY teacher latent -> garbage; the VAE is only
            # valid on clean latents), so fitting on those pairs corrupts the transport.
            # By Prop.(affine pushforward) a linear/conv transport fit on CLEAN pairs
            # already transports EVERY noise level correctly, so we drop the noisy pairs
            # from the fit. (M5/non-linear is also clean-fit here; its noisy behaviour is
            # an accepted approximation.) sigma is None for the legacy clean-only path.
            if clean_fit_only:
                clean_eps = 1e-3
                keep = [i for i, s in enumerate(sigma_list) if s is None or float(s) <= clean_eps]
                if not keep:
                    raise RuntimeError(
                        "transport_clean_fit_only=True but warm-up epoch produced NO "
                        f"clean (sigma<={clean_eps}) pairs out of {len(sigma_list)}; "
                        "set transport_warmup_trajectory=False (clean endpoint only) or "
                        "check the teacher rollout."
                    )
                z_T_list = [z_T_list[i] for i in keep]
                z_S_list = [z_S_list[i] for i in keep]
                sigma_list = [sigma_list[i] for i in keep]
            z_T_dev = [z.to(device) for z in z_T_list]
            z_S_dev = [z.to(device) for z in z_S_list]
            # 2. One transport update on this epoch's fresh data. adaln consumes the
            #    per-pair noise fractions (sigma) to condition its modulation; closed-
            #    form transports ignore the extra kwargs. Two-phase (adaln): base-only
            #    for the first base_epochs, then modulation-only on the frozen base.
            update_kwargs = {"sigma_list": sigma_list}
            if base_epochs > 0:
                update_kwargs["update_base"] = ep < base_epochs
                update_kwargs["update_mod"] = ep >= base_epochs
            if is_nn:
                # Gradient transports: many optimizer steps per epoch on the (expensive)
                # rolled-out pairs, so the learnable part actually converges (1/epoch is
                # far too slow). Closed-form transports ignore this.
                update_kwargs["inner_steps"] = int(getattr(ta, "transport_inner_steps", 1))
            recon = self.transport.update_online(z_T_dev, z_S_dev, **update_kwargs)
            logger.info(f"  transport warm-up epoch {ep}: recon_mse={recon:.6f}")
            # 3. Log loss + per-pair comparison images (each pair = one 2-row tile,
            #    top=student target / bottom=transported teacher) as a gallery on the
            #    warm-up x-axis, EVERY epoch. Capped at max_pairs to bound upload.
            if self.accelerator.is_main_process and self.logger is not None:
                data = {"warmup/transport_recon_mse": float(recon)}
                imgs = self._warmup_comparison_images(
                    z_T_dev, z_S_dev, sigma_list=sigma_list, max_pairs=64
                )
                if imgs:
                    data["warmup/target_vs_transported"] = imgs
                self.log_warmup_data(data, step=ep)
            del z_T_dev, z_S_dev

        # Freeze + broadcast from rank 0 for cross-rank determinism.
        if self.accelerator.num_processes > 1:
            import torch.distributed as dist

            if is_nn:
                # nn.Module transports (adaln MLP / conv residual nets) have BOTH
                # gradient params and frozen buffers (A_base, b_base from the closed-
                # form fit) — broadcast both so all ranks share rank-0's transport.
                for p in self.transport.parameters():
                    dist.broadcast(p.data, src=0)
                for buf in self.transport.buffers():
                    dist.broadcast(buf.data, src=0)
            else:
                A = self.transport.A.to(device)
                b = self.transport.b.to(device)
                dist.broadcast(A, src=0)
                dist.broadcast(b, src=0)
                self.transport.A, self.transport.b = A, b
        if is_nn:
            self.transport.eval()
            for p in self.transport.parameters():
                p.requires_grad_(False)
            self.transport._fitted = True
        logger.info(f"Transport ({ta.vae_transport}) warm-up complete; frozen for L1.")

    # ===================== M8: HSCT cold-start ==============================
    def _coldstart_hsct(self) -> None:
        """Cold-start the HSCT transport: fit linear P + grad-train deepstack Q on
        (raw z_S, raw z_T, student hidden h_S) triples, logging loss + recon to wandb,
        then freeze for L1.

        Data source (ta.hsct_coldstart_source):
          * 'offline_corpus': read images + prompts from a corpus ``index.jsonl`` on
            the shared disk (fast; matches the offline alignment study).
          * 'online_gen': teacher-generate images on the fly at mixed guidance scales
            (on-distribution but slower).
        DDP: a DistributedSampler shards the data; the transport all-reduces P stats and
        Q grads so every rank ends identical (then rank-0 broadcast as insurance).
        """
        import json as _json

        import torchvision.transforms as _TT
        from PIL import Image as _Image
        from PIL import ImageFile as _ImageFile
        from torch.utils.data import DataLoader as _DL
        from torch.utils.data import Dataset as _DS
        from torch.utils.data.distributed import DistributedSampler as _DSamp

        _ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate partial-write corpus PNGs

        ta = self.training_args
        acc = self.accelerator
        device = acc.device
        s_vae = self.adapter.pipeline.vae
        t_vae = self.teacher_adapter.pipeline.vae
        s_scale = float(s_vae.config.scaling_factor)
        s_shift = float(getattr(s_vae.config, "shift_factor", 0.0) or 0.0)
        sigma = float(ta.hsct_coldstart_sigma)

        self.transport.to(device)
        self.transport.set_online_lr(ta.hsct_coldstart_lr)

        # ---- dataset: (image[-1,1], prompt) -----------------------------------
        class _Corpus(_DS):
            def __init__(self, entries, res):
                self.entries = entries
                self.tf = _TT.Compose([_TT.Resize(res), _TT.CenterCrop(res), _TT.ToTensor()])

            def __len__(self):
                return len(self.entries)

            def __getitem__(self, i):
                # skip truncated/corrupt PNGs (partial gen writes) -> next valid entry
                n = len(self.entries)
                for off in range(n):
                    e = self.entries[(i + off) % n]
                    try:
                        img = _Image.open(e["path"]).convert("RGB")
                        return self.tf(img) * 2.0 - 1.0, e["prompt"]
                    except (OSError, ValueError):
                        continue
                raise RuntimeError("HSCT cold-start corpus: no decodable image found")

        if ta.hsct_coldstart_source != "offline_corpus":
            raise NotImplementedError(
                "online_gen cold-start not wired in this build; use "
                "hsct_coldstart_source='offline_corpus' (generate the corpus offline)."
            )
        index_path = os.path.join(ta.hsct_coldstart_corpus, "index.jsonl")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"HSCT cold-start needs {index_path} (lines {{'path','prompt'}}); "
                "build it from the corpus first (scripts/vae_align/build_corpus_index.py)."
            )
        with open(index_path) as f:
            entries = [_json.loads(line) for line in f if line.strip()]
        if ta.hsct_coldstart_max_images > 0:
            entries = entries[: ta.hsct_coldstart_max_images]
        ds = _Corpus(entries, ta.resolution)
        sampler = (
            _DSamp(
                ds,
                num_replicas=acc.num_processes,
                rank=acc.process_index,
                shuffle=True,
                drop_last=True,
            )
            if acc.num_processes > 1
            else None
        )
        dl = _DL(
            ds,
            batch_size=ta.hsct_coldstart_bs,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=8,
            drop_last=True,
            pin_memory=True,
        )
        if acc.is_main_process:
            _noise_desc = (
                f"noisy sigma~U(0,{ta.hsct_coldstart_sigma_max})"
                if ta.hsct_coldstart_noisy
                else f"clean sigma={sigma}"
            )
            logger.info(
                f"HSCT cold-start: {len(ds)} imgs, source={ta.hsct_coldstart_source}, "
                f"arch={ta.hsct_q_arch}/{ta.hsct_q_inject}, blocks={ta.hsct_hidden_blocks}, "
                f"{_noise_desc}, epochs={ta.hsct_coldstart_epochs}, bs={ta.hsct_coldstart_bs}."
            )

        transformer = self.adapter.pipeline.transformer
        # encode_prompt needs the (possibly CPU-offloaded) student text encoders on
        # device; onload for the cold-start, restore after (L1 uses precomputed embeds).
        _te_saved = []
        for _n in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            _te = getattr(self.adapter.pipeline, _n, None)
            if _te is not None and hasattr(_te, "parameters"):
                try:
                    _dev0 = next(_te.parameters()).device
                except StopIteration:
                    continue
                _te_saved.append((_te, _dev0))
                _te.to(device)

        # ---- FIXED viz set (rank0): encode a constant set of images ONCE so the recon
        # gallery shows the SAME images at every viz step (=> can eyeball Q improving over
        # cold-start), decoupled from the training batch size. Half are shown clean-target,
        # half noised-target (spanning sigma) inside `_hsct_recon_images`. ----
        _VIZ_N = 32
        viz_pack = None
        if acc.is_main_process and self.logger is not None:
            v_n = min(_VIZ_N, len(ds))
            with torch.no_grad(), self.autocast():
                v_imgs = torch.stack([ds[i][0] for i in range(v_n)]).to(device)
                v_prompts = [ds[i][1] for i in range(v_n)]
                v_rawS = s_vae.encode(v_imgs.to(s_vae.dtype)).latent_dist.mode().float()
                v_rawT = t_vae.encode(v_imgs.to(t_vae.dtype)).latent_dist.mode().float()
                v_pe, _, v_pool, _ = self.adapter.pipeline.encode_prompt(
                    prompt=v_prompts,
                    prompt_2=v_prompts,
                    prompt_3=v_prompts,
                    device=device,
                    do_classifier_free_guidance=False,
                )
            viz_pack = (v_rawS, v_rawT, v_imgs, v_pe, v_pool)
        cs_step = 0
        for ep in range(ta.hsct_coldstart_epochs):
            if sampler is not None:
                sampler.set_epoch(ep)
            for imgs, prompts in dl:
                imgs = imgs.to(device)
                with torch.no_grad(), self.autocast():
                    raw_S = s_vae.encode(imgs.to(s_vae.dtype)).latent_dist.mode().float()
                    raw_T = t_vae.encode(imgs.to(t_vae.dtype)).latent_dist.mode().float()
                    enc = self.adapter.pipeline.encode_prompt(
                        prompt=list(prompts),
                        prompt_2=list(prompts),
                        prompt_3=list(prompts),
                        device=device,
                        do_classifier_free_guidance=False,
                    )
                    prompt_embeds, _, pooled, _ = enc
                    B = imgs.shape[0]
                    # NOISY-domain cold-start: per-sample sigma; add INDEPENDENT flow-matching
                    # noise to the Q input (noisy z_S) AND the target (noisy z_T); h_S at the
                    # noisy state. Trains Q to map noisy_student -> noisy_teacher at matched
                    # sigma so it does not produce garbage on the L1 rollout's noisy latents.
                    # CRITICAL: noise in the SCALED (flow-matching) latent space -- exactly what
                    # the L1 rollout produces -- then convert back to raw for Q/P (which operate
                    # on raw latents). Noising raw latents directly is WRONG: the teacher raw
                    # latent has large magnitude (FLUX scale~0.36 => std~2.8) so unit noise is
                    # far too weak, and the effective sigma no longer matches L1.
                    if ta.hsct_coldstart_noisy:
                        sig = torch.rand(B, 1, 1, 1, device=device) * ta.hsct_coldstart_sigma_max
                        # FM-noise student (scaled space) + teacher (BN-packed space) to match
                        # the L1 rollout; single source of truth lives on the transport.
                        q_in_S, z_tf = self.transport.noise_student_scaled(raw_S, sig)
                        q_tgt_T = self.transport.noise_teacher_raw(raw_T, sig)
                        ts = (sig.reshape(B) * 1000.0).to(transformer.dtype)
                    else:
                        sig = float(sigma)
                        q_in_S, q_tgt_T = raw_S, raw_T
                        zS = (raw_S - s_shift) * s_scale
                        if sig > 0.0:
                            zS_n = (1.0 - sig) * zS + sig * torch.randn_like(zS)
                            q_in_S = zS_n / s_scale + s_shift
                        else:
                            zS_n = zS
                        z_tf = zS_n  # transformer input = (noisy) SCALED student latent
                        ts = torch.full((B,), sig * 1000.0, device=device, dtype=transformer.dtype)
                    self._hsct_hidden = {}
                    self._hsct_capture = True
                    transformer(
                        hidden_states=z_tf.to(transformer.dtype),
                        timestep=ts,
                        encoder_hidden_states=prompt_embeds.to(transformer.dtype),
                        pooled_projections=pooled.to(transformer.dtype),
                        return_dict=False,
                    )
                    self._hsct_capture = False
                    h_list = self._hsct_h_list(B)
                # P fits the CLEAN linear map (raw_T<-raw_S, exact pushforward); Q learns the
                # (noisy) inverse. So pass clean raw for P, noisy q_in/q_tgt for Q.
                q_mse, p_mse = self.transport.coldstart_step(
                    q_tgt_T,
                    q_in_S,
                    h_list,
                    raw_T_clean=raw_T,
                    raw_S_clean=raw_S,
                    inner_steps=ta.hsct_coldstart_inner_steps,
                    distributed=acc.num_processes > 1,
                )
                cs_step += 1
                if acc.is_main_process and cs_step % 20 == 0:
                    logger.info(
                        f"  HSCT cold-start ep{ep} step{cs_step} q_lat_mse={q_mse:.5f} p_lat_mse={p_mse:.5f}"
                    )
                    if self.logger is not None:
                        # Cold-start metrics on their OWN x-axis "cold-start/step"
                        # (separate from the L1 training "step" axis).
                        data = {
                            "cold-start/hsct_q_lat_mse": q_mse,
                            "cold-start/hsct_p_lat_mse": p_mse,
                        }
                        if cs_step % 200 == 0 and viz_pack is not None:
                            _vS, _vT, _vI, _vpe, _vpool = viz_pack
                            data["cold-start/hsct_inverse_recon"] = self._hsct_recon_images(
                                _vS, _vT, _vI, _vpe, _vpool, max_n=32
                            )
                        self.log_warmup_data(data, step=cs_step, step_key="cold-start/step")

        # restore text encoders to their original device (free GPU for L1)
        for _te, _dev0 in _te_saved:
            _te.to(_dev0)

        # broadcast rank0 (insurance) + freeze
        if acc.num_processes > 1:
            import torch.distributed as dist

            for p in self.transport.parameters():
                dist.broadcast(p.data, src=0)
            for buf in self.transport.buffers():
                dist.broadcast(buf.data, src=0)
        self.transport.eval()
        for p in self.transport.parameters():
            p.requires_grad_(False)
        self.transport._fitted = True
        if acc.is_main_process:
            logger.info("HSCT cold-start complete; transport frozen for L1.")

    def _hsct_recon_images(
        self, raw_S, raw_T, imgs, prompt_embeds, pooled, max_n=16, noisy_sigma=0.6
    ):
        """wandb gallery of per-sample tiles. Each tile stacks 3 rows:
          row0 = original image (constant anchor)
          row1 = teacher-decode(Q(z_S, h_S))   -- the transport reconstruction
          row2 = teacher-decode(z_T target)    -- the reference the transport aims at
        The FIRST half of the tiles are CLEAN (z_S clean, z_T target = clean raw_T); the
        SECOND half are NOISY: z_S is noised at ``noisy_sigma`` (with h_S taken from the
        student transformer at that noise level) AND the z_T target is noised at the SAME
        sigma. I.e. "half clean-latent-as-target, half noised-latent-as-target", so we can
        eyeball how the transport behaves against BOTH clean and noised teacher targets
        (the noisy latents it will actually see during L1). Returns LogImage."""
        from PIL import Image as _Image

        from ...logger.formatting import LogImage

        transformer = self.adapter.pipeline.transformer
        t_vae = self.teacher_adapter.pipeline.vae
        n = min(max_n, imgs.shape[0])
        n_noisy = n // 2
        n_clean = n - n_noisy
        # clean tiles at sigma=0; noisy tiles SPAN (0, noisy_sigma] so the gallery shows the
        # degradation-vs-noise curve (answers "to what noise level does Q still hold up").
        noisy_levels = (
            torch.linspace(noisy_sigma / n_noisy, noisy_sigma, n_noisy, device=raw_S.device)
            if n_noisy > 0
            else torch.empty(0, device=raw_S.device)
        )
        sig = torch.cat([torch.zeros(n_clean, device=raw_S.device), noisy_levels]).view(-1, 1, 1, 1)
        with torch.no_grad(), self.autocast():
            # FM-noise student in scaled space to match the L1 rollout (transport helper).
            q_in, z_tf = self.transport.noise_student_scaled(raw_S[:n], sig)
            ts = (sig.reshape(n) * 1000.0).to(transformer.dtype)
            self._hsct_hidden = {}
            self._hsct_capture = True
            transformer(
                hidden_states=z_tf.to(transformer.dtype),
                timestep=ts,
                encoder_hidden_states=prompt_embeds[:n].to(transformer.dtype),
                pooled_projections=pooled[:n].to(transformer.dtype),
                return_dict=False,
            )
            self._hsct_capture = False
            h_list = self._hsct_h_list(n)
            qd = self.transport.Q.base.weight.dtype
            zq = self.transport.Q(q_in.to(qd), [h.to(qd) for h in h_list])

            # chunked teacher decode: a single decode of all tiles can OOM at large n.
            def _dec(z):
                outs = [
                    t_vae.decode(z[j : j + 4].to(t_vae.dtype)).sample
                    for j in range(0, z.shape[0], 4)
                ]
                return torch.cat(outs, 0)

            x_inv = _dec(zq)
            # z_T TARGET: clean for clean tiles, BN-packed-space noised for noisy tiles ->
            # "half clean-latent-as-target, half noised-latent-as-target".
            tgt_T = self.transport.noise_teacher_raw(raw_T[:n], sig)
            x_gt = _dec(tgt_T)

        def to_pil(t):
            a = (
                ((t.float().clamp(-1, 1) + 1) / 2 * 255)
                .round()
                .byte()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            return _Image.fromarray(a)

        from PIL import ImageDraw as _ImageDraw

        out = []
        for i in range(n):
            rows = [to_pil(imgs[i]), to_pil(x_inv[i]), to_pil(x_gt[i])]
            w, h = rows[0].size
            tile = _Image.new("RGB", (w, 3 * h), "white")
            for r, im in enumerate(rows):
                tile.paste(im.resize((w, h)), (0, r * h))
            is_clean = float(sig[i]) == 0.0
            # burned-in per-row labels: row0 is the (always clean) ORIGINAL image, row1 is
            # Q's reconstruction, row2 is the z_T TARGET (clean or noised). Makes it obvious
            # that the clean top row is the source, not the target.
            tgt_lab = (
                "z_T target: clean" if is_clean else f"z_T target: noised s={float(sig[i]):.2f}"
            )
            row_labels = ["original (source)", "Q-recon", tgt_lab]
            draw = _ImageDraw.Draw(tile)
            for r, lab in enumerate(row_labels):
                y = r * h + 2
                draw.rectangle([0, y, 9 + 6 * len(lab), y + 15], fill=(0, 0, 0))
                draw.text((3, y + 3), lab, fill=(255, 255, 0))
            out.append(
                LogImage(
                    tile,
                    caption=f"s{i} [{'clean' if is_clean else f'noised s={float(sig[i]):.2f}'}]",
                )
            )
        return out

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
                        grad_norm = self._clip_grad_norm_ep_aware(
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
            # I2I (shared VAE): optional; omitted on T2I batches.
            "image_latents": batch.get("image_latents"),
            "image_latent_ids": batch.get("image_latent_ids"),
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

        if not self._is_ode and not self._cross_vae:
            # Evaluation may leave a different-length schedule on the scheduler.
            # Resolve stochastic train steps only after inference configures this
            # rollout's schedule inside the same-architecture adapter.
            trajectory_indices = SCHEDULER_TRAIN_INDICES
        else:
            # Per-batch selective mode: store latents for EVERY candidate step so each
            # micro-batch can later draw its own k steps; else store just this epoch's subset.
            rollout_steps = (
                self._candidate_train_timestep_indices
                if self.training_args.xopd_resample_steps_per_batch
                else self._train_timestep_indices
            )
            trajectory_indices = compute_trajectory_indices(
                train_timestep_indices=rollout_steps,
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
                    # Nothing reads per-step log-probs: they existed for the REINFORCE
                    # trajectory term, which L1 no longer has.
                    "compute_log_prob": False,
                    "trajectory_indices": trajectory_indices,
                    **batch,
                }
                if self._is_popd:
                    # P-OPD needs the behavior policy's exact transition mean and
                    # covariance from this SAME rollout. Callback collection reuses the
                    # already-computed scheduler output; it adds no model forward.
                    sample_kwargs["extra_call_back_kwargs"] = [
                        "next_latents_mean",
                        "std_dev_t",
                        "dt",
                    ]
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

            # Per-batch selective teacher guidance: draw this micro-batch's k steps (fixed
            # count) so the pre-pass and the grad pass use the SAME steps; None => per-epoch
            # default (self._train_timestep_indices) inside both callees.
            batch_steps = (
                self._draw_batch_steps(batch_idx)
                if self.training_args.xopd_resample_steps_per_batch
                else None
            )

            mu_teacher_list = self._precompute_teacher_means(
                batch=batch,
                latents_index_map=latents_index_map,
                num_timesteps=num_timesteps,
                timestep_indices=batch_steps,
            )
            step_indices = batch_steps if batch_steps is not None else self._train_timestep_indices
            popd_cache_list = (
                self._precompute_popd_step_caches(
                    batch=batch,
                    latents_index_map=latents_index_map,
                    mu_teacher_list=mu_teacher_list,
                    timestep_indices=step_indices,
                )
                if self._is_popd
                else None
            )

            loss_info = self._optimize_train_pass(
                batch=batch,
                latents_index_map=latents_index_map,
                num_timesteps=num_timesteps,
                mu_teacher_list=mu_teacher_list,
                popd_cache_list=popd_cache_list,
                loss_info=loss_info,
                timestep_indices=batch_steps,
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
        student_hidden: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Route the L1 teacher transition mean to the same-arch or cross-VAE path.

        ``student_hidden`` (HSCT only) carries the student transformer hidden states
        captured at this state for the hidden-state-conditioned inverse transport.
        """
        if self._cross_vae:
            return self._teacher_next_latents_mean_cross_vae(
                forward_kwargs, teacher_text_cond, student_hidden=student_hidden
            )
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

    def _build_teacher_query(
        self,
        forward_kwargs: Dict[str, Any],
        teacher_text_cond: Dict[str, torch.Tensor],
    ) -> Callable[..., torch.Tensor]:
        """Build the no-grad teacher transition-mean query closure for the current state.

        TIMESTEP/SIGMA ALIGNMENT (verified force-aligned): t/t_next are the STUDENT's
        rollout timesteps. They are passed straight to the teacher, and because we always
        provide `t_next`, `FlowMatchEulerDiscreteSDEScheduler.step` takes the
        `sigma = t/1000`, `sigma_prev = t_next/1000` branch (it does NOT index the teacher's
        own shifted `self.sigmas`). So the teacher steps at EXACTLY the student's sigma, and
        its transformer sees `t/1000` = the student's noise fraction. => no student->teacher
        sigma mismatch in the L1 target.
        """
        t = forward_kwargs.get("t")
        t_next = forward_kwargs.get("t_next")

        def query_teacher_mean(x_T: torch.Tensor, latent_ids=None) -> torch.Tensor:
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
            # FLUX.2 teacher forward needs the packed-latent position ids. Use the
            # ones the transport supplies (pixel bridge returns them from
            # encode_pixels) else the cached fixed-resolution ids (affine path,
            # whose from_spatial_latent ordering matches _prepare_latent_ids).
            ids = latent_ids if latent_ids is not None else self._teacher_latent_ids
            if ids is not None:
                if ids.dim() == 3 and ids.shape[0] == 1 and x_T.dim() == 3 and x_T.shape[0] > 1:
                    ids = ids.expand(x_T.shape[0], -1, -1)
                tkw["latent_ids"] = ids.to(x_T.device)
            tkw = filter_kwargs(self.teacher_adapter.forward, **tkw)
            out = self.teacher_adapter.forward(**tkw)
            if out.next_latents_mean is None:
                raise RuntimeError("Cross-VAE teacher forward did not return `next_latents_mean`.")
            return out.next_latents_mean.detach()

        return query_teacher_mean

    def _teacher_next_latents_mean_cross_vae(
        self,
        forward_kwargs: Dict[str, Any],
        teacher_text_cond: Dict[str, torch.Tensor],
        student_hidden: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Cross-VAE teacher transition mean, mapped into the student LATENT space (P).

        The transport maps ``x_S`` to teacher space (Q), the INDEPENDENT teacher adapter
        steps there, then the teacher mean is mapped back to the student space via the linear
        P (displacement form). Exact & cheap for affine P (Prop. 3). All under no_grad.
        """
        x_S = forward_kwargs["latents"]
        query_teacher_mean = self._build_teacher_query(forward_kwargs, teacher_text_cond)
        # Pass the student's current noise level as the scheduler-agnostic fraction
        # sigma=t/num_train_timesteps so a sigma-conditioned transport (adaln) modulates
        # at the matching level; affine transports ignore the kwarg (still exact, Prop. 3).
        sigma = self._noise_fraction(self.adapter, forward_kwargs.get("t"))
        transport_ctx = {}
        if student_hidden is not None:
            transport_ctx["student_hidden"] = student_hidden
        mu_S = self.transport.transition_mean_to_student(
            x_S, query_teacher_mean, sigma=sigma, **transport_ctx
        )
        return mu_S.detach()

    def _teacher_next_pixels_cross_vae(
        self,
        forward_kwargs: Dict[str, Any],
        teacher_text_cond: Dict[str, torch.Tensor],
        student_hidden: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """PIXEL-space L1 target: the teacher next state decoded by its OWN decoder D_T.

        Skips the P transport entirely -> the loss never pays P's raw-latent floor. Returns
        detached teacher pixels ``(B,3,H,W)`` to compare against ``D_S(mu_student)``.
        """
        x_S = forward_kwargs["latents"]
        query_teacher_mean = self._build_teacher_query(forward_kwargs, teacher_text_cond)
        sigma = self._noise_fraction(self.adapter, forward_kwargs.get("t"))
        transport_ctx = {}
        if student_hidden is not None:
            transport_ctx["student_hidden"] = student_hidden
        raw_muT = self.transport.teacher_next_mean_raw(
            x_S, query_teacher_mean, sigma=sigma, **transport_ctx
        )
        return self._decode_teacher_pixels(raw_muT)

    def _decode_teacher_pixels(self, raw_T: torch.Tensor) -> torch.Tensor:
        """Decode a RAW teacher latent (B,32,64,64) to pixels via the teacher VAE (no grad)."""
        t_vae = self.teacher_adapter.pipeline.vae
        with torch.no_grad():
            return t_vae.decode(raw_T.to(t_vae.dtype)).sample.detach()

    def _decode_student_pixels(self, mu_student_scaled: torch.Tensor) -> torch.Tensor:
        """Decode the student next-mean (SCALED rollout space) to pixels via the student VAE,
        KEEPING grad so the pixel L1 backprops through the frozen D_S into the LoRA."""
        s_vae = self.adapter.pipeline.vae
        raw_S = self.transport._student_scaled_to_raw(mu_student_scaled)
        return s_vae.decode(raw_S.to(s_vae.dtype)).sample

    def _build_teacher_text_cond(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Teacher text conditioning for the L1 teacher query.

        Same-arch (shared VAE/TE): the teacher embeddings were precomputed offline
        and ride on the rollout samples under ``teacher_*`` keys (the FLUX student
        sample class carries them); read them off the stacked batch.

        Cross-VAE: the student is a *different* architecture (e.g. SD3.5) whose
        sample class carries NO ``teacher_*`` fields, so the stacked batch lacks
        them. Instead the independent teacher adapter has its own resident text
        encoder; encode the raw prompt strings on the fly. Constant across
        timesteps, so this runs once per optimize batch.
        """
        if not self._cross_vae:
            cond = {
                "prompt_embeds": batch["teacher_prompt_embeds"],
                "text_ids": batch["teacher_text_ids"],
            }
            if self.teacher_gs > 1.0:
                cond["negative_prompt_embeds"] = batch["teacher_negative_prompt_embeds"]
                cond["negative_text_ids"] = batch["teacher_negative_text_ids"]
            return cond

        # Cross-VAE: encode with the teacher adapter's own text encoder.
        prompts = batch["prompt"]
        if isinstance(prompts, str):
            prompts = [prompts]
        enc = self.teacher_adapter.encode_prompt(
            prompt=list(prompts),
            guidance_scale=self.teacher_gs,
            device=self.accelerator.device,
        )
        cond = {
            "prompt_embeds": enc["prompt_embeds"],
            "text_ids": enc["text_ids"],
        }
        if self.teacher_gs > 1.0 and "negative_prompt_embeds" in enc:
            cond["negative_prompt_embeds"] = enc["negative_prompt_embeds"]
            cond["negative_text_ids"] = enc["negative_text_ids"]
        return cond

    def _l1_step_inputs(
        self,
        batch: Dict[str, Any],
        latents_index_map: torch.Tensor,
        num_timesteps: int,
        timestep_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """``(t, latents, forward_kwargs)`` for one trajectory position of a stored rollout."""
        t = batch["timesteps"][:, timestep_index]
        # Final timestep has no successor -> t_next=0. Must stay BATCHED (shape [B], like t):
        # the I2I ragged fallback in the adapter indexes t_next[idx], which raises on a 0-dim
        # scalar.
        t_next = (
            batch["timesteps"][:, timestep_index + 1]
            if timestep_index + 1 < num_timesteps
            else torch.zeros_like(t)
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
        return t, latents, forward_kwargs

    def _precompute_teacher_means(
        self,
        batch: Dict[str, Any],
        latents_index_map: torch.Tensor,
        num_timesteps: int,
        timestep_indices: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """No-grad pre-pass: the per-timestep teacher mean the gradient pass regresses onto.

        Teacher-only by design. The gradient pass runs its own student forward and computes
        ``D_k`` there, so a student forward here would be duplicate work -- the student weights
        do not move in between (one optimizer step per epoch). The one exception is HSCT, whose
        teacher mean is conditioned on hidden states that can only be captured from a student
        forward, so that path still pays for one.

        ``timestep_indices`` defaults to the L1 training subset
        (``self._train_timestep_indices``); the returned list is aligned to it.
        """
        step_indices = (
            timestep_indices if timestep_indices is not None else self._train_timestep_indices
        )
        mu_teacher_list: List[torch.Tensor] = []

        with torch.no_grad(), self.autocast():
            # Teacher text conditioning. Same-arch: the teacher's own embeddings are
            # precomputed offline and carried on the rollout samples (teacher_*
            # fields). Cross-VAE: the SD3.5 student samples carry NO teacher_* fields
            # (different sample class), but the independent teacher adapter has its
            # OWN resident text encoder, so encode the teacher prompts on the fly
            # from the raw prompt strings (constant across timesteps).
            teacher_text_cond = self._build_teacher_text_cond(batch)
            for timestep_index in tqdm(
                step_indices,
                desc=f"Epoch {self.epoch} Teacher pre-pass",
                position=1,
                leave=False,
                disable=not self.show_progress_bar,
            ):
                _, _, forward_kwargs = self._l1_step_inputs(
                    batch, latents_index_map, num_timesteps, timestep_index
                )

                student_hidden = None
                if self._is_hsct:
                    # The ONLY reason to run the student here: the inverse transport is
                    # conditioned on this forward's transformer hidden states.
                    self._hsct_capture = True
                    self.adapter.forward(**forward_kwargs)
                    self._hsct_capture = False
                    student_hidden = self._hsct_h_list(forward_kwargs["latents"].shape[0])

                if self._pixel_loss:
                    # PIXEL-space target: the teacher next state decoded by D_T (detached).
                    mu_teacher_list.append(
                        self._teacher_next_pixels_cross_vae(
                            forward_kwargs, teacher_text_cond, student_hidden=student_hidden
                        )
                    )
                    continue

                mu_teacher_list.append(
                    self._teacher_mean_dispatch(
                        forward_kwargs, teacher_text_cond, student_hidden=student_hidden
                    )
                )

        return mu_teacher_list

    def _precompute_popd_step_caches(
        self,
        *,
        batch: Dict[str, Any],
        latents_index_map: torch.Tensor,
        mu_teacher_list: List[torch.Tensor],
        timestep_indices: List[int],
    ) -> List[_POPDStepCache]:
        """Build detached P-OPD responsibilities from cached behavior transitions."""
        if len(mu_teacher_list) != len(timestep_indices):
            raise ValueError(
                "P-OPD teacher cache must align one-to-one with training timesteps, "
                f"got len(mu_teacher_list)={len(mu_teacher_list)} and "
                f"timestep_indices={timestep_indices!r}."
            )
        if not isinstance(latents_index_map, torch.Tensor) or latents_index_map.ndim != 1:
            raise ValueError(
                "P-OPD expects a shared 1D latent_index_map, "
                f"got type={type(latents_index_map).__name__}, "
                f"shape={getattr(latents_index_map, 'shape', None)}."
            )
        if "all_latents" not in batch or not isinstance(batch["all_latents"], torch.Tensor):
            raise KeyError(
                "P-OPD requires tensor batch['all_latents'] from the behavior rollout, "
                f"available keys={sorted(batch.keys())!r}."
            )

        caches: List[_POPDStepCache] = []
        for k_idx, timestep_index in enumerate(timestep_indices):
            if timestep_index + 1 >= latents_index_map.shape[0]:
                raise ValueError(
                    f"P-OPD timestep_index={timestep_index} has no successor in "
                    f"latent_index_map of length {latents_index_map.shape[0]}."
                )
            next_compact_index = int(latents_index_map[timestep_index + 1].item())
            if next_compact_index < 0 or next_compact_index >= batch["all_latents"].shape[1]:
                raise ValueError(
                    "P-OPD next-latent index is missing or out of range: "
                    f"timestep_index={timestep_index}, compact_index={next_compact_index}, "
                    f"all_latents.shape={tuple(batch['all_latents'].shape)}."
                )
            next_latents = batch["all_latents"][:, next_compact_index].detach()
            behavior = extract_popd_behavior_transition(
                batch,
                timestep_index=int(timestep_index),
            )
            variance = compute_transition_variance(
                behavior.std_dev_t,
                behavior.dt,
                self.adapter.scheduler.dynamics_type,
            )
            mu_teacher = mu_teacher_list[k_idx].detach()
            responsibility = compute_popd_responsibility(
                next_latents=next_latents,
                mu_old=behavior.mu_old,
                mu_teacher=mu_teacher,
                transition_variance=variance,
                alpha=self.popd_alpha,
                temperature=self.popd_temperature,
            )
            caches.append(
                _POPDStepCache(
                    next_latents=next_latents,
                    mu_old=behavior.mu_old,
                    transition_variance=variance,
                    dt=behavior.dt,
                    responsibility=responsibility,
                )
            )
        return caches

    # Only these two diagnostics are broken out per trained transition. Everything the gate does
    # depends on where in the trajectory the transition sits -- K varies by four decades along the
    # denoising axis against roughly 30% between samples at a fixed step (see
    # docs/xopd/popd_exact_sum_gate_saturation.tex) -- so the joint KL and the gate need a
    # per-step view while the rest are readable in aggregate. Expanding all of them per step is
    # what turned ten diagnostics into several hundred logged series.
    _POPD_PER_STEP_KEYS = ("teacher_old_kl_joint", "gamma")

    @classmethod
    def _append_popd_diagnostics(
        cls,
        loss_info: Dict[str, List[torch.Tensor]],
        diagnostics: Dict[str, torch.Tensor],
        *,
        timestep_index: int,
    ) -> None:
        """Accumulate detached global and per-timestep P-OPD scalar diagnostics."""
        for name, values in diagnostics.items():
            loss_info[f"popd/{name}"].append(values)
            if name in cls._POPD_PER_STEP_KEYS:
                # Per-sample values, so reduce_loss_info reports true min/max/mean/std for this
                # transition rather than a mean of means.
                loss_info[f"popd/{name}/t{timestep_index}"].append(values)

    def _gather_popd_gamma_quantiles(
        self,
        loss_info: Dict[str, List[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Gather per-sample gamma values and compute true global quantiles.

        Quantiles are computed for the pooled gate only. The per-step gate is already summarized
        by its mean and standard deviation, and per-step quantiles multiplied the logged series by
        five for information that the pooled distribution plus the per-step means already carry.
        """
        quantile_metrics: Dict[str, torch.Tensor] = {}
        local = torch.cat(loss_info["popd/gamma"]).detach()
        gathered = self.accelerator.gather(local)
        for quantile_name, value in compute_popd_quantiles(gathered).items():
            quantile_metrics[f"popd/gamma_{quantile_name}"] = value
        return quantile_metrics

    def _validation_d_k_per_timestep(
        self,
        batch: Dict[str, Any],
        latents_index_map: torch.Tensor,
        num_timesteps: int,
        timestep_indices: List[int],
    ) -> List[torch.Tensor]:
        """No-grad per-timestep ``D_k`` over a stored rollout, for the eval validation metric.

        Unlike the training pre-pass this DOES need the student mean, because ``D_k`` itself is
        what the caller reports (:meth:`_evaluate_validation_d_k`).
        """
        if self._pixel_loss:
            raise ValueError(
                "validation D_k is not defined for the pixel-space L1 loss "
                "(xopd_pixel_loss=True): there is no latent-space mu_teacher to compare against."
            )
        d_list: List[torch.Tensor] = []

        with torch.no_grad(), self.autocast():
            teacher_text_cond = self._build_teacher_text_cond(batch)
            for timestep_index in timestep_indices:
                t, latents, forward_kwargs = self._l1_step_inputs(
                    batch, latents_index_map, num_timesteps, timestep_index
                )

                self._hsct_capture = self._is_hsct
                student_out = self.adapter.forward(**forward_kwargs)
                self._hsct_capture = False
                if student_out.next_latents_mean is None:
                    raise RuntimeError(
                        "Student forward did not return `next_latents_mean` during the "
                        f"validation D_k pass; requested return_kwargs={_TEACHER_RETURN_KWARGS!r}."
                    )
                student_hidden = (
                    self._hsct_h_list(forward_kwargs["latents"].shape[0])
                    if self._is_hsct else None
                )

                mu_teacher = self._teacher_mean_dispatch(
                    forward_kwargs, teacher_text_cond, student_hidden=student_hidden
                )
                d_k = compute_per_step_kl(
                    mu_student=student_out.next_latents_mean,
                    mu_teacher=mu_teacher,
                    std_dev_t=student_out.std_dev_t,
                    dt=student_out.dt,
                    normalize=self.normalize_d_k,
                    space=self.xopd_dk_space,
                    latents=latents,
                    sigma=self._noise_fraction(self.adapter, t),
                )
                d_list.append(d_k.detach())

        return d_list

    # ===================== Eval validation D_k =====================
    def _evaluate_validation_d_k(self) -> None:
        """Validation D_k on every guidance_scale==1.0 eval set.

        For each gs=1.0 test set: REUSE the trajectory-storing student rollout
        already produced (and CPU-cached) by the reward-eval pass
        (:meth:`_run_eval_inference_batches`), which uses the SAME settings
        (``eval.num_inference_steps``, that set's ``guidance_scale``, EMA params).
        No second rollout is done here. We then run
        :meth:`_validation_d_k_per_timestep` over ALL rollout steps to get the
        per-timestep Gaussian-transition KL ``D_k``. Per-step means (gathered
        across ranks) are logged as ``eval/{set}/d_k/{ti}`` and the trajectory
        mean as ``eval/{set}/d_k_mean`` into the shared ``_eval_log_sink`` so they
        land on the same wandb step as the student/teacher reward curves.

        This is a validation loss: with kl_beta=0 the training L1 loss equals the
        pathwise ``D_k``, so this mirrors ``train/d_k`` on held-out prompts. Only
        gs=1.0 sets are scored (matches the no-CFG training transition and the
        gs=1.0 cached teacher text embeddings).
        """
        if not self.test_dataloaders:
            return
        device = self.accelerator.device
        pbs = self.training_args.per_device_batch_size

        for test_set_name in sorted(self.test_dataloaders.keys()):
            merged_eval = self._merged_eval_args_for_test_set_name(test_set_name)
            gs = float(getattr(merged_eval, "guidance_scale", 1.0))
            if abs(gs - 1.0) > 1e-6:
                continue  # validation D_k is defined only for the no-CFG (gs=1.0) sets

            eval_steps = int(
                getattr(merged_eval, "num_inference_steps", self.training_args.num_inference_steps)
            )
            step_indices = list(range(eval_steps))
            log_pfx = self._eval_log_prefix(test_set_name)

            # Reuse the trajectory-storing student rollout already produced (and
            # CPU-cached) by the reward-eval pass; no second rollout. Missing cache
            # (e.g. gs!=1.0, or reward eval produced no samples) -> skip.
            cache = getattr(self, "_eval_rollout_cache", None) or {}
            rollout_samples = cache.pop(test_set_name, None)
            if not rollout_samples:
                continue

            with torch.no_grad(), self.autocast(), self._eval_inference_context():
                # Per-timestep D_k over the full (reused) rollout, accumulated per step.
                per_step_local: List[List[torch.Tensor]] = [[] for _ in step_indices]
                num_micro = (len(rollout_samples) + pbs - 1) // pbs
                for j in range(num_micro):
                    start = j * pbs
                    end = min(start + pbs, len(rollout_samples))
                    micro = [rollout_samples[i].to(device) for i in range(start, end)]
                    mbatch = BaseSample.stack(micro)
                    lim = mbatch["latent_index_map"]
                    nt = mbatch["timesteps"].shape[1]
                    d_list = self._validation_d_k_per_timestep(
                        batch=mbatch,
                        latents_index_map=lim,
                        num_timesteps=nt,
                        timestep_indices=step_indices,
                    )
                    for ti, d_k in enumerate(d_list):
                        per_step_local[ti].append(d_k.detach())

            # 3) Gather each step's per-sample D_k across ranks (collective on ALL
            #    ranks; DistributedSampler pads the eval shards to equal length so
            #    the gather aligns, exactly like _gather_eval_rewards).
            step_means: List[float] = []
            for ti in step_indices:
                local = (
                    torch.cat(per_step_local[ti]).to(device)
                    if per_step_local[ti]
                    else torch.zeros(0, device=device)
                )
                gathered = self.accelerator.gather(local)
                step_means.append(float(gathered.float().mean().item()))

            if self.accelerator.is_main_process and self._eval_log_sink is not None:
                for ti, m in zip(step_indices, step_means):
                    self._eval_log_sink[f"{log_pfx}/d_k/{ti}"] = m
                self._eval_log_sink[f"{log_pfx}/d_k_mean"] = float(np.mean(step_means))

            self.accelerator.wait_for_everyone()

    def _optimize_train_pass(
        self,
        batch: Dict[str, Any],
        latents_index_map: torch.Tensor,
        num_timesteps: int,
        mu_teacher_list: List[torch.Tensor],
        popd_cache_list: Optional[List[_POPDStepCache]],
        loss_info: Dict[str, List[torch.Tensor]],
        timestep_indices: Optional[List[int]] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """Gradient main pass: per-timestep student forward + loss + backward.

        ``timestep_indices`` MUST match the list passed to ``_precompute_teacher_means``
        (so ``mu_teacher_list[k_idx]`` aligns with this step); defaults to the per-epoch
        ``self._train_timestep_indices``.
        """
        device = self.accelerator.device
        step_indices = (
            timestep_indices if timestep_indices is not None else self._train_timestep_indices
        )
        if self._is_popd and (popd_cache_list is None or len(popd_cache_list) != len(step_indices)):
            raise ValueError(
                "P-OPD requires one behavior cache per training timestep, "
                f"got cache_count={None if popd_cache_list is None else len(popd_cache_list)} "
                f"and timestep_indices={step_indices!r}."
            )
        if not self._is_popd and popd_cache_list is not None:
            raise ValueError(
                "Direct XOPD must not receive P-OPD behavior caches, "
                f"got cache_count={len(popd_cache_list)}."
            )

        with self.autocast():
            for k_idx, timestep_index in enumerate(
                tqdm(
                    step_indices,
                    desc=f"Epoch {self.epoch} Student (grad)",
                    position=1,
                    leave=False,
                    disable=not self.show_progress_bar,
                )
            ):
                with self.accelerator.accumulate(*self.adapter.trainable_components):
                    t = batch["timesteps"][:, timestep_index]
                    # Final timestep -> t_next=0, kept BATCHED (shape [B], like t):
                    # the I2I ragged fallback indexes t_next[idx] and a 0-dim
                    # scalar raises IndexError.
                    t_next = (
                        batch["timesteps"][:, timestep_index + 1]
                        if timestep_index + 1 < num_timesteps
                        else torch.zeros_like(t)
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
                        return_kwargs=self._student_return_kwargs_for_train(),
                        guidance_scale=self.student_gs,
                    )

                    student_out = self.adapter.forward(**forward_kwargs)
                    if student_out.next_latents_mean is None:
                        raise RuntimeError(
                            "Student forward must return `next_latents_mean` for XOPD; got None."
                        )

                    mu_teacher = mu_teacher_list[k_idx]
                    ti = int(timestep_index)

                    if self._is_popd:
                        if student_out.std_dev_t is None or student_out.dt is None:
                            raise RuntimeError(
                                "P-OPD student forward must return `std_dev_t` and `dt`; "
                                f"got std_dev_t={student_out.std_dev_t!r}, dt={student_out.dt!r}, "
                                f"timestep_index={ti}."
                            )
                        popd_cache = popd_cache_list[k_idx]
                        current_variance = compute_transition_variance(
                            student_out.std_dev_t,
                            student_out.dt,
                            self.adapter.scheduler.dynamics_type,
                        )
                        if current_variance.shape != popd_cache.transition_variance.shape or not (
                            torch.allclose(
                                current_variance.detach(),
                                popd_cache.transition_variance,
                                rtol=1.0e-5,
                                atol=1.0e-7,
                            )
                        ):
                            raise RuntimeError(
                                "P-OPD rollout/current transition covariance mismatch at "
                                f"timestep_index={ti}: rollout_variance="
                                f"{popd_cache.transition_variance.detach().cpu().tolist()}, "
                                f"current_variance={current_variance.detach().cpu().tolist()}."
                            )
                        current_dt = (
                            student_out.dt.detach().float().reshape(student_out.dt.shape[0], -1)
                        )
                        rollout_dt = (
                            popd_cache.dt.detach().float().reshape(popd_cache.dt.shape[0], -1)
                        )
                        if current_dt.shape != rollout_dt.shape or not torch.allclose(
                            current_dt,
                            rollout_dt,
                            rtol=1.0e-5,
                            atol=1.0e-7,
                        ):
                            raise RuntimeError(
                                "P-OPD rollout/current dt mismatch at "
                                f"timestep_index={ti}: rollout_dt={rollout_dt.cpu().tolist()}, "
                                f"current_dt={current_dt.cpu().tolist()}."
                            )
                        ungated_mean_kl = compute_popd_gaussian_mean_kl(
                            student_out.next_latents_mean,
                            mu_teacher,
                            popd_cache.transition_variance,
                        )
                        d_k_grad = (
                            popd_cache.responsibility.teacher_responsibility * ungated_mean_kl
                        )
                        diagnostics = compute_popd_diagnostics(
                            next_latents=popd_cache.next_latents,
                            mu_old=popd_cache.mu_old,
                            mu_teacher=mu_teacher,
                            mu_student=student_out.next_latents_mean,
                            transition_variance=popd_cache.transition_variance,
                            dt=popd_cache.dt,
                            responsibility=popd_cache.responsibility,
                            verbose=self.popd_verbose_diagnostics,
                        )
                        self._append_popd_diagnostics(
                            loss_info,
                            diagnostics,
                            timestep_index=ti,
                        )
                    elif self._pixel_loss:
                        # DECODE-space L1: D_S(mu_student) (grad -> LoRA) vs cached D_T(mu_teacher)
                        # pixels. Both faithful decoders land in the SAME RGB space, so the match
                        # is unbiased with NO base-point/displacement correction (unlike raw
                        # latent), and never pays P's ~0.24 image-irrelevant floor.
                        student_px = self._decode_student_pixels(student_out.next_latents_mean)
                        d_k_grad = (
                            (student_px.float() - mu_teacher.float())
                            .abs()
                            .mean(dim=tuple(range(1, student_px.ndim)))
                        )
                    else:
                        d_k_grad = compute_per_step_kl(
                            mu_student=student_out.next_latents_mean,
                            mu_teacher=mu_teacher,
                            std_dev_t=student_out.std_dev_t,
                            dt=student_out.dt,
                            normalize=self.normalize_d_k,
                            space=self.xopd_dk_space,
                            latents=latents,
                            sigma=self._noise_fraction(self.adapter, t),
                        )

                    pathwise_loss = d_k_grad.mean()
                    loss = self.pathwise_coef * pathwise_loss

                    # MoE load-balancing aux (only when the student is a weight-space MoE
                    # and moe_load_balance_coeff > 0; a no-op otherwise). Read right after
                    # the student forward so the aux is from this timestep's graph.
                    moe_coef = getattr(self.training_args, "moe_load_balance_coeff", 0.0)
                    collect_moe_aux = getattr(self.adapter, "collect_moe_aux_loss", None)
                    if moe_coef > 0 and collect_moe_aux is not None:
                        moe_aux = collect_moe_aux()
                        if moe_aux is not None:
                            loss = loss + moe_coef * moe_aux
                            loss_info["moe_aux"].append(moe_aux.detach())

                    # MoF router regularizers (velocity-MoF only; no-ops when coeff==0 or the student
                    # is not a MoF). z-loss bounds the router logits; weight-sum penalty is the soft
                    # replacement for the hard sum-to-1 (keeps the blended velocity magnitude ~1).
                    z_coef = getattr(self.training_args, "router_z_loss_coeff", 0.0)
                    collect_z = getattr(self.adapter, "collect_router_z_loss", None)
                    if z_coef > 0 and collect_z is not None:
                        z = collect_z()
                        if z is not None:
                            loss = loss + z_coef * z
                            loss_info["router_z"].append(z.detach())
                    wsum_coef = getattr(self.training_args, "mof_weight_sum_penalty_coeff", 0.0)
                    collect_wsum = getattr(self.adapter, "collect_weight_sum_penalty", None)
                    if wsum_coef > 0 and collect_wsum is not None:
                        wsum = collect_wsum()
                        if wsum is not None:
                            loss = loss + wsum_coef * wsum
                            loss_info["w_sum_pen"].append(wsum.detach())

                    if self.enable_kl_loss:
                        kl_div, kl_loss = self._compute_kl_anchor(student_out, forward_kwargs)
                        loss = loss + kl_loss
                        loss_info["kl_div"].append(kl_div.detach())
                        loss_info["kl_loss"].append(kl_loss.detach())

                    loss_info["d_k"].append(pathwise_loss.detach())
                    loss_info["loss"].append(loss.detach())
                    # Per-timestep-index d_k/loss detail (on top of the all-timestep
                    # averages above). Keyed by the trajectory position so the same
                    # physical step maps to a stable series even when xopd_train_steps/
                    # num_xopd_steps subset or re-draw the trained steps; reduce_loss_info
                    # then averages each key over that step's num_batches_per_epoch samples.
                    loss_info[f"d_k/{ti}"].append(pathwise_loss.detach())
                    loss_info[f"loss/{ti}"].append(loss.detach())

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_norm = self._clip_grad_norm_ep_aware(
                            self.adapter.get_trainable_parameters(),
                            self.training_args.max_grad_norm,
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        popd_quantiles = (
                            self._gather_popd_gamma_quantiles(loss_info) if self._is_popd else {}
                        )
                        loss_info = reduce_loss_info(self.accelerator, loss_info)
                        loss_info.update(popd_quantiles)
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
            if self._is_popd:
                transition_variance = compute_transition_variance(
                    student_out.std_dev_t,
                    student_out.dt,
                    self.adapter.scheduler.dynamics_type,
                )
                kl_div = compute_popd_gaussian_mean_kl(
                    student_out.next_latents_mean,
                    ref_out.next_latents_mean,
                    transition_variance,
                ).mean()
            else:
                kl_div = compute_per_step_kl(
                    mu_student=student_out.next_latents_mean,
                    mu_teacher=ref_out.next_latents_mean,
                    std_dev_t=student_out.std_dev_t,
                    dt=student_out.dt,
                    normalize=self.normalize_d_k,
                ).mean()

        kl_loss = self.training_args.kl_beta * kl_div
        return kl_div, kl_loss
