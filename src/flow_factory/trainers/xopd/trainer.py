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
from .transport import AdaLNTransport, ConvTransport, build_transport

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
        # Cross-VAE FLUX teacher packed-latent layout (position ids + spatial size),
        # captured at warm-up and injected into the affine-transport converters.
        self._teacher_latent_ids = None
        self._teacher_spatial_hw = None
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
        # case the validation below trips with a clear message. With
        # xopd_train_steps/num_xopd_steps the per-epoch subset is random but its SIZE
        # is FIXED, so use get_num_train_timesteps (the canonical fixed count) rather
        # than len() of one (possibly random) draw, keeping GAS consistent across epochs.
        self.num_train_timesteps = ta.get_num_train_timesteps(self.config)
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
        elif ta.vae_transport in ("conv", "conv_linear"):
            # STRICTLY-LINEAR conv transport: a learned (PixelShuffle) upsample + conv
            # residual on top of the frozen closed-form base affine (do-no-harm), plus
            # a paired linear inverse net. Adds a spatial receptive field the per-pixel
            # affine/adaln lack, while keeping the L1 pushforward exact (still linear).
            # Trained ONLY during warm-up (forward+inverse+cycle recon), then frozen.
            self.transport = build_transport(
                "conv",
                teacher_to_spatial=self._teacher_to_spatial,
                teacher_from_spatial=self._teacher_from_spatial,
                student_to_spatial=self._student_to_spatial,
                student_from_spatial=self._student_from_spatial,
                teacher_channels=self._adapter_latent_channels(self.teacher_adapter),
                student_channels=self._adapter_latent_channels(self.adapter),
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
        if (
            self._cross_vae
            and self.transport is not None
            and self.accelerator.is_main_process
        ):
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
        return self._adapter_to_spatial(self.teacher_adapter, z, **ctx)

    def _teacher_from_spatial(self, z, **ctx):
        return self._adapter_from_spatial(self.teacher_adapter, z, **ctx)

    def _student_to_spatial(self, z, **ctx):
        return self._adapter_to_spatial(self.adapter, z, **ctx)

    def _student_from_spatial(self, z, **ctx):
        return self._adapter_from_spatial(self.adapter, z, **ctx)

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

                snap = snapshot_download(
                    ta.teacher_model_name_or_path, local_files_only=True
                )
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
            sel = torch.randperm(len(pool), generator=g)[: ta.num_xopd_steps]
            sel = sorted(sel.tolist())  # ascending trajectory order
            self._xopd_tts_cache = [pool[i] for i in sel]
            self._xopd_tts_cache_epoch = epoch
        return self._xopd_tts_cache


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
            self._warmup_transport()

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

        # Cross-VAE: the teacher is an independent adapter (its own VAE/TE), so it
        # rolls out directly from prompts in ITS OWN latent space — no whole-module
        # swap (use_teacher_transformer is klein-only) and no cached teacher_* on the
        # heterogeneous student batch. Same-arch: swap the teacher transformer into
        # the student pipeline and route cached teacher embeddings.
        if self._cross_vae:
            with torch.no_grad(), self.autocast():
                all_samples = self._run_teacher_eval_inference_batches_cross_vae(
                    test_set_name, merged_eval, eval_seed
                )
                gathered_rewards = self._gather_eval_rewards()
                gathered_tags = self._gather_eval_tags(all_samples)
                if self.accelerator.is_main_process:
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
            self.accelerator.wait_for_everyone()
            return

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
            }
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
                    self._capture_teacher_layout(
                        t_ids if t_ids.dim() == 3 else t_ids.unsqueeze(0)
                    )

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
                t_ids = torch.stack(
                    [s.latent_ids for s in teacher_samples], dim=0
                ).to(device)
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
                    float(ts_vals[step_idx])
                    if (ts_vals is not None and step_idx < T_len)
                    else 0.0
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
        rows of each tile match.
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
            with torch.no_grad():
                for bi, (z_T, z_S, s_b) in enumerate(zip(z_T_list, z_S_list, sigma_list)):
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
        for ep in range(n_epochs):
            # 1. Fresh rollout for THIS epoch (full denoising trajectory per batch).
            z_T_list, z_S_list, sigma_list, _img = self._collect_warmup_pairs()
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
                raise RuntimeError(
                    "Cross-VAE teacher forward did not return `next_latents_mean`."
                )
            return out.next_latents_mean.detach()

        # Pass the student's current noise level as the scheduler-agnostic fraction
        # sigma=t/num_train_timesteps so a sigma-conditioned transport (adaln) modulates
        # at the matching level (consistent with warm-up); affine transports ignore the
        # kwarg. For any fixed sigma the adaln map is still affine, so this transition-
        # mean pushforward stays exact (Prop. 3).
        sigma = self._noise_fraction(self.adapter, t)
        mu_S = self.transport.transition_mean_to_student(x_S, query_teacher_mean, sigma=sigma)
        return mu_S.detach()

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
            # Teacher text conditioning. Same-arch: the teacher's own embeddings are
            # precomputed offline and carried on the rollout samples (teacher_*
            # fields). Cross-VAE: the SD3.5 student samples carry NO teacher_* fields
            # (different sample class), but the independent teacher adapter has its
            # OWN resident text encoder, so encode the teacher prompts on the fly
            # from the raw prompt strings (constant across timesteps).
            teacher_text_cond = self._build_teacher_text_cond(batch)
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
