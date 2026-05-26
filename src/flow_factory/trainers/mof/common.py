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

# src/flow_factory/trainers/mof/common.py
"""
MoF (Mixture-of-Flow) Trainer — Shared Infrastructure.

Contains MoFMixingModule (learnable mixing weights) and MoFTrainerBase
(all shared infrastructure: teacher loading, lambda weights, source routing,
velocity combination, reward/advantage pipeline, evaluation, checkpointing).

Algorithm-specific subclasses (MoFNFTTrainer, MoFGRPOTrainer) are in
separate files within this package.
"""
import os
from typing import List, Dict, Any, Optional, Set
from functools import partial
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ..abc import BaseTrainer
from ...hparams import MoFBaseTrainingArguments
from ...samples import BaseSample
from ...rewards import RewardBuffer
from ...ema import EMAModuleWrapper
from ...utils.base import (
    filter_kwargs,
    create_generator,
    create_generator_by_prompt,
    to_broadcast_tensor,
    stitch_batch_metadata,
)
from ...utils.logger_utils import setup_logger
from ...utils.noise_schedule import TimeSampler, flow_match_sigma
from ...utils.dist import reduce_loss_info
from ..opd.common import load_teachers, cache_forward_signature, filter_forward_kwargs
from ..ensemble_eval.common import (
    cache_scheduler_step_signature,
    _build_scheduler_step_kwargs,
)

logger = setup_logger(__name__)


class MoFMixingModule(nn.Module):
    """Learnable per-timestep, per-set teacher mixing weights.

    Encapsulates the (K, T, S) logits tensor and softmax normalization.
    Being a proper nn.Module, it is compatible with DeepSpeed ZeRO
    (params discoverable via named_parameters()).

    Args:
        K: Number of teachers.
        T: Number of denoising timesteps.
        S: Number of prompt sets.
        temperature: Softmax temperature (lower = sharper selection).
        init_mode: One of 'zeros', 'random', 'teacher_biased'.
        init_bias: Bias strength for 'teacher_biased' init.
        teacher_set_mapping: Dict mapping set_id → teacher_index for biased init.
    """

    def __init__(
        self,
        K: int,
        T: int,
        S: int,
        temperature: float = 1.0,
        normalize_weights: bool = True,
        init_mode: str = "teacher_biased",
        init_bias: float = 2.0,
        teacher_set_mapping: Optional[Dict[int, int]] = None,
    ):
        super().__init__()
        self.K = K
        self.T = T
        self.S = S
        self.temperature = temperature
        self.normalize_weights = normalize_weights

        # Initialize logits
        if normalize_weights:
            # Softmax mode: logits are in log-space, softmax produces weights
            if init_mode == "uniform":
                data = torch.zeros(K, T, S)
            elif init_mode == "random":
                data = torch.randn(K, T, S) * 0.01
            elif init_mode == "teacher_biased":
                data = torch.zeros(K, T, S)
                if teacher_set_mapping:
                    for s_id, k_idx in teacher_set_mapping.items():
                        data[k_idx, :, s_id] = init_bias
            else:
                raise ValueError(f"Invalid logits_init: {init_mode!r}")
        else:
            # Unnormalized mode: logits ARE the weights directly.
            # Init to produce same effective weights as softmax mode would.
            if init_mode == "uniform":
                # Uniform: each teacher gets 1/K
                data = torch.full((K, T, S), 1.0 / K)
            elif init_mode == "random":
                data = torch.full((K, T, S), 1.0 / K) + torch.randn(K, T, S) * 0.01
            elif init_mode == "teacher_biased":
                # Compute softmax([bias, 0, ...]) to get the target weights
                logit_init = torch.zeros(K)
                if teacher_set_mapping:
                    # Use first mapping to determine bias structure
                    for s_id, k_idx in teacher_set_mapping.items():
                        logit_init[k_idx] = init_bias
                target_weights = F.softmax(logit_init / temperature, dim=0)
                data = target_weights.view(K, 1, 1).expand(K, T, S).clone()
                # For per-set biased init: adjust per set
                if teacher_set_mapping:
                    data.fill_(0.0)
                    for s_id, k_idx in teacher_set_mapping.items():
                        logit_s = torch.zeros(K)
                        logit_s[k_idx] = init_bias
                        w_s = F.softmax(logit_s / temperature, dim=0)
                        data[:, :, s_id] = w_s.unsqueeze(1).expand(K, T)
            else:
                raise ValueError(f"Invalid logits_init: {init_mode!r}")

        self.logits = nn.Parameter(data)

    def forward(self) -> torch.Tensor:
        """Compute mixing weights from logits.

        Returns:
            Tensor of shape (K, T, S).
            If normalize_weights=True: softmax over K (dim=0).
            If normalize_weights=False: raw logits (unnormalized).
        """
        if self.normalize_weights:
            return F.softmax(self.logits / self.temperature, dim=0)
        else:
            return self.logits

    def get_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute mixing weights from an arbitrary logits tensor.

        Used by the trainer to compute weights from EMA-swapped logits
        (which share the same Parameter but may hold different values
        during use_ema_parameters context).

        Args:
            logits: Tensor of shape (K, T, S) — may be self.logits or EMA shadow.

        Returns:
            Tensor of shape (K, T, S) with the same normalization as forward().
        """
        if self.normalize_weights:
            return F.softmax(logits / self.temperature, dim=0)
        else:
            return logits

    def get_weights_for_set(self, set_id: int) -> torch.Tensor:
        """Get (K, T) weights for a specific prompt set."""
        return self.forward()[:, :, set_id]


class MoFTrainerBase(BaseTrainer):
    """
    MoF (Mixture-of-Flow) Trainer Base Class.

    Shared infrastructure for all MoF optimization variants (NFT, GRPO).
    Handles teacher loading, lambda weight management, source routing,
    velocity combination, reward/advantage pipeline, evaluation, and
    checkpointing.

    Subclasses must override: sample() and optimize().
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Note: _initialization() (called by super) already created:
        #   self._teacher_names, self.K, self.S, self._lambda_logits, self.optimizer
        self.training_args: MoFBaseTrainingArguments

        # ---- Unpack config shortcuts ----
        self.off_policy = self.training_args.off_policy
        self.time_sampling_strategy = self.training_args.time_sampling_strategy
        self.time_shift = self.training_args.time_shift
        self.timestep_range = self.training_args.timestep_range
        self.num_train_timesteps = self.training_args.num_train_timesteps

        # ---- EMA for logits (off-policy old policy, aligned with NFT schedule) ----
        ema_device = (
            self.accelerator.device
            if self.training_args.ema_device == "cuda"
            else torch.device("cpu")
        )
        self._logits_ema = EMAModuleWrapper(
            parameters=[self._lambda_logits],
            decay=self.training_args.ema_decay,
            update_step_interval=self.training_args.ema_update_interval,
            device=ema_device,
            decay_schedule=self.training_args.ema_decay_schedule,
            # Pass schedule-specific params (flat_steps, ramp_rate, etc.) from training_args
            **self.training_args
        )

        # ---- Cache forward signature for cheap kwarg filtering ----
        self._forward_param_names, self._forward_accepts_var_kwargs = (
            cache_forward_signature(self.adapter.forward)
        )

        # ---- Cache scheduler step signature ----
        self._sched_cache = cache_scheduler_step_signature(self.adapter.scheduler.step)

        # ---- Reward normalization running stats ----
        self._reward_running_mean: Dict[str, float] = {}  # Reserved for future use
        self._reward_running_var: Dict[str, float] = {}   # Reserved for future use

    # =========================================================================
    # Initialization (called by super().__init__ → _initialization)
    # =========================================================================

    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Override: return optimizer for mixing module only.

        NOTE: We do NOT freeze adapter LoRA params here. They remain
        requires_grad=True so that use_named_parameters() (which filters
        by requires_grad) can correctly swap teacher weights during inference.
        Since teacher velocities are always .detach()'ed, no gradient will
        flow to adapter params regardless.

        Weight decay policy:
        - normalize_weights=True (softmax): weight_decay forced to 0 because
          softmax already bounds outputs to [0,1]. L2 penalty on logits would
          push toward uniform mixing, counteracting teacher-biased init.
        - normalize_weights=False (unnormalized): weight_decay applied as
          configured to prevent unbounded weight drift.
        """
        weight_decay = self.training_args.adam_weight_decay
        if self.training_args.normalize_weights:
            if weight_decay > 0:
                logger.info(
                    f"normalize_weights=True: overriding adam_weight_decay={weight_decay} → 0.0 "
                    f"(softmax bounds output, weight decay would push logits toward uniform)."
                )
            weight_decay = 0.0

        self.optimizer = torch.optim.AdamW(
            self._mixing_module.parameters(),
            lr=self.training_args.learning_rate,
            betas=self.training_args.adam_betas,
            weight_decay=weight_decay,
            eps=self.training_args.adam_epsilon,
        )
        return self.optimizer

    def _initialization(self):
        """Override base _initialization to create logits BEFORE accelerator.prepare().

        Order:
        1. Sync frozen components (if FSDP)
        2. Init dataloaders
        3. Load teachers + create logits (MoF-specific, needs adapter ready)
        4. Create optimizer with logits params
        5. accelerator.prepare(modules + optimizer + dataloaders)
        6. Load inference components + reward model
        """
        if self.adapter._is_fsdp_cpu_efficient_loading():
            logger.info("FSDP CPU Efficient Loading detected. Synchronizing frozen components...")
            self._synchronize_frozen_components()

        # ---- Step 1: Init dataloaders ----
        self.dataloader, self.train_dataloaders_by_source, self.test_dataloaders = self._init_dataloader()

        # ---- Step 2: Load teachers + create logits ----
        # Extract custom names from TeacherConfig (None entries use default)
        teacher_names_from_config = None
        if self.training_args.teachers is not None:
            teacher_names_from_config = [
                tc.name if tc.name is not None else f"opd_teacher_{i}"
                for i, tc in enumerate(self.training_args.teachers)
            ]

        self._teacher_names: List[str] = load_teachers(
            self.adapter,
            list(self.training_args.teacher_paths),
            self.training_args.teacher_param_device,
            teacher_names=teacher_names_from_config,
        )
        self.K = len(self._teacher_names)

        self._teacher_sources: List[Optional[Set[str]]] = []
        self._source_to_set_id: Dict[str, int] = {}
        self._set_id_to_source: Dict[int, str] = {}
        self._init_source_routing()
        self.S = len(self._source_to_set_id)

        logger.info(
            f"MoF: {self.K} teacher(s), T={self.training_args.num_inference_steps} timesteps, "
            f"S={self.S} prompt set(s), "
            f"total learnable params = {self.K * self.training_args.num_inference_steps * self.S}"
        )
        logger.info(f"MoF source mapping: {self._source_to_set_id}")

        self._lambda_logits = self._init_lambda_logits()

        # ---- Step 3: Create optimizer (with mixing module params) ----
        self.optimizer = self._init_optimizer()

        # Keep a direct reference to logits BEFORE prepare() wraps the module.
        # After DDP wrapping, self._mixing_module becomes DistributedDataParallel
        # and the original module is at self._mixing_module.module. But
        # _lambda_logits is the same tensor object regardless of wrapping.
        self._lambda_logits = self._mixing_module.logits
        # Store unwrapped module for direct method access (avoids DDP/DeepSpeed wrapper checks)
        self._mixing_module_unwrapped = self._mixing_module

        # ---- Step 4: accelerator.prepare ----
        # Adapter modules are frozen (inference only) — prepare them for device placement.
        # _mixing_module + optimizer are prepared for DDP gradient sync.
        trainable_module_names = list(self.adapter.target_module_map.keys())
        trainable_modules = [
            getattr(self.adapter, name)
            for name in trainable_module_names
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None
        ]

        to_prepare = trainable_modules + [self._mixing_module, self.optimizer]
        sorted_test_names = sorted(self.test_dataloaders.keys())
        for n in sorted_test_names:
            to_prepare.append(self.test_dataloaders[n])

        prepared = self.accelerator.prepare(*to_prepare)

        # Reassign prepared adapter modules
        for i, name in enumerate(trainable_module_names):
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None:
                self.adapter.set_component(name, prepared[i])

        # Reassign prepared mixing module (now DDP-wrapped) + optimizer
        n_tm = len(trainable_modules)
        self._mixing_module = prepared[n_tm]
        self.optimizer = prepared[n_tm + 1]
        # NOTE: self._lambda_logits still points to the same Parameter tensor
        # (DDP wraps the module but doesn't copy params). Verified by:
        # assert self._lambda_logits.data_ptr() == self._mixing_module.module.logits.data_ptr()

        # Reassign prepared test dataloaders
        for i, tn in enumerate(sorted_test_names):
            self.test_dataloaders[tn] = prepared[n_tm + 2 + i]

        # ---- Step 5: Load inference modules + reward model ----
        self._load_inference_components(trainable_module_names)
        self._init_reward_model()

    # =========================================================================
    # Initialization Helpers
    # =========================================================================

    def _init_source_routing(self):
        """Initialize teacher-to-source mapping from TeacherConfig (OPD pattern).

        Populates:
          - self._teacher_sources: per-teacher set of allowed sources
          - self._source_to_set_id: source name → integer set index
          - self._set_id_to_source: inverse mapping
          - self._source_to_reward_name: source → in-domain reward name (from TeacherConfig.reward_name)
        """
        self._source_to_reward_name: Dict[str, str] = {}

        if (
            self.training_args.teachers is not None
            and self.training_args.teacher_route_by_source
        ):
            all_sources: Set[str] = set()
            for tc in self.training_args.teachers:
                self._teacher_sources.append(
                    set(tc.sources) if tc.sources else None
                )
                if tc.sources:
                    all_sources.update(tc.sources)
                    # Map each source to its in-domain reward name
                    if tc.reward_name:
                        for src in tc.sources:
                            self._source_to_reward_name[src] = tc.reward_name
            # Assign set IDs in sorted order for determinism
            for idx, src in enumerate(sorted(all_sources)):
                self._source_to_set_id[src] = idx
                self._set_id_to_source[idx] = src
        else:
            # Legacy / single-set mode: all teachers broadcast
            self._teacher_sources = [None] * self.K
            self._source_to_set_id = {"default": 0}
            self._set_id_to_source = {0: "default"}

        if self._source_to_reward_name:
            logger.info(f"MoF source→reward mapping: {self._source_to_reward_name}")

    def _init_lambda_logits(self) -> nn.Parameter:
        """Create MoFMixingModule and return its logits parameter.

        The module is stored as self._mixing_module and included in
        accelerator.prepare() for DeepSpeed ZeRO compatibility.
        """
        # Build teacher→set mapping for biased init
        teacher_set_mapping: Dict[int, int] = {}
        for src, s_id in self._source_to_set_id.items():
            for k, teacher_srcs in enumerate(self._teacher_sources):
                if teacher_srcs is not None and src in teacher_srcs:
                    teacher_set_mapping[s_id] = k
                    break

        self._mixing_module = MoFMixingModule(
            K=self.K,
            T=self.training_args.num_inference_steps,
            S=self.S,
            temperature=self.training_args.temperature,
            normalize_weights=self.training_args.normalize_weights,
            init_mode=self.training_args.logits_init,
            init_bias=self.training_args.logits_init_bias,
            teacher_set_mapping=teacher_set_mapping,
        ).to(self.accelerator.device)

        return self._mixing_module.logits

    # =========================================================================
    # Lambda Weights
    # =========================================================================

    def _get_lambda_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute mixing weights — delegates to MoFMixingModule.get_weights().

        Uses the unwrapped module reference (stored before accelerator.prepare)
        to avoid DDP/DeepSpeed wrapper differences.
        """
        return self._mixing_module_unwrapped.get_weights(logits)

    @property
    def enable_kl_loss(self) -> bool:
        """Check if KL penalty is enabled."""
        return self.training_args.kl_beta > 0.0

    # =========================================================================
    # Source / Set ID Utilities
    # =========================================================================

    def _get_sample_set_ids(self, samples: List[BaseSample]) -> torch.Tensor:
        """Extract per-sample set IDs from __source__ metadata.

        Args:
            samples: List of BaseSample with extra_kwargs["__source__"].

        Returns:
            Tensor of shape (B,) with integer set IDs on accelerator device.
        """
        ids = []
        for s in samples:
            source = s.extra_kwargs.get("__source__", "default")
            set_id = self._source_to_set_id.get(source, None)
            if set_id is None:
                logger.warning_once(
                    f"MoF: unknown __source__={source!r}, falling back to set 0. "
                    f"Known sources: {list(self._source_to_set_id.keys())}"
                )
                set_id = 0
            ids.append(set_id)
        return torch.tensor(ids, dtype=torch.long, device=self.accelerator.device)

    def _get_batch_set_id(self, batch: Dict[str, Any]) -> int:
        """Get the set ID for a homogeneous batch (all samples same source).

        Falls back to 0 with warning if __source__ not recognized.
        """
        source = batch.get("__source__", "default")
        set_id = self._source_to_set_id.get(source, None)
        if set_id is None:
            logger.warning_once(
                f"MoF: unknown __source__={source!r} in batch, falling back to set 0."
            )
            return 0
        return set_id

    # =========================================================================
    # Per-Sample Velocity Combination
    # =========================================================================

    def _combine_velocities_per_sample(
        self,
        teacher_velocities: torch.Tensor,
        t_idx: int,
        weights: torch.Tensor,
        set_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Combine teacher velocities using per-sample set-specific weights.

        Args:
            teacher_velocities: (K, B, *latent_dims) — detached teacher preds.
            t_idx: Index into T dimension of weights.
            weights: (K, T, S) precomputed softmax weights.
            set_ids: (B,) integer set IDs for each sample.

        Returns:
            Combined velocity of shape (B, *latent_dims). Gradient flows
            through weights (and thus logits) only.
        """
        # weights[:, t_idx, :] → (K, S)
        w_at_t = weights[:, t_idx, :]  # (K, S)
        # Gather per-sample weights: (K, S)[:, set_ids] → (K, B)
        w_per_sample = w_at_t[:, set_ids]  # (K, B) via advanced indexing

        # Expand for broadcast: (K, B) → (K, B, 1, 1, ...) to match teacher_velocities
        n_spatial = teacher_velocities.ndim - 2  # number of spatial dims (C, H, W)
        w_expanded = w_per_sample.view(self.K, -1, *([1] * n_spatial))

        # Weighted sum over K dimension
        combined = (w_expanded * teacher_velocities).sum(dim=0)  # (B, C, H, W)
        return combined

    # =========================================================================
    # Context Managers
    # =========================================================================

    @contextmanager
    def sampling_context(self):
        """Use EMA logits for sampling when off-policy."""
        if self.off_policy:
            with self._logits_ema.use_ema_parameters([self._lambda_logits]):
                yield
        else:
            yield

    @contextmanager
    def _bypass_ddp_for_weight_swap(self):
        """Temporarily replace DDP-wrapped transformer with unwrapped module.

        DDP's internal parameter buffers (used for gradient bucketing and
        find_unused_parameters tracking) are NOT updated by .data.copy_()
        which use_named_parameters relies on. In inference mode (no_grad),
        calling the DDP-wrapped module still reads from these stale buffers,
        causing all teacher forwards to produce identical outputs.

        This context manager temporarily points the adapter's transformer
        component to the raw unwrapped module, bypassing DDP's forward path.
        Safe during inference since DDP's gradient sync is not needed.
        """
        unwrapped = self.adapter.get_component_unwrapped('transformer')
        wrapped = self.adapter.get_component('transformer')
        if unwrapped is not wrapped:
            self.adapter.set_component('transformer', unwrapped)
        try:
            yield
        finally:
            if unwrapped is not wrapped:
                self.adapter.set_component('transformer', wrapped)

    @contextmanager
    def _mof_inference_context(self, set_id: int = 0):
        """Patch adapter.forward to return lambda-combined teacher velocity.

        Args:
            set_id: Which prompt set's weights to use for this inference pass.

        During sampling (rollout), replaces adapter.forward so each denoising
        step:
        1. Runs each teacher forward → K noise_pred tensors
        2. Combines them with current lambda weights for set_id
        3. Passes combined to scheduler.step
        """
        original_forward = self.adapter.forward
        step_counter = [0]

        # Precompute weights (detached since sampling under no_grad)
        with torch.no_grad():
            weights = self._get_lambda_weights(self._lambda_logits)  # (K, T, S)
            # Extract (K, T) slice for this set
            set_weights = weights[:, :, set_id]  # (K, T)

        def patched_forward(**kwargs):
            t_idx = min(step_counter[0], self.num_train_timesteps - 1)
            step_counter[0] += 1

            # Get noise_pred from each teacher
            noise_only_kwargs = dict(kwargs)
            noise_only_kwargs["return_kwargs"] = ["noise_pred"]

            velocities = []
            for name in self._teacher_names:
                with self.adapter.use_named_parameters(name):
                    out = original_forward(**noise_only_kwargs)
                velocities.append(out.noise_pred)

            # Combine using lambda weights for this set at this timestep
            stacked = torch.stack(velocities, dim=0)  # (K, B, ...)
            w_i = set_weights[:, t_idx]  # (K,)
            expand_shape = (self.K,) + (1,) * (stacked.ndim - 1)
            combined_noise_pred = (w_i.view(*expand_shape) * stacked).sum(dim=0)

            # Run scheduler step
            scheduler_kwargs = _build_scheduler_step_kwargs(
                kwargs, combined_noise_pred, self._sched_cache
            )
            sched_output = self.adapter.scheduler.step(**scheduler_kwargs)

            return sched_output

        self.adapter.forward = patched_forward  # type: ignore[method-assign]
        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        with self._bypass_ddp_for_weight_swap():
            try:
                step_counter[0] = 0
                yield
            finally:
                torch.set_autocast_cache_enabled(prev_cache)
                self.adapter.forward = original_forward  # type: ignore[method-assign]

    @contextmanager
    def _single_teacher_inference_context(self, teacher_idx: int):
        """Patch adapter.forward to use a single teacher's velocity (one-hot weights).

        Used for baseline evaluation: measures each teacher's standalone performance
        on its applicable datasets before training begins.
        """
        original_forward = self.adapter.forward
        step_counter = [0]
        teacher_name = self._teacher_names[teacher_idx]

        def patched_forward(**kwargs):
            step_counter[0] += 1
            noise_only_kwargs = dict(kwargs)
            noise_only_kwargs["return_kwargs"] = ["noise_pred"]

            with self.adapter.use_named_parameters(teacher_name):
                out = original_forward(**noise_only_kwargs)

            scheduler_kwargs = _build_scheduler_step_kwargs(
                kwargs, out.noise_pred, self._sched_cache
            )
            return self.adapter.scheduler.step(**scheduler_kwargs)

        self.adapter.forward = patched_forward  # type: ignore[method-assign]
        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        with self._bypass_ddp_for_weight_swap():
            try:
                step_counter[0] = 0
                yield
            finally:
                torch.set_autocast_cache_enabled(prev_cache)
                self.adapter.forward = original_forward  # type: ignore[method-assign]

    def _compute_teacher_velocities(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Forward each teacher and return stacked velocities (all detached).

        Disables autocast weight cache and bypasses DDP during the loop
        (see CLAUDE.md invariant and _bypass_ddp_for_weight_swap docstring).

        Returns:
            Tensor of shape (K, B, *latent_dims) with all teacher predictions.
        """
        forward_kwargs = self._build_forward_kwargs(batch, timestep, noised_latents)
        velocities = []

        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        with self._bypass_ddp_for_weight_swap():
            try:
                for name in self._teacher_names:
                    with self.adapter.use_named_parameters(name):
                        output = self.adapter.forward(**forward_kwargs)
                    velocities.append(output.noise_pred.detach())
            finally:
                torch.set_autocast_cache_enabled(prev_cache)

        return torch.stack(velocities, dim=0)

    def _build_forward_kwargs(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
    ) -> Dict[str, Any]:
        """Build kwargs for a single adapter.forward call (noise_pred only)."""
        t_b = timestep.view(-1)
        full_kwargs: Dict[str, Any] = {
            **self.training_args,
            't': t_b,
            't_next': torch.zeros_like(t_b),
            'latents': noised_latents,
            'compute_log_prob': False,
            'return_kwargs': ['noise_pred'],
            'noise_level': self.adapter.scheduler.get_noise_level_for_timestep(t_b),
            **{k: v for k, v in batch.items()
               if k not in ['all_latents', 'timesteps', 'advantage', '__source__']},
        }
        forward_kwargs = filter_forward_kwargs(
            full_kwargs, self._forward_param_names, self._forward_accepts_var_kwargs
        )
        forward_kwargs['return_kwargs'] = ['noise_pred']
        return forward_kwargs

    # =========================================================================
    # Timestep Sampling
    # =========================================================================

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample training timesteps. Returns shape (T, B) in [0, 1000]."""
        device = self.accelerator.device
        strategy = self.time_sampling_strategy.lower()
        available = ['logit_normal', 'uniform', 'discrete', 'discrete_with_init', 'discrete_wo_init']

        if strategy == 'logit_normal':
            return TimeSampler.logit_normal_shifted(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
                stratified=True,
            )
        elif strategy == 'uniform':
            return TimeSampler.uniform(
                batch_size=batch_size,
                num_timesteps=self.num_train_timesteps,
                timestep_range=self.timestep_range,
                time_shift=self.time_shift,
                device=device,
            )
        elif strategy.startswith('discrete'):
            discrete_config = {
                'discrete': (True, False),
                'discrete_with_init': (True, True),
                'discrete_wo_init': (False, False),
            }
            if strategy not in discrete_config:
                raise ValueError(
                    f"Unknown time_sampling_strategy: {strategy}. Available: {available}"
                )
            include_init, force_init = discrete_config[strategy]
            return TimeSampler.discrete(
                batch_size=batch_size,
                num_train_timesteps=self.num_train_timesteps,
                scheduler_timesteps=self.adapter.scheduler.timesteps,
                timestep_range=self.timestep_range,
                include_init=include_init,
                force_init=force_init,
            )
        else:
            raise ValueError(
                f"Unknown time_sampling_strategy: {strategy}. Available: {available}"
            )

    # =========================================================================
    # Data Loading (multi-source, OPD pattern)
    # =========================================================================

    def _interleaved_source_iter(self):
        """Round-robin iterator over per-source dataloaders (OPD pattern).

        Each yielded batch is tagged with ``__source__`` for downstream routing.
        """
        source_names = sorted(self.train_dataloaders_by_source.keys())
        iters = {name: iter(dl) for name, dl in self.train_dataloaders_by_source.items()}

        while True:
            for name in source_names:
                try:
                    batch = next(iters[name])
                except StopIteration:
                    iters[name] = iter(self.train_dataloaders_by_source[name])
                    batch = next(iters[name])
                batch["__source__"] = name
                if "metadata" in batch:
                    for meta in batch["metadata"]:
                        if isinstance(meta, dict):
                            meta["__source__"] = name
                yield batch

    # =========================================================================
    # Advantage Computation
    # =========================================================================

    def compute_advantages(
        self,
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
        store_to_samples: bool = True,
        aggregation_func=None,
    ) -> torch.Tensor:
        """Compute advantages — delegates to AdvantageProcessor."""
        aggregation_func = aggregation_func or self.training_args.advantage_aggregation
        return self.advantage_processor.compute_advantages(
            samples=samples,
            rewards=rewards,
            store_to_samples=store_to_samples,
            aggregation_func=aggregation_func,
        )

    # =========================================================================
    # Per-Set Reward Computation
    # =========================================================================

    def _compute_per_set_rewards(
        self,
        samples: List[BaseSample],
        raw_rewards: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """[DEPRECATED] Compute per-set composite rewards with OOD bonus.

        NOTE: This method is superseded by the per-reward normalization pipeline
        in prepare_feedback() (Step 1→2→3). Kept for backward compatibility and
        potential use in evaluation/debugging contexts where raw-space aggregation
        is acceptable.

        Aggregation only (no normalization): for each sample in set s,
            composite = R_in_domain + gamma * mean(R_ood for applicable j != s)

        Normalization is delegated to GDPO advantage computation downstream
        (group mean subtraction + global std division).

        NaN values (from applicable_sources filtering) are gracefully skipped.

        Returns:
            Dict with single key "mof_reward" → tensor of per-sample rewards.
        """
        gamma = self.training_args.ood_bonus_gamma
        reward_names = list(raw_rewards.keys())
        composite = torch.zeros(len(samples), device=self.accelerator.device)

        for i, sample in enumerate(samples):
            source = sample.extra_kwargs.get("__source__", "default")
            in_domain_name = self._find_in_domain_reward(source, reward_names)

            # Collect applicable (non-NaN) raw rewards for this sample
            applicable: Dict[str, float] = {}
            for rn in reward_names:
                val = raw_rewards[rn][i]
                if not torch.isnan(val):
                    applicable[rn] = val.item()

            if in_domain_name and in_domain_name in applicable:
                composite[i] = applicable[in_domain_name]
                if gamma > 0:
                    ood_vals = [v for k, v in applicable.items() if k != in_domain_name]
                    if ood_vals:
                        composite[i] += gamma * sum(ood_vals) / len(ood_vals)
            elif applicable:
                # Fallback: average all applicable rewards
                composite[i] = sum(applicable.values()) / len(applicable)

        return {"mof_reward": composite}

    def _find_in_domain_reward(self, source: str, reward_names: List[str]) -> Optional[str]:
        """Find the in-domain reward name for a given source.

        Uses explicit mapping from TeacherConfig.reward_name (populated in _init_source_routing).
        """
        reward_name = self._source_to_reward_name.get(source)
        if reward_name and reward_name in reward_names:
            return reward_name
        return None

    # =========================================================================
    # Main Training Loop
    # =========================================================================

    def start(self):
        """Main training loop."""
        # Evaluate each teacher at epoch 0 to establish baselines
        if self.epoch == 0 and self.training_args.eval_teachers_at_start:
            self.evaluate_teachers()

        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)

            # Checkpoint
            if (
                self.log_args.save_freq > 0
                and self.epoch % self.log_args.save_freq == 0
                and self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    'checkpoints',
                )
                self._save_mof_checkpoint(save_dir, epoch=self.epoch)

            # Evaluation
            if (
                self.eval_args.eval_freq > 0
                and self.epoch % self.eval_args.eval_freq == 0
            ):
                self.evaluate()

            # Sampling: use EMA logits if off-policy
            with self.sampling_context():
                samples = self.sample()

            self.prepare_feedback(samples)
            self.optimize(samples)

            # Update EMA of logits
            self._logits_ema.step([self._lambda_logits], optimization_step=self.epoch)
            self.epoch += 1

    # =========================================================================
    # Sampling (override in subclass)
    # =========================================================================

    def sample(self) -> List[BaseSample]:
        """Generate rollouts. Must be overridden by subclass (NFT or GRPO)."""
        raise NotImplementedError("MoFTrainerBase.sample() must be overridden by subclass.")

    # =========================================================================
    # Feedback
    # =========================================================================

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Finalize rewards and compute per-set advantages.

        Implements the correct 3-step pipeline (see Section 9 of analysis doc):
          Step 1: Per-reward, per-group normalization → per-reward advantages
          Step 2: Per-set aggregation in advantage space (a_in + γ * mean(a_ood))
          Step 3: Global-std normalization on the combined advantage

        This ensures γ has consistent semantic meaning regardless of raw reward
        scales (e.g. GenEval ∈ [0,1] vs PickScore ∈ [0.8, 0.95]).

        Handles distributed training via AdvantageProcessor's gather/scatter
        infrastructure (correctly accounts for groups split across ranks).
        """
        device = self.accelerator.device
        raw_rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')

        # Pack source_id as pseudo-reward for cross-rank gathering.
        # All samples in a group share the same source, so this is consistent.
        source_ids = torch.tensor(
            [self._source_to_set_id.get(
                s.extra_kwargs.get("__source__", "default"), 0
            ) for s in samples],
            dtype=torch.float32,
            device=device,
        )
        rewards_with_meta = {**raw_rewards, "__source_id__": source_ids}

        # Gather all per-reward values + source_ids + group indices across ranks
        gathered_rewards, group_indices = self.advantage_processor.collect_group_rewards(
            samples, rewards_with_meta
        )

        # Extract source IDs (rounded to int) and remove from reward dict
        gathered_source_ids = gathered_rewards.pop("__source_id__").astype(np.int64)
        reward_names = list(gathered_rewards.keys())
        N = len(group_indices)

        # ====================================================================
        # Log raw rewards (before any normalization)
        # Format: train/{source_name}/raw_{reward_name}
        # ====================================================================
        log_data: Dict[str, Any] = {}
        for src_name, src_id in self._source_to_set_id.items():
            src_mask = gathered_source_ids == src_id
            if not src_mask.any():
                continue
            for rname in reward_names:
                r_vals = gathered_rewards[rname]
                valid = src_mask & ~np.isnan(r_vals)
                if valid.any():
                    log_data[f"train/{src_name}/raw_{rname}"] = float(np.mean(r_vals[valid]))

        # ====================================================================
        # Step 1: Per-Reward, Per-Group Normalization
        # Each reward is independently normalized to ~N(0,1) within each group.
        # NaN values (non-applicable samples) are skipped.
        # ====================================================================
        per_reward_advantages: Dict[str, np.ndarray] = {}

        for rname in reward_names:
            r_vals = gathered_rewards[rname]  # (N,) numpy, may contain NaN
            a_vals = np.full(N, np.nan, dtype=np.float64)

            for group_id in np.unique(group_indices):
                mask = group_indices == group_id
                group_r = r_vals[mask]

                # Skip NaN values (non-applicable samples for this reward)
                valid = ~np.isnan(group_r)
                if valid.sum() < 2:
                    # Not enough valid samples for meaningful normalization
                    if valid.sum() == 1:
                        a_vals[mask] = np.where(valid, 0.0, np.nan)
                    continue

                valid_r = group_r[valid]
                mu = np.mean(valid_r)
                sigma = max(np.std(valid_r), 1e-8)
                normalized = (valid_r - mu) / sigma

                # Write back only to valid positions within this group
                group_a = np.full(mask.sum(), np.nan)
                group_a[valid] = normalized
                a_vals[mask] = group_a

            per_reward_advantages[rname] = a_vals

        # ====================================================================
        # Step 2: Per-Set Advantage Aggregation
        # For sample from source s: A = a_in(s) + γ * mean(a_ood)
        # Aggregation in advantage space (all a_k already ~N(0,1)).
        # ====================================================================
        gamma = self.training_args.ood_bonus_gamma
        combined_advantages = np.zeros(N, dtype=np.float64)

        for i in range(N):
            source_id = int(gathered_source_ids[i])
            source_name = self._set_id_to_source.get(source_id, "default")
            in_domain_reward = self._source_to_reward_name.get(source_name)

            # Get in-domain advantage
            if in_domain_reward and in_domain_reward in per_reward_advantages:
                in_domain_a = per_reward_advantages[in_domain_reward][i]
                if np.isnan(in_domain_a):
                    in_domain_a = 0.0
            else:
                in_domain_a = 0.0

            # Get OOD advantages (non-NaN, non-in-domain)
            ood_vals = []
            for rname, a_vals in per_reward_advantages.items():
                if rname == in_domain_reward:
                    continue
                val = a_vals[i]
                if not np.isnan(val):
                    ood_vals.append(val)

            # Combine: in-domain + gamma * mean(ood)
            combined = in_domain_a
            if gamma > 0 and ood_vals:
                combined += gamma * np.mean(ood_vals)

            combined_advantages[i] = combined

        # ====================================================================
        # Step 3: Global-Std Normalization (Option B from analysis)
        # Ensures final advantage has consistent scale for NFT loss.
        # ====================================================================
        global_std = max(float(np.std(combined_advantages)), 1e-8)
        final_advantages = combined_advantages / global_std

        # Apply clipping if configured
        adv_clip = getattr(self.training_args, 'adv_clip_range', None)
        if adv_clip is not None:
            clip_min, clip_max = adv_clip[0], adv_clip[1]
            final_advantages = np.clip(final_advantages, clip_min, clip_max)

        # Scatter back to local rank
        local_advantages = self.advantage_processor._to_local(final_advantages)
        if isinstance(local_advantages, torch.Tensor):
            local_advantages = local_advantages.cpu().numpy()

        # Store to samples
        for sample, adv in zip(samples, local_advantages):
            sample.extra_kwargs["advantage"] = float(adv)

        # Log per-source: designed_reward and advantage
        # Format: train/{source_name}/designed_reward, train/{source_name}/advantage
        for src_name, src_id in self._source_to_set_id.items():
            src_mask = gathered_source_ids == src_id
            if src_mask.any():
                log_data[f"train/{src_name}/designed_reward"] = float(
                    np.mean(combined_advantages[src_mask])
                )
                log_data[f"train/{src_name}/advantage"] = float(
                    np.mean(final_advantages[src_mask])
                )

        # Global advantage stats
        log_data["train/advantage_global_std"] = global_std
        # Advantage magnitude diagnostics (before and after normalization)
        abs_combined = np.abs(combined_advantages)
        log_data["train/advantage_pre_norm_abs_mean"] = float(np.mean(abs_combined))
        log_data["train/advantage_pre_norm_abs_min"] = float(np.min(abs_combined))
        log_data["train/advantage_pre_norm_abs_max"] = float(np.max(abs_combined))
        abs_final = np.abs(final_advantages)
        log_data["train/advantage_post_norm_abs_mean"] = float(np.mean(abs_final))
        log_data["train/advantage_post_norm_abs_min"] = float(np.min(abs_final))
        log_data["train/advantage_post_norm_abs_max"] = float(np.max(abs_final))

        # Log training samples (images) for qualitative inspection
        log_data["train_samples"] = samples[:30]
        self.log_data(log_data, step=self.step)

    # =========================================================================
    # Optimization (override in subclass)
    # =========================================================================

    def optimize(self, samples: List[BaseSample]) -> None:
        """Optimize lambda logits. Must be overridden by subclass (NFT or GRPO)."""
        raise NotImplementedError("MoFTrainerBase.optimize() must be overridden by subclass.")

    # =========================================================================
    # Evaluation
    # =========================================================================

    def _evaluate_test_set(self, test_set_name: str) -> None:
        """Override: use per-test-set lambda weights for evaluation.

        Maps test_set_name → set_id so each eval dataset uses its own
        learned teacher mixing strategy (e.g., geneval test uses geneval
        lambda weights, not a default set_id=0).
        """
        set_id = self._source_to_set_id.get(test_set_name, 0)

        self.eval_reward_buffer = RewardBuffer(
            self._eval_reward_processor_for_test_set(test_set_name),
            self.training_args.group_size,
        )
        merged_eval = self._merged_eval_args_for_test_set_name(test_set_name)
        log_pfx = self._eval_log_prefix(test_set_name)
        eval_seed = merged_eval.seed if merged_eval.seed is not None else self.training_args.seed

        with torch.no_grad(), self.autocast(), self._mof_inference_context(set_id):
            all_samples = self._run_eval_inference_batches(test_set_name, merged_eval, eval_seed)
            gathered_rewards = self._gather_eval_rewards()
            gathered_tags = self._gather_eval_tags(all_samples)
            if self.accelerator.is_main_process:
                self._log_eval_reward_metrics(
                    gathered_rewards, log_pfx, all_samples, gathered_tags=gathered_tags
                )
        self.accelerator.wait_for_everyone()

    def _run_eval_inference_batches(
        self,
        test_set_name: str,
        merged_eval,
        eval_seed: int,
    ) -> List[BaseSample]:
        """Override: tag each eval batch with __source__ = test_set_name.

        This ensures eval_rewards with applicable_sources filtering (e.g.,
        geneval reward requiring __source__='geneval') can correctly match
        samples from the corresponding test set.
        """
        all_samples: List[BaseSample] = []
        for batch in tqdm(
            self.test_dataloaders[test_set_name],
            desc=self._eval_progress_desc(test_set_name),
            disable=not self.show_progress_bar,
        ):
            # Tag batch with __source__ so stitch_batch_metadata propagates it
            batch["__source__"] = test_set_name
            if "metadata" in batch:
                for meta in batch["metadata"]:
                    if isinstance(meta, dict):
                        meta["__source__"] = test_set_name

            generator = create_generator_by_prompt(batch["prompt"], eval_seed)
            inference_kwargs = {
                "compute_log_prob": False,
                "generator": generator,
                "trajectory_indices": None,
                **merged_eval,
            }
            inference_kwargs.update(**batch)
            inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
            samples = self.adapter.inference(**inference_kwargs)

            stitch_batch_metadata(batch, samples)

            all_samples.extend(samples)
            self.eval_reward_buffer.add_samples(samples)
        return all_samples

    def evaluate_teachers(self) -> None:
        """Evaluate each teacher independently on its applicable test sets.

        For each teacher k, runs inference using only that teacher's velocity
        (single-teacher context) on every test set that matches its configured
        sources. Results are logged under:
            teacher/{teacher_name}/{test_set_name}/reward_{metric}_mean

        This establishes per-teacher baselines for comparison with the MoF
        student's performance after training.
        """
        if not self.test_dataloaders:
            return

        self.adapter.eval()
        for k, teacher_name in enumerate(self._teacher_names):
            # Evaluate every teacher on ALL test sets (full cross-table)
            applicable_test_sets = sorted(self.test_dataloaders.keys())

            if not applicable_test_sets:
                continue

            logger.info(
                f"Evaluating teacher '{teacher_name}' on test sets: {applicable_test_sets}"
            )

            for ts_name in applicable_test_sets:
                self.eval_reward_buffer = RewardBuffer(
                    self._eval_reward_processor_for_test_set(ts_name),
                    self.training_args.group_size,
                )
                merged_eval = self._merged_eval_args_for_test_set_name(ts_name)
                eval_seed = (
                    merged_eval.seed
                    if merged_eval.seed is not None
                    else self.training_args.seed
                )

                with torch.no_grad(), self.autocast(), \
                        self._single_teacher_inference_context(k):
                    all_samples = self._run_eval_inference_batches(
                        ts_name, merged_eval, eval_seed
                    )
                    gathered_rewards = self._gather_eval_rewards()
                    gathered_tags = self._gather_eval_tags(all_samples)

                    if self.accelerator.is_main_process:
                        log_pfx = f"teacher/{teacher_name}/{ts_name}"
                        self._log_eval_reward_metrics(
                            gathered_rewards, log_pfx, all_samples,
                            gathered_tags=gathered_tags,
                        )
                self.accelerator.wait_for_everyone()

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def _save_mof_checkpoint(self, save_directory: str, epoch: Optional[int] = None):
        """Save MoF-specific state (lambda logits + EMA + source mapping)."""
        if epoch is not None:
            save_directory = os.path.join(save_directory, f"checkpoint-{epoch}")

        if self.accelerator.is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            state = {
                'lambda_logits': self._lambda_logits.detach().cpu(),
                'logits_ema': self._logits_ema.state_dict(),
                'epoch': self.epoch,
                'step': self.step,
                'K': self.K,
                'T': self.num_train_timesteps,
                'S': self.S,
                'source_to_set_id': self._source_to_set_id,
                'reward_running_mean': self._reward_running_mean,
                'reward_running_var': self._reward_running_var,
            }
            save_path = os.path.join(save_directory, 'mof_state.pt')
            torch.save(state, save_path)
            logger.info(f"MoF checkpoint saved to {save_path}")

        self.accelerator.wait_for_everyone()

    def load_mof_checkpoint(self, path: str):
        """Load MoF state from checkpoint."""
        mof_path = os.path.join(path, 'mof_state.pt')

        if not os.path.exists(mof_path):
            raise FileNotFoundError(
                f"MoF checkpoint not found at {mof_path}"
            )

        state = torch.load(mof_path, map_location=self.accelerator.device)
        logits = state['lambda_logits']

        # Validate dimensions
        if logits.shape[0] != self.K:
            raise ValueError(
                f"Checkpoint K={logits.shape[0]} != current K={self.K}"
            )
        if logits.shape[1] != self.num_train_timesteps:
            raise ValueError(
                f"Checkpoint T={logits.shape[1]} != current T={self.num_train_timesteps}"
            )
        if logits.shape[2] != self.S:
            raise ValueError(
                f"Checkpoint S={logits.shape[2]} != current S={self.S}"
            )

        self._lambda_logits.data.copy_(logits.to(self.accelerator.device))
        if 'logits_ema' in state:
            self._logits_ema.load_state_dict(state['logits_ema'])
        self.epoch = state.get('epoch', 0)
        self.step = state.get('step', 0)
        if 'reward_running_mean' in state:
            self._reward_running_mean = state['reward_running_mean']
            self._reward_running_var = state['reward_running_var']
        logger.info(f"MoF checkpoint loaded from {mof_path} (epoch={self.epoch})")

