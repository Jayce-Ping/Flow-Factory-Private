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

# src/flow_factory/trainers/mof.py
"""
MoF (Mixture-of-Flow) Trainer.

Learns optimal per-timestep, per-prompt-set softmax mixing weights over K
frozen teacher flow-matching velocities, optimized via the DiffusionNFT
algorithm with external reward feedback.

Key extension over MoTV: each prompt set (identified by ``__source__``
metadata) has its own independent lambda weights, enabling:
  - Level-1 guarantee: >= single-teacher in-domain performance
  - Level-2 possibility: > single-teacher via timestep complementarity

Theoretical basis: The convex combination of flow-matching velocities is
itself a valid velocity field (ODE linearity). Per-set independence means
each prompt set can find its own optimal teacher schedule.

Total learnable parameters: K × T × S (one weight per teacher per
denoising step per prompt set, with softmax normalization over K).
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

from .abc import BaseTrainer
from ..hparams import MoFTrainingArguments
from ..samples import BaseSample
from ..rewards import RewardBuffer
from ..ema import EMAModuleWrapper
from ..utils.base import (
    filter_kwargs,
    create_generator,
    create_generator_by_prompt,
    to_broadcast_tensor,
    stitch_batch_metadata,
)
from ..utils.logger_utils import setup_logger
from ..utils.noise_schedule import TimeSampler, flow_match_sigma
from ..utils.dist import reduce_loss_info
from .opd.common import load_teachers, cache_forward_signature, filter_forward_kwargs
from .ensemble_eval.common import (
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
        init_mode: str = "teacher_biased",
        init_bias: float = 2.0,
        teacher_set_mapping: Optional[Dict[int, int]] = None,
    ):
        super().__init__()
        self.K = K
        self.T = T
        self.S = S
        self.temperature = temperature

        # Initialize logits
        if init_mode == "zeros":
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

        self.logits = nn.Parameter(data)

    def forward(self) -> torch.Tensor:
        """Compute softmax mixing weights from logits.

        Returns:
            Tensor of shape (K, T, S) normalized over K (dim=0) per (timestep, set).
        """
        return F.softmax(self.logits / self.temperature, dim=0)

    def get_weights_for_set(self, set_id: int) -> torch.Tensor:
        """Get (K, T) weights for a specific prompt set."""
        return self.forward()[:, :, set_id]


class MoFTrainer(BaseTrainer):
    """
    MoF (Mixture-of-Flow): learns per-timestep, per-prompt-set softmax
    mixing weights over K frozen teacher velocities, optimized via
    DiffusionNFT with external reward.

    Key differences from MoTV:
    - Lambda logits shape: (K, T, S) instead of (K, T)
    - Each prompt set gets independent teacher mixing strategy
    - Teacher-to-source routing via TeacherConfig.sources (OPD pattern)
    - Per-set reward with OOD bonus for cross-teacher complementarity
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Note: _initialization() (called by super) already created:
        #   self._teacher_names, self.K, self.S, self._lambda_logits, self.optimizer
        self.training_args: MoFTrainingArguments

        # ---- Unpack config shortcuts ----
        self.nft_beta = self.training_args.nft_beta
        self.off_policy = self.training_args.off_policy
        self.temperature = self.training_args.temperature
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
        """
        self.optimizer = torch.optim.AdamW(
            self._mixing_module.parameters(),
            lr=self.training_args.learning_rate,
            betas=self.training_args.adam_betas,
            weight_decay=0.0,
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
        self._teacher_names: List[str] = load_teachers(
            self.adapter,
            list(self.training_args.teacher_paths),
            self.training_args.teacher_param_device,
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
            init_mode=self.training_args.logits_init,
            init_bias=self.training_args.logits_init_bias,
            teacher_set_mapping=teacher_set_mapping,
        ).to(self.accelerator.device)

        return self._mixing_module.logits

    # =========================================================================
    # Lambda Weights
    # =========================================================================

    def _get_lambda_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute softmax weights from logits: (K, T, S) → (K, T, S).

        Softmax applied over K (dim=0), independently per (timestep, set).
        """
        return F.softmax(logits / self.temperature, dim=0)

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

    @staticmethod
    def _nft_weighted_mse(
        v_pred: torch.Tensor,
        noised_latents: torch.Tensor,
        sigma_broadcast: torch.Tensor,
        clean_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NFT's self-normalized MSE loss (per-sample scalar).

        Computes ||x0_pred - x0||^2 / weight, where weight is the detached
        mean absolute error (prevents gradient through normalization).

        Returns:
            Per-sample loss of shape (B,).
        """
        x0_pred = noised_latents - sigma_broadcast * v_pred
        with torch.no_grad():
            weight = (
                torch.abs(x0_pred.double() - clean_latents.double())
                .mean(dim=tuple(range(1, clean_latents.ndim)), keepdim=True)
                .clip(min=1e-5)
            )
        return (
            (x0_pred - clean_latents) ** 2 / weight
        ).mean(dim=tuple(range(1, clean_latents.ndim)))

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
            return self.adapter.scheduler.step(**scheduler_kwargs)

        self.adapter.forward = patched_forward  # type: ignore[method-assign]
        # Disable autocast cache (weight swaps via .data.copy_)
        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
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
        try:
            step_counter[0] = 0
            yield
        finally:
            torch.set_autocast_cache_enabled(prev_cache)
            self.adapter.forward = original_forward  # type: ignore[method-assign]

    # =========================================================================
    # Teacher Velocity Computation
    # =========================================================================

    def _compute_teacher_velocities(
        self,
        batch: Dict[str, Any],
        timestep: torch.Tensor,
        noised_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Forward each teacher and return stacked velocities (all detached).

        Disables autocast weight cache during the loop (see CLAUDE.md invariant).

        Returns:
            Tensor of shape (K, B, *latent_dims) with all teacher predictions.
        """
        forward_kwargs = self._build_forward_kwargs(batch, timestep, noised_latents)
        velocities = []

        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
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
            'noise_level': 0.0,
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
    # Sampling
    # =========================================================================

    def sample(self) -> List[BaseSample]:
        """Generate rollouts using per-set lambda-combined teacher velocity.

        Uses interleaved source iterator (OPD pattern) so each batch is
        homogeneous per source, enabling efficient per-set inference.
        """
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []

        # Use multi-source iterator if available, else single dataloader
        if self.train_dataloaders_by_source:
            data_iter = self._interleaved_source_iter()
        else:
            data_iter = iter(self.dataloader)

        with torch.no_grad(), self.autocast():
            for _ in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                set_id = self._get_batch_set_id(batch)

                # Use per-set inference context
                with self._mof_inference_context(set_id):
                    sample_kwargs = {
                        **self.training_args,
                        'compute_log_prob': False,
                        'trajectory_indices': [-1],
                        **{k: v for k, v in batch.items() if k != '__source__'},
                    }
                    sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                    sample_batch = self.adapter.inference(**sample_kwargs)

                # Propagate __source__ metadata to samples
                stitch_batch_metadata(batch, sample_batch)

                self._maybe_offload_samples_to_cpu(sample_batch)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)

        return samples

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
    # Optimization
    # =========================================================================

    def optimize(self, samples: List[BaseSample]) -> None:
        """Policy optimization: NFT loss on per-set lambda-weighted teacher velocities.

        For each micro-batch:
        1. Extract per-sample set_ids from __source__ metadata.
        2. Precompute old_v_pred using EMA logits (per-sample set-specific).
        3. Recompute teacher velocities and combine with current logits.
        4. Accumulate gradients over T timesteps, single optimizer step.
        """
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size

        for inner_epoch in range(self.training_args.num_inner_epochs):
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]

            loss_info = defaultdict(list)

            for batch_idx in tqdm(
                range(num_batches),
                total=num_batches,
                desc=f'Epoch {self.epoch} Training',
                position=0,
                disable=not self.show_progress_bar,
            ):
                start = batch_idx * per_device_batch_size
                batch_samples = [
                    sample.to(device)
                    for sample in shuffled_samples[start:start + per_device_batch_size]
                ]
                batch = BaseSample.stack(batch_samples)
                batch_size = batch['all_latents'].shape[0]
                clean_latents = batch['all_latents'][:, -1]

                # Per-sample set IDs
                set_ids = self._get_sample_set_ids(batch_samples)  # (B,)

                # ---- Phase 1: Precompute noised_latents + old_v_pred ----
                # Teacher velocities are cached here for reuse in Phase 2,
                # avoiding redundant K teacher forward passes per timestep.
                self.adapter.rollout()
                with torch.no_grad(), self.autocast():
                    all_timesteps = self._sample_timesteps(batch_size)
                    all_noised_latents: List[torch.Tensor] = []
                    all_sigma_broadcast: List[torch.Tensor] = []
                    all_teacher_velocities: List[torch.Tensor] = []
                    old_v_pred_list: List[torch.Tensor] = []

                    with self.sampling_context():
                        ema_weights = self._get_lambda_weights(self._lambda_logits)

                    for t_idx in range(self.num_train_timesteps):
                        t_flat = all_timesteps[t_idx]
                        sigma_broadcast = to_broadcast_tensor(
                            flow_match_sigma(t_flat), clean_latents
                        )
                        noise = randn_tensor(
                            clean_latents.shape,
                            device=clean_latents.device,
                            dtype=clean_latents.dtype,
                        )
                        noised_latents = (
                            (1 - sigma_broadcast) * clean_latents
                            + sigma_broadcast * noise
                        )
                        all_noised_latents.append(noised_latents)
                        all_sigma_broadcast.append(sigma_broadcast)

                        # Teacher velocities: deterministic given (x_t, t), cache for Phase 2
                        teacher_velocities = self._compute_teacher_velocities(
                            batch, t_flat, noised_latents
                        )
                        all_teacher_velocities.append(teacher_velocities)

                        # Per-sample old_v using EMA weights + set_ids
                        old_v = self._combine_velocities_per_sample(
                            teacher_velocities, t_idx, ema_weights, set_ids
                        )
                        old_v_pred_list.append(old_v.detach())

                # ---- Phase 2: Train with current logits ----
                # Follows NFT trainer pattern: accelerator.accumulate() handles
                # DDP no_sync + auto loss scaling by gradient_accumulation_steps.
                # get_num_train_timesteps() returns num_inference_steps, so
                # gradient_accumulation_steps already includes the T factor.
                self.adapter.train()
                with self.autocast():
                    for t_idx in tqdm(
                        range(self.num_train_timesteps),
                        desc=f'Epoch {self.epoch} Timestep',
                        position=1,
                        leave=False,
                        disable=not self.show_progress_bar,
                    ):
                        with self.accelerator.accumulate(self._mixing_module):
                            t_flat = all_timesteps[t_idx]
                            sigma_broadcast = all_sigma_broadcast[t_idx]
                            noised_latents = all_noised_latents[t_idx]
                            old_v_pred = old_v_pred_list[t_idx]
                            teacher_velocities = all_teacher_velocities[t_idx]

                            # Recompute weights INSIDE the loop to create a fresh
                            # computation graph each iteration. Avoids "backward
                            # through graph a second time" since each iteration
                            # independently builds: _lambda_logits → softmax →
                            # weights → new_v_pred → loss, and backward frees
                            # only that iteration's graph.
                            current_weights = self._get_lambda_weights(self._lambda_logits)

                            # Per-sample combination with current weights
                            new_v_pred = self._combine_velocities_per_sample(
                                teacher_velocities, t_idx, current_weights, set_ids
                            )

                            # NFT loss computation
                            adv = torch.as_tensor(
                                batch['advantage'], dtype=torch.float32, device=device
                            )
                            adv_clip_range = self.training_args.adv_clip_range
                            adv = torch.clamp(adv, adv_clip_range[0], adv_clip_range[1])

                            normalized_adv = (adv / max(adv_clip_range)) / 2.0 + 0.5
                            r = torch.clamp(normalized_adv, 0, 1).view(
                                -1, *([1] * (new_v_pred.dim() - 1))
                            )

                            # Positive/negative predictions
                            positive_pred = (
                                self.nft_beta * new_v_pred
                                + (1 - self.nft_beta) * old_v_pred
                            )
                            negative_pred = (
                                (1.0 + self.nft_beta) * old_v_pred
                                - self.nft_beta * new_v_pred
                            )

                            # NFT self-normalized MSE losses
                            positive_loss = self._nft_weighted_mse(
                                positive_pred, noised_latents, sigma_broadcast, clean_latents
                            )
                            negative_loss = self._nft_weighted_mse(
                                negative_pred, noised_latents, sigma_broadcast, clean_latents
                            )

                            # Combined loss
                            ori_policy_loss = (
                                r.squeeze() * positive_loss
                                + (1.0 - r.squeeze()) * negative_loss
                            ) / self.nft_beta
                            policy_loss = (ori_policy_loss * adv_clip_range[1]).mean()
                            loss = policy_loss

                            # Optional KL penalty
                            if self.enable_kl_loss:
                                with torch.no_grad(), self.adapter.use_ref_parameters():
                                    ref_forward_kwargs = self._build_forward_kwargs(
                                        batch, t_flat, noised_latents
                                    )
                                    ref_output = self.adapter.forward(**ref_forward_kwargs)
                                kl_div = torch.mean(
                                    (new_v_pred - ref_output.noise_pred) ** 2,
                                    dim=tuple(range(1, new_v_pred.ndim)),
                                )
                                kl_loss = self.training_args.kl_beta * kl_div.mean()
                                loss = loss + kl_loss
                                loss_info['kl_div'].append(kl_div.detach())
                                loss_info['kl_loss'].append(kl_loss.detach())

                            loss_info['policy_loss'].append(policy_loss.detach())
                            loss_info['unweighted_policy_loss'].append(
                                ori_policy_loss.mean().detach()
                            )
                            loss_info['loss'].append(loss.detach())

                            # Backward (accumulate handles loss scaling + DDP sync)
                            self.accelerator.backward(loss)

                            # Optimizer step only when gradients are synced
                            # (after all T accumulation steps complete)
                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self._mixing_module.parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()

                                # Log batch-level metrics
                                loss_info_reduced = reduce_loss_info(self.accelerator, loss_info)
                                loss_info_reduced['grad_norm'] = grad_norm
                                with torch.no_grad():
                                    log_weights = self._get_lambda_weights(self._lambda_logits)
                                    mean_weights = log_weights.mean(dim=1)  # (K, S)
                                    for k in range(self.K):
                                        for s in range(self.S):
                                            src_name = self._set_id_to_source.get(s, str(s))
                                            loss_info_reduced[f'lambda_t{k}_{src_name}_mean'] = (
                                                mean_weights[k, s].item()
                                            )
                                self.log_data(
                                    {f'train/{k}': v for k, v in loss_info_reduced.items()},
                                    step=self.step,
                                )
                                self.step += 1
                                loss_info = defaultdict(list)

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
        """Load MoF state from checkpoint (with backward compat for MoTV)."""
        # Try MoF format first, then MoTV format
        mof_path = os.path.join(path, 'mof_state.pt')
        motv_path = os.path.join(path, 'motv_state.pt')

        if os.path.exists(mof_path):
            state_path = mof_path
        elif os.path.exists(motv_path):
            state_path = motv_path
        else:
            raise FileNotFoundError(
                f"MoF checkpoint not found at {mof_path} or {motv_path}"
            )

        state = torch.load(state_path, map_location=self.accelerator.device)

        # Handle MoTV→MoF migration: (K, T) → (K, T, 1)
        logits = state['lambda_logits']
        if logits.ndim == 2:
            logger.info("Loading legacy MoTV checkpoint (K, T) → expanding to (K, T, S)")
            logits = logits.unsqueeze(-1).expand(-1, -1, self.S).contiguous()

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
        logger.info(f"MoF checkpoint loaded from {state_path} (epoch={self.epoch})")
