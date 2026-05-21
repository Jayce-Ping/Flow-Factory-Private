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

# src/flow_factory/trainers/diffusion_opd/trainer.py
"""Multi-task DiffusionOPD Trainer (Algorithm 1).

Implements balanced multi-task on-policy distillation: each training round
samples equal numbers of prompts per teacher from the shared dataloader
(which contains all sources merged with __source__ metadata), ensuring
every teacher gets ``per_device_batch_size`` samples from its sources.

Data is declared once in ``data.dataset_dirs`` and preprocessed once by
BaseTrainer._init_dataloader(). No separate per-task dataloaders.
"""

from __future__ import annotations

import os
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, Iterator, List, Set

import torch
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ..abc import BaseTrainer
from ...hparams import DiffusionOPDTrainingArguments
from ...samples import BaseSample
from ...utils.base import create_generator
from ...utils.dist import reduce_loss_info
from ...utils.logger_utils import setup_logger
from ..opd.common import (
    cache_forward_signature,
    filter_forward_kwargs,
    load_teachers,
    prepare_train_timesteps,
)

logger = setup_logger(__name__)


class DiffusionOPDTrainer(BaseTrainer):
    """Multi-task On-Policy Distillation trainer for diffusion models.

    Uses the base class's merged dataloader (all sources concatenated with
    __source__ tags) but samples in a balanced way: each teacher gets exactly
    ``per_device_batch_size`` samples from its assigned sources per step.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.training_args: DiffusionOPDTrainingArguments

        if self.adapter.scheduler.dynamics_type != "ODE":
            raise ValueError(
                "DiffusionOPDTrainer requires `scheduler.dynamics_type == 'ODE'`, "
                f"got {self.adapter.scheduler.dynamics_type!r}."
            )

        # Cache forward signature
        self._fwd_params, self._fwd_var_kwargs = cache_forward_signature(
            self.adapter.forward
        )

        # Load M teacher LoRA snapshots
        self._teacher_names: List[str] = load_teachers(
            self.adapter,
            [tc.path for tc in self.training_args.teachers],
            self.training_args.teacher_param_device,
        )
        self._teacher_sources: List[Set[str]] = [
            set(tc.sources) for tc in self.training_args.teachers
        ]
        self.num_tasks = len(self._teacher_names)

        # Cache timesteps (constant across training)
        height = self.training_args.height
        width = self.training_args.width
        if height is None or width is None:
            raise ValueError(
                f"DiffusionOPDTrainer requires height and width, got {height!r}, {width!r}."
            )
        patch_size = self.adapter.pipeline.transformer.config.patch_size
        self._timesteps = prepare_train_timesteps(
            self.adapter.scheduler,
            num_inference_steps=self.training_args.num_inference_steps,
            height=height,
            width=width,
            patch_size=patch_size,
            device=self.accelerator.device,
        )
        self._num_steps = len(self._timesteps)

        # Per-source dataloader iterators for balanced sampling
        self._source_iters: Dict[str, Iterator] = {
            name: iter(dl) for name, dl in self.train_dataloaders_by_source.items()
        }

        # Validate teacher sources match available dataloaders
        available_sources = set(self.train_dataloaders_by_source.keys())
        for m, tc in enumerate(self.training_args.teachers):
            for src in tc.sources:
                if src not in available_sources:
                    raise ValueError(
                        f"Teacher {m} (path={tc.path!r}) references source '{src}' "
                        f"not in train_dataloaders_by_source. "
                        f"Available: {sorted(available_sources)}. "
                        f"Check data.dataset_dirs has a directory with basename '{src}'."
                    )

        logger.info(
            f"DiffusionOPDTrainer initialized: {self.num_tasks} task(s), "
            f"{self._num_steps} timesteps, "
            f"sources={sorted(available_sources)}, "
            f"pathwise_coef={self.training_args.pathwise_coef}"
        )

    # ========================= Balanced Sampling =========================

    def _next_source_batch(self, source_name: str, device: torch.device) -> Dict[str, Any]:
        """Get next batch from a source's dataloader (cyclic restart)."""
        try:
            batch = next(self._source_iters[source_name])
        except StopIteration:
            self._source_iters[source_name] = iter(
                self.train_dataloaders_by_source[source_name]
            )
            batch = next(self._source_iters[source_name])
        return {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    def _sample_balanced_batches(self, device: torch.device) -> List[Dict[str, Any]]:
        """Sample one batch per teacher from its assigned source dataloaders.

        Each teacher gets one full batch (per_device_batch_size samples) from
        the first source in its sources list. Guarantees equal sample counts.
        """
        per_teacher_batches = []
        for m in range(self.num_tasks):
            # Use the first source for this teacher
            src = list(self._teacher_sources[m])[0]
            batch = self._next_source_batch(src, device)
            per_teacher_batches.append(batch)
        return per_teacher_batches

    def start(self) -> None:
        """Main training loop (Algorithm 1 Stage 2)."""
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

            self.optimize()
            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    def sample(self) -> List[BaseSample]:
        """No-op: DiffusionOPD samples inline within optimize()."""
        return []

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """No-op."""
        pass

    # ========================= Optimization =========================

    def optimize(self, samples=None) -> None:
        """One epoch of Algorithm 1.

        Processes num_batches_per_epoch // num_tasks rounds, each round
        sampling one batch per teacher. Per-timestep backward + gradient
        accumulation keeps peak memory at 1 forward pass.
        """
        self.adapter.train()
        device = self.accelerator.device
        loss_info: Dict[str, List[torch.Tensor]] = defaultdict(list)
        timesteps = self._timesteps
        num_steps = self._num_steps

        # Number of rounds: each round processes M batches (one per teacher)
        batches_per_task = max(1, self.training_args.num_batches_per_epoch // self.num_tasks)
        total_steps = batches_per_task * self.num_tasks * (num_steps - 1)

        # Disable autocast cache for entire optimize (teacher swaps inside)
        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        try:
            step_counter = 0
            for round_idx in tqdm(
                range(batches_per_task),
                desc=f"Epoch {self.epoch} Training",
                disable=not self.show_progress_bar,
            ):
                per_teacher_batches = self._sample_balanced_batches(device)

                for m in range(self.num_tasks):
                    batch = per_teacher_batches[m]
                    teacher_name = self._teacher_names[m]

                    # Sample initial noise for this teacher's rollout
                    x = self._sample_initial_latents(batch).float()

                    for j in range(num_steps - 1):
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
                            with self.autocast():
                                t = timesteps[j]
                                t_next = timesteps[j + 1]

                                # Student forward (WITH grad)
                                student_fwd = self._build_forward_kwargs(
                                    batch, t, t_next, x,
                                    return_kwargs=["next_latents_mean"],
                                )
                                student_out = self.adapter.forward(**student_fwd)
                                if student_out.next_latents_mean is None:
                                    raise RuntimeError(
                                        "Student forward must return `next_latents_mean`."
                                    )
                                mu_S = student_out.next_latents_mean

                                # Teacher forward (no grad, frozen)
                                with torch.no_grad():
                                    with self._teacher_frozen_context(teacher_name):
                                        teacher_fwd = self._build_forward_kwargs(
                                            batch, t, t_next, x,
                                            return_kwargs=["next_latents_mean"],
                                        )
                                        teacher_out = self.adapter.forward(**teacher_fwd)
                                if teacher_out.next_latents_mean is None:
                                    raise RuntimeError(
                                        f"Teacher '{teacher_name}' must return "
                                        "`next_latents_mean`."
                                    )
                                mu_T = teacher_out.next_latents_mean

                                # D_j = pathwise_coef * (1/2) * mean(||μ_S - μ_T||²)
                                d_j = self.training_args.pathwise_coef * 0.5 * (
                                    (mu_S.float() - mu_T.float())
                                    .pow(2)
                                    .flatten(1)
                                    .mean(dim=1)
                                    .mean()
                                )

                            # Per-timestep backward (activations freed immediately)
                            self.accelerator.backward(d_j)
                            loss_info["d_j"].append(d_j.detach())

                            # Advance trajectory (detached from graph)
                            x = mu_S.detach().float()

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
        finally:
            torch.set_autocast_cache_enabled(prev_cache)

    # ========================= Helpers =========================

    def _sample_initial_latents(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Sample initial Gaussian noise x_{t_0}."""
        prompt_embeds = batch.get("prompt_embeds")
        if prompt_embeds is None:
            raise RuntimeError(
                f"DiffusionOPDTrainer requires 'prompt_embeds' in batch; "
                f"got keys={sorted(batch.keys())}."
            )
        batch_size = prompt_embeds.shape[0]
        pipe = self.adapter.pipeline
        dtype = pipe.transformer.dtype
        device = self.accelerator.device
        num_channels = pipe.transformer.config.in_channels

        generator = create_generator(
            self.training_args.seed, self.epoch, self.step, device=device
        )
        return pipe.prepare_latents(
            batch_size, num_channels,
            self.training_args.height, self.training_args.width,
            dtype, device, generator,
        )

    def _build_forward_kwargs(
        self,
        batch: Dict[str, Any],
        t: torch.Tensor,
        t_next: torch.Tensor,
        latents: torch.Tensor,
        return_kwargs: List[str],
    ) -> Dict[str, Any]:
        """Assemble per-step adapter.forward kwargs."""
        full_kwargs = {
            **self.training_args,
            "t": t,
            "t_next": t_next,
            "latents": latents,
            "next_latents": None,
            "compute_log_prob": False,
            "noise_level": 0.0,
            **batch,
        }
        forward_kwargs = filter_forward_kwargs(
            full_kwargs, self._fwd_params, self._fwd_var_kwargs,
        )
        forward_kwargs["return_kwargs"] = return_kwargs
        return forward_kwargs

    @contextmanager
    def _teacher_frozen_context(self, name: str) -> Iterator[None]:
        """Swap in teacher LoRA + disable requires_grad on swapped slots."""
        info = self.adapter._named_parameters[name]
        live = self.adapter._get_component_parameters(info.target_components)
        saved_flags = [p.requires_grad for p in live]
        try:
            with self.adapter.use_named_parameters(name):
                for p in live:
                    p.requires_grad_(False)
                yield
        finally:
            for p, flag in zip(live, saved_flags, strict=True):
                p.requires_grad_(flag)
