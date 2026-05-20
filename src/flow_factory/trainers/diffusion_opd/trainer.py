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

Implements the DiffusionOPD paper's multi-task on-policy distillation:

    for each training round:
        L_total ← 0
        for m = 1,...,M:
            Sample prompts c ~ C^(m)
            Roll out student v_θ on c → {x_{t_j}}    // no_grad
            L_m = Σ_j (1/2)||μ_S(x_{t_j}) - μ_T^(m)(x_{t_j})||²
            L_total += L_m
        Update θ via backward(L_total) + optimizer.step()

The rollout and loss computation are interleaved per-timestep to avoid
storing the full trajectory in memory and to eliminate redundant student
forward passes.
"""

from __future__ import annotations

import os
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

import torch
from torch.utils.data import DataLoader

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
)
from ..opd.ode import prepare_train_timesteps

logger = setup_logger(__name__)


class DiffusionOPDTrainer(BaseTrainer):
    """Multi-task On-Policy Distillation trainer for diffusion models.

    Each task has its own prompt dataset and teacher LoRA. In each training
    round, all M tasks are iterated: the student rolls out a trajectory and
    the loss is computed comparing student vs. that task's teacher at each
    timestep. All task losses are summed and a single gradient step is taken.

    Rollout and loss are interleaved per-timestep: the student forward (with
    grad) both produces μ_S for the loss AND advances the trajectory, so the
    student model is called exactly N times per task (not 2N).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.training_args: DiffusionOPDTrainingArguments

        if self.adapter.scheduler.dynamics_type != "ODE":
            raise ValueError(
                "DiffusionOPDTrainer requires `scheduler.dynamics_type == 'ODE'`, "
                f"got {self.adapter.scheduler.dynamics_type!r}. "
                "Set `scheduler.dynamics_type: 'ODE'` and `scheduler.noise_level: 0`."
            )

        # Cache forward signature for efficient kwarg filtering
        self._fwd_params, self._fwd_var_kwargs = cache_forward_signature(
            self.adapter.forward
        )

        # Load M teacher LoRA snapshots
        task_teacher_paths = [t.teacher_path for t in self.training_args.tasks]
        self._teacher_names: List[str] = load_teachers(
            self.adapter,
            task_teacher_paths,
            self.training_args.teacher_param_device,
        )

        # Cache timesteps (constant across training)
        height = self.training_args.height
        width = self.training_args.width
        if height is None or width is None:
            raise ValueError(
                "DiffusionOPDTrainer requires training height and width, "
                f"got height={height!r}, width={width!r}."
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

        # Create per-task dataloaders
        self._task_dataloaders: List[DataLoader] = self._init_task_dataloaders()
        self._task_data_iters: List[Iterator] = [
            iter(dl) for dl in self._task_dataloaders
        ]

        self.num_tasks = len(self.training_args.tasks)
        logger.info(
            f"DiffusionOPDTrainer initialized: {self.num_tasks} task(s), "
            f"{self._num_steps} timesteps, "
            f"pathwise_coef={self.training_args.pathwise_coef}"
        )

    # ========================= Dataloader Setup =========================

    def _init_task_dataloaders(self) -> List[DataLoader]:
        """Create one DataLoader per task from each task's dataset_dir."""
        from ...data_utils.loader import _create_or_load_dataset
        from ...data_utils.sampler_loader import get_data_sampler
        from ...data_utils.dataset import GeneralDataset
        from ...utils.base import filter_kwargs as base_filter_kwargs

        task_dataloaders: List[DataLoader] = []
        data_args = self.config.data_args
        training_args = self.training_args

        for task_idx, task_cfg in enumerate(training_args.tasks):
            logger.info(
                f"  Task {task_idx}: loading dataset from '{task_cfg.dataset_dir}' "
                f"(teacher: {task_cfg.teacher_path})"
            )

            base_kwargs = {
                "preprocess_func": self.adapter.preprocess_func,
                "preprocess_kwargs": (
                    base_filter_kwargs(self.adapter.preprocess_func, **data_args)
                    if self.adapter.preprocess_func
                    else None
                ),
                "extra_hash_strs": [
                    self.config.model_args.model_type,
                    self.config.model_args.model_name_or_path,
                ],
            }
            data_kwargs = base_filter_kwargs(GeneralDataset.__init__, **data_args)
            data_kwargs["dataset_dir"] = task_cfg.dataset_dir
            base_kwargs.update(data_kwargs)
            base_kwargs["force_reprocess"] = data_args.force_reprocess

            train_preprocess_kwargs = (base_kwargs.get("preprocess_kwargs") or {}).copy()
            train_preprocess_kwargs.update({"is_train": True, **training_args})
            train_preprocess_kwargs["guidance_scale"] = (
                training_args.get_preprocess_guidance_scale()
            )
            if self.adapter.preprocess_func:
                train_preprocess_kwargs = base_filter_kwargs(
                    self.adapter.preprocess_func, **train_preprocess_kwargs
                )

            enable_distributed = (
                self.accelerator.num_processes > 1 and data_args.enable_preprocess
            )
            preprocess_parallelism: str = getattr(
                data_args, "preprocess_parallelism", "local"
            )  # type: ignore[assignment]

            dataset = _create_or_load_dataset(
                split="train",
                accelerator=self.accelerator,
                base_kwargs={**base_kwargs, "preprocess_kwargs": train_preprocess_kwargs},
                enable_distributed=enable_distributed,
                preprocess_parallelism=preprocess_parallelism,
            )

            sampler = get_data_sampler(
                dataset=dataset,
                config=self.config,
                accelerator=self.accelerator,
            )

            dl = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=data_args.dataloader_num_workers,
                pin_memory=True,
                collate_fn=GeneralDataset.collate_fn,
            )
            task_dataloaders.append(dl)

        return task_dataloaders

    # ========================= Main Training Loop =========================

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
        """No-op: DiffusionOPD uses teacher KL as the training signal."""
        pass

    # ========================= Optimization (Algorithm 1) =========================

    def optimize(self, samples=None) -> None:
        """One training round of Algorithm 1.

        For each of M tasks: interleave rollout and loss computation
        per-timestep (student forward serves both purposes), then sum
        all task losses for a single backward + optimizer step.
        """
        self.adapter.train()
        device = self.accelerator.device
        loss_info: Dict[str, List[torch.Tensor]] = defaultdict(list)

        with self.accelerator.accumulate(*self.adapter.trainable_components):
            loss_total = None  # Avoid spurious AddBackward0 from zero init

            for m in range(self.num_tasks):
                batch = self._next_task_batch(m, device)

                with self.autocast():
                    loss_m = self._interleaved_rollout_and_loss(
                        batch, teacher_idx=m
                    )

                if loss_total is None:
                    loss_total = loss_m
                else:
                    loss_total = loss_total + loss_m
                loss_info[f"loss_task_{m}"].append(loss_m.detach())

            assert loss_total is not None  # At least one task is guaranteed
            loss_info["loss"].append(loss_total.detach())

            self.accelerator.backward(loss_total)

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

    # ========================= Core Algorithm =========================

    def _interleaved_rollout_and_loss(
        self, batch: Dict[str, Any], teacher_idx: int
    ) -> torch.Tensor:
        """Interleaved on-policy rollout + loss computation.

        Single loop over N timesteps where each student forward (with grad):
        - Produces μ_S for the loss (Eq. 12)
        - Advances the trajectory via x_{j+1} = μ_S.detach()

        This halves student forward passes (N instead of 2N) and eliminates
        O(N) trajectory storage.

        CRITICAL: autocast cache disabled for the entire loop because
        use_named_parameters swaps weights via .data.copy_() preserving
        data_ptr — the autocast cache (keyed by data_ptr) would otherwise
        serve stale casted weights.
        """
        teacher_name = self._teacher_names[teacher_idx]
        device = self.accelerator.device
        timesteps = self._timesteps
        num_steps = self._num_steps

        # Sample initial noise
        x = self._sample_initial_latents(batch).float()
        loss = torch.tensor(0.0, device=device)

        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        try:
            for j in range(num_steps - 1):
                t = timesteps[j]
                t_next = timesteps[j + 1]

                # Student forward (WITH grad) — produces μ_S AND advances rollout
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

                # Teacher forward (no grad, frozen LoRA)
                with torch.no_grad():
                    with self._teacher_frozen_context(teacher_name):
                        teacher_fwd = self._build_forward_kwargs(
                            batch, t, t_next, x,
                            return_kwargs=["next_latents_mean"],
                        )
                        teacher_out = self.adapter.forward(**teacher_fwd)

                if teacher_out.next_latents_mean is None:
                    raise RuntimeError(
                        f"Teacher '{teacher_name}' forward must return "
                        "`next_latents_mean`."
                    )
                mu_T = teacher_out.next_latents_mean

                # D_j = (1/2) * mean(||μ_S - μ_T||²) — Eq. 12
                d_j = 0.5 * (
                    (mu_S.float() - mu_T.float())
                    .pow(2)
                    .flatten(1)
                    .mean(dim=1)
                    .mean()
                )
                loss = loss + d_j

                # Advance trajectory: x_{j+1} = student's predicted next state
                x = mu_S.detach().float()

        finally:
            torch.set_autocast_cache_enabled(prev_cache)

        return self.training_args.pathwise_coef * loss

    # ========================= Helpers =========================

    def _next_task_batch(self, task_idx: int, device: torch.device) -> Dict[str, Any]:
        """Get next batch from task m's dataloader (cyclic restart)."""
        try:
            batch = next(self._task_data_iters[task_idx])
        except StopIteration:
            self._task_data_iters[task_idx] = iter(self._task_dataloaders[task_idx])
            batch = next(self._task_data_iters[task_idx])
        return {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    def _sample_initial_latents(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Sample initial Gaussian noise x_{t_0} for the rollout."""
        prompt_embeds = batch.get("prompt_embeds")
        if prompt_embeds is None:
            raise RuntimeError(
                "DiffusionOPDTrainer requires 'prompt_embeds' in the batch; "
                f"got keys={sorted(batch.keys())}."
            )
        batch_size = prompt_embeds.shape[0]

        pipe = self.adapter.pipeline
        height = self.training_args.height
        width = self.training_args.width
        dtype = pipe.transformer.dtype
        device = self.accelerator.device
        num_channels = pipe.transformer.config.in_channels

        generator = create_generator(
            self.training_args.seed, self.epoch, 0, device=device
        )

        return pipe.prepare_latents(
            batch_size, num_channels, height, width, dtype, device, generator,
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
        """Swap in teacher LoRA + temporarily disable requires_grad on swapped slots.

        Prevents spurious gradient accumulation into the student's LoRA .grad
        channel when the teacher forward runs through the same Parameter slots.
        """
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
