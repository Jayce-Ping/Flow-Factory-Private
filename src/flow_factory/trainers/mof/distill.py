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

# src/flow_factory/trainers/mof/distill.py
"""MoF Distillation Trainer: distill a weighted teacher mixture into a student LoRA.

After MoF training learns per-timestep mixing weights λ_{k,t,s}, this trainer
distills the weighted velocity mixture (Σ_k λ_k * v_teacher_k) into a single
student LoRA via pure pathwise MSE loss on velocities.

Loss:
    L = E_{t, x_t} [ ||v_student(x_t, t, c) - Σ_k λ_k(t, s) * v_teacher_k(x_t, t, c)||² ]

    Optionally normalized by 2σ²(t) for time-reweighted MSE (SDE regime).

The trainer is self-contained — no trajectory sampling, no REINFORCE, no
dependency on OPDTrainer. Just:
  1. Sample timestep + noise
  2. Student forward → v_student
  3. All teachers forward → v_teacher_k (detached)
  4. Weighted target: v_target = Σ_k λ_k * v_teacher_k
  5. Loss = MSE(v_student, v_target)
  6. Backward → update student LoRA

Register as trainer_type: 'mof-distill'.
"""
from __future__ import annotations

import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import tqdm as tqdm_

tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from ..abc import BaseTrainer
from ...hparams import MoFDistillTrainingArguments
from ...samples import BaseSample
from ...utils.base import filter_kwargs, stitch_batch_metadata
from ...utils.logger_utils import setup_logger
from ...utils.noise_schedule import flow_match_sigma
from ...utils.dist import reduce_loss_info
from ..opd.common import load_teachers, cache_forward_signature, filter_forward_kwargs
from .common import create_mixing_module

logger = setup_logger(__name__)


class MoFDistillTrainer(BaseTrainer):
    """Distill a MoF-learned weighted teacher mixture into a single student LoRA.

    Pure pathwise MSE distillation — no trajectory sampling, no REINFORCE.
    The student learns to match the MoF-weighted combination of teacher velocities
    at each denoising timestep.

    Supports multi-source data loading (round-robin interleaving) with automatic
    source → set_id routing from the MoF checkpoint.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_args: MoFDistillTrainingArguments

        # ---- Teacher loading ----
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

        # ---- Cache forward signature ----
        self._forward_param_names, self._forward_accepts_var_kwargs = (
            cache_forward_signature(self.adapter.forward)
        )

        # ---- Loss settings ----
        self.normalize_loss = self.training_args.normalize_d_k

        # ---- Load MoF weights ----
        self._load_mof_weights()

        logger.info(
            f"MoF Distill: K={self.K} teachers, "
            f"module_type={self.training_args.mof_module_type}, "
            f"normalize_loss={self.normalize_loss}"
        )

    # =========================================================================
    # MoF Weight Loading
    # =========================================================================

    def _load_mof_weights(self) -> None:
        """Load MoF mixing weights from checkpoint (LUT or router)."""
        args = self.training_args
        checkpoint_path = Path(args.mof_checkpoint)

        # Find mof_state.pt
        if checkpoint_path.is_dir():
            candidates = [checkpoint_path / "mof_state.pt", checkpoint_path / "motv_state.pt"]
            state_path = None
            for c in candidates:
                if c.exists():
                    state_path = c
                    break
            if state_path is None:
                raise FileNotFoundError(
                    f"No mof_state.pt found in {checkpoint_path}. "
                    f"Contents: {[p.name for p in checkpoint_path.iterdir()]}"
                )
        else:
            state_path = checkpoint_path

        state = torch.load(state_path, map_location="cpu", weights_only=False)

        # Source routing (auto from checkpoint)
        self._mof_source_to_set_id: Dict[str, int] = state.get("source_to_set_id", {"default": 0})
        self._mof_K = state.get("K", 0)
        self._mof_T = state.get("T", 0)
        self._mof_S = state.get("S", 1)

        # Validate teacher count
        if self._mof_K != self.K:
            raise ValueError(
                f"MoF checkpoint has K={self._mof_K} teachers but "
                f"{self.K} teacher paths provided. Must match."
            )

        module_type = args.mof_module_type
        self._mof_is_router = module_type != "lut"

        if self._mof_is_router:
            # Router mode: load neural network
            d_text = args.mof_d_text or 4096
            router = create_mixing_module(
                module_type=module_type,
                K=self._mof_K,
                d_text=d_text,
                d_hidden=args.mof_hidden_dim,
                temperature=args.mof_temperature,
            )
            if "mixing_module_state_dict" in state:
                router.load_state_dict(state["mixing_module_state_dict"])
            else:
                logger.warning(
                    "MoF checkpoint missing 'mixing_module_state_dict'. "
                    "Router uses random init (likely incorrect)."
                )
            router = router.to(self.accelerator.device).eval()
            for param in router.parameters():
                param.requires_grad_(False)
            self._mof_router = router
            self._mof_weights = None
            logger.info(f"MoF distill: loaded {module_type} router (K={self._mof_K})")
        else:
            # LUT mode: logits → softmax → frozen weights
            if args.mof_use_ema and "logits_ema" in state:
                ema_state = state["logits_ema"]
                if "ema_parameters" in ema_state and len(ema_state["ema_parameters"]) > 0:
                    logits = ema_state["ema_parameters"][0]
                    logger.info("MoF distill: using EMA logits")
                else:
                    logits = state["lambda_logits"]
            else:
                logits = state["lambda_logits"]

            weights = F.softmax(logits / args.mof_temperature, dim=0)  # (K, T, S)
            self._mof_weights = weights.to(self.accelerator.device)
            self._mof_router = None
            logger.info(
                f"MoF distill: loaded LUT (K={self._mof_K}, T={self._mof_T}, S={self._mof_S}), "
                f"source_map={self._mof_source_to_set_id}"
            )
            for k in range(self.K):
                w = self._mof_weights[k]
                logger.info(
                    f"  {self._teacher_names[k]}: "
                    f"mean={w.mean():.4f}, min={w.min():.4f}, max={w.max():.4f}"
                )

    # =========================================================================
    # MoF Weight Computation
    # =========================================================================

    def _get_mof_weights(
        self,
        t: torch.Tensor,
        batch: Dict[str, Any],
        batch_samples: Optional[List[BaseSample]] = None,
    ) -> torch.Tensor:
        """Get per-sample MoF mixing weights (K, B).

        LUT mode: route by (per-sample t → timestep_index, __source__ → set_id).
        Router mode: predict from (t, prompt_embeds).
        """
        B = t.shape[0]

        if self._mof_is_router:
            prompt_embeds = batch.get("prompt_embeds")
            pooled = batch.get("pooled_prompt_embeds")
            with torch.no_grad():
                return self._mof_router(t, prompt_embeds, pooled)  # (K, B)
        else:
            T_w = self._mof_weights.shape[1]

            # Per-sample timestep → LUT index (t is in [0, 1000] range)
            t_normalized = t.float() / 1000.0  # → [0, 1]
            t_indices = (t_normalized * (T_w - 1)).long().clamp(0, T_w - 1)  # (B,)

            # Per-sample source → set_id
            if batch_samples is not None:
                set_ids = torch.tensor(
                    [
                        self._mof_source_to_set_id.get(
                            s.extra_kwargs.get("__source__", "default"), 0
                        )
                        for s in batch_samples
                    ],
                    device=self._mof_weights.device,
                    dtype=torch.long,
                )
            else:
                # Batch-level source (from interleaved iter: all samples share same source)
                batch_source = batch.get("__source__", "default")
                set_id = self._mof_source_to_set_id.get(batch_source, 0)
                set_ids = torch.full((B,), set_id, device=self._mof_weights.device, dtype=torch.long)

            # Gather per-sample weights: mof_weights[k, t_idx, set_id] for each sample
            # mof_weights shape: (K, T, S)
            weights_per_sample = self._mof_weights[:, t_indices, :]  # (K, B, S)
            # Select per-sample set: (K, B, S) → (K, B) using set_ids
            weights_per_sample = weights_per_sample[
                :, torch.arange(B, device=self._mof_weights.device), set_ids
            ]  # (K, B)
            return weights_per_sample

    # =========================================================================
    # Teacher Velocity Computation
    # =========================================================================

    def _compute_teacher_velocities(
        self,
        forward_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """Forward all K teachers and return stacked velocities (detached).

        Disables autocast weight cache during weight swap (CLAUDE.md invariant).

        Returns:
            (K, B, C, H, W) detached teacher velocity predictions.
        """
        velocities = []
        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        try:
            for name in self._teacher_names:
                with self.adapter.use_named_parameters(name):
                    out = self.adapter.forward(**forward_kwargs)
                velocities.append(out.noise_pred.detach())
        finally:
            torch.set_autocast_cache_enabled(prev_cache)

        return torch.stack(velocities, dim=0)  # (K, B, C, H, W)

    # =========================================================================
    # Forward Kwargs Builder
    # =========================================================================

    def _build_forward_kwargs(
        self,
        batch: Dict[str, Any],
        t: torch.Tensor,
        latents: torch.Tensor,
    ) -> Dict[str, Any]:
        """Build kwargs for adapter.forward (noise_pred only, no log_prob)."""
        full_kwargs = {
            **self.training_args,
            "t": t,
            "t_next": torch.zeros_like(t),
            "latents": latents,
            "compute_log_prob": False,
            "return_kwargs": ["noise_pred"],
            "noise_level": 0.0,
            **{k: v for k, v in batch.items()
               if k not in ["all_latents", "timesteps", "__source__"]},
        }
        forward_kwargs = filter_forward_kwargs(
            full_kwargs, self._forward_param_names, self._forward_accepts_var_kwargs
        )
        forward_kwargs["return_kwargs"] = ["noise_pred"]
        return forward_kwargs

    # =========================================================================
    # Data Loading
    # =========================================================================

    def _interleaved_source_iter(self):
        """Round-robin iterator over per-source dataloaders."""
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
    # Timestep Sampling
    # =========================================================================

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps for distillation. Returns (B,) in [0, 1]."""
        device = self.accelerator.device
        # Uniform sampling in [0, 1] (flow matching convention)
        return torch.rand(batch_size, device=device)

    # =========================================================================
    # Main Training Loop
    # =========================================================================

    def start(self):
        """Main training loop: pure MSE velocity distillation."""
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
                    "checkpoints",
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            # Evaluation + training reward monitoring
            if self.eval_args.eval_freq > 0 and self.epoch % self.eval_args.eval_freq == 0:
                self.evaluate()
                self._log_training_rewards()

            # Optimize (velocity MSE distillation)
            self.optimize()

            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    def _log_training_rewards(self) -> None:
        """Generate a batch of training samples and log per-source reward metrics.

        Runs inference on a subset of training data (from each source),
        computes rewards, and logs as train/{source}/{reward_name}.
        This provides training-time quality monitoring without affecting
        the distillation loss.
        """
        if not self.reward_models:
            return

        self.adapter.rollout()
        self.reward_buffer.clear()
        all_samples: List[BaseSample] = []

        # Generate samples from each source
        if self.train_dataloaders_by_source:
            data_iter = self._interleaved_source_iter()
        else:
            data_iter = iter(self.dataloader)

        # Generate a small number of batches for monitoring
        num_monitor_batches = min(
            getattr(self.training_args, 'num_batches_per_epoch', 4), 4
        )

        with torch.no_grad(), self.autocast():
            for _ in range(num_monitor_batches):
                batch = next(data_iter)
                inference_kwargs = {
                    **self.training_args,
                    "compute_log_prob": False,
                    **batch,
                }
                inference_kwargs = filter_kwargs(self.adapter.inference, **inference_kwargs)
                samples = self.adapter.inference(**inference_kwargs)
                stitch_batch_metadata(batch, samples)
                all_samples.extend(samples)
                self.reward_buffer.add_samples(samples)

        # Compute rewards and log per-source
        if all_samples:
            rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
            if rewards and self.accelerator.is_main_process:
                log_data: Dict[str, Any] = {}
                for rname, rvals in rewards.items():
                    rvals_np = torch.as_tensor(rvals).cpu().numpy()
                    # Global mean
                    valid = ~np.isnan(rvals_np)
                    if valid.any():
                        log_data[f"train/reward_{rname}_mean"] = float(np.nanmean(rvals_np))

                    # Per-source breakdown
                    source_groups: Dict[str, List[float]] = defaultdict(list)
                    for i, sample in enumerate(all_samples):
                        source = sample.extra_kwargs.get("__source__", "default")
                        val = float(rvals_np[i])
                        if not np.isnan(val):
                            source_groups[source].append(val)

                    for source, vals in source_groups.items():
                        if vals:
                            log_data[f"train/{source}/reward_{rname}_mean"] = float(np.mean(vals))

                # Log training sample images
                log_data["train_samples"] = all_samples[:30]
                self.log_data(log_data, step=self.step)

    def optimize(self) -> None:
        """One epoch of MSE velocity distillation.

        For each batch:
          1. Sample random timestep t ~ U[0, 1]
          2. Noise latents: x_t = (1-σ(t)) * x_0 + σ(t) * ε
          3. Student forward → v_student
          4. All teachers forward → v_teacher_k (detached)
          5. MoF weighted target: v_target = Σ_k λ_k(t, s) * v_teacher_k
          6. Loss = MSE(v_student, v_target)
          7. Backward → optimizer.step()
        """
        device = self.accelerator.device
        num_batches = self.training_args.num_batches_per_epoch

        # Data iterator
        if self.train_dataloaders_by_source:
            data_iter = self._interleaved_source_iter()
        else:
            data_iter = iter(self.dataloader)

        self.adapter.train()
        loss_info = defaultdict(list)

        with self.autocast():
            for batch_idx in tqdm(
                range(num_batches),
                desc=f"Epoch {self.epoch} Distill",
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)

                # Move batch data to device (prompt_embeds, etc.)
                batch_on_device = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Get clean latents from batch (encoded images)
                # For distillation we need latent-space data
                inference_kwargs = {
                    **self.training_args,
                    "compute_log_prob": False,
                    **batch_on_device,
                }
                inference_kwargs = filter_kwargs(self.adapter.encode_latents, **inference_kwargs)
                clean_latents = self.adapter.encode_latents(**inference_kwargs)  # (B, C, H, W)

                B = clean_latents.shape[0]

                # Sample timestep
                t = self._sample_timesteps(B)  # (B,) in [0, 1]
                sigma = flow_match_sigma(t)  # (B,)

                # Noise latents
                noise = torch.randn_like(clean_latents)
                sigma_broadcast = sigma.view(B, *([1] * (clean_latents.ndim - 1)))
                noised_latents = (1 - sigma_broadcast) * clean_latents + sigma_broadcast * noise

                # Scale timestep to scheduler range (typically [0, 1000])
                t_scaled = t * 1000.0

                # Build forward kwargs
                forward_kwargs = self._build_forward_kwargs(
                    batch=batch_on_device, t=t_scaled, latents=noised_latents
                )

                with self.accelerator.accumulate(self.adapter.get_trainable_module()):
                    # Student forward
                    student_out = self.adapter.forward(**forward_kwargs)
                    v_student = student_out.noise_pred  # (B, C, H, W)

                    # Teacher forwards (all K, detached)
                    teacher_velocities = self._compute_teacher_velocities(forward_kwargs)
                    # (K, B, C, H, W)

                    # MoF weighted target (per-sample t and source routing)
                    weights = self._get_mof_weights(
                        t=t_scaled,
                        batch=batch_on_device,
                    )  # (K, B)

                    # Combine: v_target = Σ_k w_k * v_teacher_k
                    n_spatial = teacher_velocities.ndim - 2
                    w_expanded = weights.view(self.K, B, *([1] * n_spatial))
                    v_target = (w_expanded * teacher_velocities).sum(dim=0)  # (B, C, H, W)

                    # MSE loss
                    diff_sq = (v_student.float() - v_target.float()) ** 2
                    per_sample_loss = diff_sq.mean(dim=tuple(range(1, diff_sq.ndim)))  # (B,)

                    if self.normalize_loss:
                        # Time-weighted: divide by 2σ²(t) for SDE regime
                        sigma_sq = (sigma ** 2).clamp(min=1e-12)
                        per_sample_loss = per_sample_loss / (2.0 * sigma_sq)

                    loss = per_sample_loss.mean()

                    # Logging
                    loss_info["loss"].append(loss.detach())
                    loss_info["mse_raw"].append(diff_sq.mean().detach())

                    # Backward
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.adapter.get_trainable_parameters(),
                            self.training_args.max_grad_norm,
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # Log
                        loss_info_reduced = reduce_loss_info(self.accelerator, loss_info)
                        loss_info_reduced["grad_norm"] = grad_norm
                        self.log_data(
                            {f"train/{k}": v for k, v in loss_info_reduced.items()},
                            step=self.step,
                        )
                        self.step += 1
                        loss_info = defaultdict(list)
