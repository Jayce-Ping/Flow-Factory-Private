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

"""MoF On-Policy Distillation Trainer.

Distills a weighted teacher mixture (MoF) into a student LoRA via on-policy
trajectory MSE. The student generates trajectories, then for each trajectory
point the teacher-weighted velocity target is matched via MSE.

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
from ...rewards import RewardBuffer
from ...samples import BaseSample
from ...utils.base import filter_kwargs, create_generator, create_generator_by_prompt, stitch_batch_metadata
from ...utils.logger_utils import setup_logger
from ...utils.dist import reduce_loss_info
from ...utils.trajectory_collector import compute_trajectory_indices
from ..opd.common import load_teachers, cache_forward_signature, filter_forward_kwargs
from .common import create_mixing_module, _validate_teacher_order
from .utils import bypass_ddp_for_weight_swap, interleaved_source_iter

logger = setup_logger(__name__, rank_zero_only=True)

_FORWARD_EXCLUDE_KEYS = frozenset({"all_latents", "timesteps", "__source__", "latent_index_map"})


class MoFDistillTrainer(BaseTrainer):
    """On-policy MoF distillation: sample student trajectories, then MSE-fit teachers.

    Algorithm:
      1. sample(): Student generates full trajectories (stores all_latents at each step)
      2. optimize(samples): Two-pass per batch (matching OPD's precompute pattern):
         Pre-pass  (no_grad): K teacher forwards per timestep → cache v_target
         Main-pass (grad):    student forward only → MSE(v_student, v_target) → backward

    The 2-pass design avoids calling use_named_parameters (parameter swapping)
    inside the gradient-enabled accumulate() scope, preventing DDP/autocast
    cache interference with parameter switching.

    Register as trainer_type: 'mof-distill'.
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

        # ---- Verify teacher snapshots differ from student ----
        self._verify_teacher_snapshots()

        # ---- Cache forward signature ----
        self._forward_param_names, self._forward_accepts_var_kwargs = (
            cache_forward_signature(self.adapter.forward)
        )

        # ---- Load MoF weights ----
        self._load_mof_weights()

        logger.info(
            f"MoF Distill: K={self.K} teachers, "
            f"module_type={self.training_args.mof_module_type}, "
            f"normalize_loss={self.training_args.normalize_d_k}"
        )

    # =========================================================================
    # Teacher Snapshot Verification
    # =========================================================================

    def _verify_teacher_snapshots(self) -> None:
        """Verify teacher parameter snapshots differ from current student weights."""
        student_params = list(self.adapter.get_trainable_parameters())
        if not student_params:
            return

        first_param = student_params[0]
        student_fp = first_param.data.flatten()[:20].detach().cpu()

        all_identical = True
        for name in self._teacher_names:
            info = self.adapter._named_parameters[name]
            teacher_fp = info.ema_wrapper.ema_parameters[0].flatten()[:20].cpu()
            max_diff = float((student_fp - teacher_fp).abs().max())
            if max_diff < 1e-8:
                logger.error(
                    f"Teacher snapshot '{name}' has IDENTICAL weights to student "
                    f"(max_diff={max_diff:.2e}). "
                    f"This indicates _load_lora failed to load teacher weights."
                )
            else:
                all_identical = False
                logger.info(
                    f"Teacher '{name}' snapshot verified: "
                    f"max_diff_from_student={max_diff:.4e}"
                )

        if all_identical:
            raise RuntimeError(
                "ALL teacher snapshots are identical to student weights. "
                "Teacher LoRA loading failed silently. "
                f"Teacher paths: {list(self.training_args.teacher_paths)}"
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

        # Validate teacher list ORDER (not just count). The K axis of the
        # saved LUT/router is position-bound to ``teachers[k]``; if distill's
        # teacher list is reordered relative to MoF training, the saved
        # weights silently apply to the wrong teachers.
        _validate_teacher_order(
            saved_names=state.get("teacher_names"),
            current_names=self._teacher_names,
            context="MoF distill load",
        )

        module_type = args.mof_module_type

        if module_type == "time_router":
            # Time Router: continuous-time MLP, NO text branch.
            # d_pool / d_seq are not used — keep them None for the factory.
            d_hidden = args.mof_hidden_dim
            d_time = args.mof_d_time
            temperature = args.mof_temperature

            saved_arch = state.get("router_arch")
            if saved_arch is not None:
                # d_pool / d_seq are None on both sides for time_router; only
                # validate the dims that actually exist on the module.
                if (
                    saved_arch.get("d_hidden") is not None
                    and saved_arch.get("d_hidden") != d_hidden
                ):
                    logger.warning(
                        f"Distill: config mof_hidden_dim={d_hidden} differs "
                        f"from checkpoint d_hidden={saved_arch['d_hidden']}. "
                        f"Using checkpoint value."
                    )
                    d_hidden = saved_arch["d_hidden"]
                if (
                    saved_arch.get("d_time") is not None
                    and saved_arch.get("d_time") != d_time
                ):
                    logger.warning(
                        f"Distill: config mof_d_time={d_time} differs from "
                        f"checkpoint d_time={saved_arch['d_time']}. Using "
                        f"checkpoint value."
                    )
                    d_time = saved_arch["d_time"]
                if (
                    saved_arch.get("tau") is not None
                    and abs(saved_arch.get("tau") - temperature) > 1e-9
                ):
                    logger.warning(
                        f"Distill: config mof_temperature={temperature} "
                        f"differs from checkpoint tau={saved_arch['tau']}. "
                        f"Using checkpoint value."
                    )
                    temperature = saved_arch["tau"]
            else:
                logger.warning(
                    "MoF time_router checkpoint has no 'router_arch' metadata "
                    "(legacy format). Building router from current config; "
                    "verify mof_hidden_dim / mof_d_time / mof_temperature "
                    "match the training config manually."
                )

            router = create_mixing_module(
                module_type="time_router",
                K=self._mof_K,
                d_hidden=d_hidden,
                d_time=d_time,
                temperature=temperature,
            )
            if "mixing_module_state_dict" in state:
                router.load_state_dict(state["mixing_module_state_dict"])
            else:
                logger.warning("MoF checkpoint missing 'mixing_module_state_dict'.")
            router = router.to(self.accelerator.device).eval()
            for param in router.parameters():
                param.requires_grad_(False)
            self._mof_router = router
            self._mof_weights = None
            logger.info(
                f"MoF distill: loaded time_router "
                f"(K={self._mof_K}, d_hidden={d_hidden}, d_time={d_time}, "
                f"tau={temperature})"
            )
        elif module_type not in ("lut", "lut_simple"):
            # All these fields are first-class dataclass members on
            # MoFDistillTrainingArguments with defined defaults; read them
            # directly. `mof_d_pool=None` means "auto-default to 4096"; any
            # other field is taken at its dataclass default if unset.
            d_pool = args.mof_d_pool or 4096
            d_seq = args.mof_d_seq
            d_hidden = args.mof_hidden_dim
            d_time = args.mof_d_time
            temperature = args.mof_temperature

            # If the checkpoint records the router architecture, prefer
            # those values over config drift. This avoids silently building
            # a router with different dimensions than what was trained.
            saved_arch = state.get("router_arch")
            if saved_arch is not None:
                if (
                    saved_arch.get("d_pool") is not None
                    and saved_arch.get("d_pool") != d_pool
                ):
                    logger.warning(
                        f"Distill: config mof_d_pool={d_pool} differs from "
                        f"checkpoint d_pool={saved_arch['d_pool']}. Using "
                        f"checkpoint value."
                    )
                    d_pool = saved_arch["d_pool"]
                if saved_arch.get("d_seq") is not None and saved_arch.get("d_seq") != d_seq:
                    logger.warning(
                        f"Distill: config mof_d_seq={d_seq} differs from "
                        f"checkpoint d_seq={saved_arch['d_seq']}. Using "
                        f"checkpoint value."
                    )
                    d_seq = saved_arch["d_seq"]
                if (
                    saved_arch.get("d_hidden") is not None
                    and saved_arch.get("d_hidden") != d_hidden
                ):
                    logger.warning(
                        f"Distill: config mof_hidden_dim={d_hidden} differs "
                        f"from checkpoint d_hidden={saved_arch['d_hidden']}. "
                        f"Using checkpoint value."
                    )
                    d_hidden = saved_arch["d_hidden"]
                if (
                    saved_arch.get("d_time") is not None
                    and saved_arch.get("d_time") != d_time
                ):
                    logger.warning(
                        f"Distill: config mof_d_time={d_time} differs from "
                        f"checkpoint d_time={saved_arch['d_time']}. Using "
                        f"checkpoint value."
                    )
                    d_time = saved_arch["d_time"]
                if (
                    saved_arch.get("tau") is not None
                    and abs(saved_arch.get("tau") - temperature) > 1e-9
                ):
                    logger.warning(
                        f"Distill: config mof_temperature={temperature} "
                        f"differs from checkpoint tau={saved_arch['tau']}. "
                        f"Using checkpoint value (router weights were trained "
                        f"with this temperature)."
                    )
                    temperature = saved_arch["tau"]
            else:
                logger.warning(
                    "MoF router checkpoint has no 'router_arch' metadata "
                    "(legacy format). Building router from current config; "
                    "verify mof_d_pool / mof_hidden_dim / mof_temperature "
                    "match the training config manually."
                )

            router = create_mixing_module(
                module_type=module_type,
                K=self._mof_K,
                d_pool=d_pool,
                d_hidden=d_hidden,
                d_time=d_time,
                d_seq=d_seq,
                temperature=temperature,
            )
            if "mixing_module_state_dict" in state:
                router.load_state_dict(state["mixing_module_state_dict"])
            else:
                logger.warning("MoF checkpoint missing 'mixing_module_state_dict'.")
            router = router.to(self.accelerator.device).eval()
            for param in router.parameters():
                param.requires_grad_(False)
            self._mof_router = router
            self._mof_weights = None
            logger.info(
                f"MoF distill: loaded {module_type} router "
                f"(K={self._mof_K}, d_pool={d_pool}, d_seq={d_seq}, "
                f"d_hidden={d_hidden}, d_time={d_time}, tau={temperature})"
            )
        else:
            if args.mof_use_ema and "logits_ema" in state:
                ema_state = state["logits_ema"]
                if "ema_parameters" in ema_state and len(ema_state["ema_parameters"]) > 0:
                    logits = ema_state["ema_parameters"][0]
                    logger.info("MoF distill: using EMA logits")
                else:
                    logits = state["lambda_logits"]
            else:
                logits = state["lambda_logits"]

            # "lut_simple" stores logits of shape (K, T); broadcast to
            # (K, T, S) so the rest of distill (which indexes weights by
            # set_id) works without a separate code path.
            if module_type == "lut_simple":
                if logits.ndim == 2:
                    logits = logits.unsqueeze(-1).expand(
                        self._mof_K, self._mof_T, max(1, self._mof_S)
                    )
                else:
                    # Edge case: legacy ckpt that already saved (K, T, 1) or
                    # similar despite being source-agnostic. Still safe.
                    pass

            weights = F.softmax(logits / args.mof_temperature, dim=0)  # (K, T, S)
            self._mof_weights = weights.to(self.accelerator.device)
            self._mof_router = None

            # Validate T consistency: LUT columns are indexed by inference step
            # (0..T-1), so the student's num_inference_steps must match the T
            # the LUT was trained with — otherwise lookups are misaligned.
            student_T = self.training_args.num_inference_steps
            lut_T = self._mof_weights.shape[1]
            if lut_T != student_T:
                raise ValueError(
                    f"MoF LUT has T={lut_T} columns but distill is configured "
                    f"with num_inference_steps={student_T}. The LUT is indexed "
                    f"by inference step, so these must match. Either re-run "
                    f"MoF training with T={student_T} or set "
                    f"num_inference_steps={lut_T} in the distill config."
                )
            kind = "LUT-SA (source-agnostic)" if module_type == "lut_simple" else "LUT"
            logger.info(
                f"MoF distill: loaded {kind} (K={self._mof_K}, T={self._mof_T}, S={self._mof_S}), "
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
        timestep_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Get per-sample MoF mixing weights (K, B).

        LUT mode: route by (timestep_index → LUT column, __source__ → set_id).
        Router mode: predict from (t, prompt_embeds).

        IMPORTANT (LUT semantics): the MoF LUT was trained with the convention
        that **column 0 corresponds to inference step 0 = highest t (most noisy
        latent)**, and column T-1 corresponds to step T-1 = lowest t (cleanest).
        See ``MoFTrainerBase._mof_inference_context`` and
        ``_combine_velocities_per_sample`` in ``trainers/mof/common.py`` —
        both index the LUT by the loop step counter, not by a normalized
        timestep value. So distill must also use the loop variable
        ``timestep_index`` (0..T-1) directly to query the LUT, NOT
        ``t / TIMESTEP_MAX``, which would produce a reversed mapping
        (e.g., t≈950 → column 8 instead of 0).

        Args:
            t: (B,) timestep values. Used by router mode only.
            batch: stacked batch dict (provides prompt_embeds for router mode).
            batch_samples: per-sample list (provides ``__source__`` for LUT
                source routing).
            timestep_index: scalar inference step index (0..T-1). Required
                for LUT mode; ignored by router mode.
        """
        B = t.shape[0]

        if self._mof_router is not None:
            prompt_embeds = batch.get("prompt_embeds")
            pooled = batch.get("pooled_prompt_embeds")
            with torch.no_grad():
                return self._mof_router(t, prompt_embeds, pooled)  # (K, B)
        else:
            if timestep_index is None:
                raise ValueError(
                    "MoFDistillTrainer LUT mode requires `timestep_index` "
                    "(the inference step counter 0..T-1). The LUT was trained "
                    "with column-as-step semantics; deriving the column from "
                    "the timestep value `t` produces a reversed mapping."
                )
            T_w = self._mof_weights.shape[1]
            t_idx = max(0, min(int(timestep_index), T_w - 1))

            if batch_samples is not None:
                set_ids = torch.tensor(
                    [self._mof_source_to_set_id.get(
                        s.extra_kwargs.get("__source__", "default"), 0
                    ) for s in batch_samples],
                    device=self._mof_weights.device, dtype=torch.long,
                )
            else:
                batch_source = batch.get("__source__", "default")
                set_id = self._mof_source_to_set_id.get(batch_source, 0)
                set_ids = torch.full((B,), set_id, device=self._mof_weights.device, dtype=torch.long)

            # weights[:, t_idx, :] → (K, S); then index by per-sample set_ids
            w_at_t = self._mof_weights[:, t_idx, :]  # (K, S)
            weights_per_sample = w_at_t[:, set_ids]  # (K, B)
            return weights_per_sample

    # =========================================================================
    # Teacher Velocity Computation
    # =========================================================================

    def _compute_teacher_velocities(self, forward_kwargs: Dict[str, Any]) -> torch.Tensor:
        """Forward all K teachers, return stacked velocities (detached). Autocast-safe.

        Disables autocast weight cache and bypasses DDP during the loop
        (see CLAUDE.md invariant and mof/utils.py:bypass_ddp_for_weight_swap).

        Runs under torch.no_grad() to avoid activation storage and prevent DDP
        from registering gradient hooks for teacher forward calls.
        """
        velocities = []
        prev_cache = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        with bypass_ddp_for_weight_swap(self.adapter):
            try:
                with torch.no_grad():
                    for name in self._teacher_names:
                        with self.adapter.use_named_parameters(name):
                            out = self.adapter.forward(**forward_kwargs)
                        velocities.append(out.noise_pred.detach())
            finally:
                torch.set_autocast_cache_enabled(prev_cache)
        return torch.stack(velocities, dim=0)  # (K, B, C, H, W)

    def _combine_weighted(self, weights: torch.Tensor, teacher_velocities: torch.Tensor) -> torch.Tensor:
        """Combine teacher velocities with MoF weights: (K,B) x (K,B,C,H,W) → (B,C,H,W)."""
        n_spatial = teacher_velocities.ndim - 2
        w_expanded = weights.view(self.K, -1, *([1] * n_spatial))
        return (w_expanded * teacher_velocities).sum(dim=0)

    # =========================================================================
    # Forward Kwargs Builder
    # =========================================================================

    def _build_forward_kwargs(
        self, batch: Dict[str, Any], t: torch.Tensor, latents: torch.Tensor,
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
            **{k: v for k, v in batch.items() if k not in _FORWARD_EXCLUDE_KEYS},
        }
        forward_kwargs = filter_forward_kwargs(
            full_kwargs, self._forward_param_names, self._forward_accepts_var_kwargs
        )
        # Re-inject: filter_forward_kwargs strips non-signature keys, but adapters
        # use return_kwargs to control which outputs are computed.
        forward_kwargs["return_kwargs"] = ["noise_pred"]
        return forward_kwargs

    # =========================================================================
    # Data Loading
    # =========================================================================

    def _run_eval_inference_batches(self, test_set_name, merged_eval, eval_seed):
        """Override: tag eval batches with __source__ for reward applicable_sources filtering."""
        all_samples: List[BaseSample] = []
        for batch in tqdm(
            self.test_dataloaders[test_set_name],
            desc=self._eval_progress_desc(test_set_name),
            disable=not self.show_progress_bar,
        ):
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

    # =========================================================================
    # Baseline Evaluation
    # =========================================================================

    def evaluate_baselines(self) -> None:
        """Evaluate each teacher and base model on all test sets."""
        if not self.test_dataloaders:
            return

        self.adapter.eval()

        for k, teacher_name in enumerate(self._teacher_names):
            logger.info(f"Evaluating teacher '{teacher_name}'")
            for ts_name in sorted(self.test_dataloaders.keys()):
                self.eval_reward_buffer = RewardBuffer(
                    self._eval_reward_processor_for_test_set(ts_name),
                    self.training_args.group_size,
                )
                merged_eval = self._merged_eval_args_for_test_set_name(ts_name)
                eval_seed = merged_eval.seed if merged_eval.seed is not None else self.training_args.seed

                prev_cache = torch.is_autocast_cache_enabled()
                torch.set_autocast_cache_enabled(False)
                with bypass_ddp_for_weight_swap(self.adapter):
                    with torch.no_grad(), self.autocast(), \
                            self.adapter.use_named_parameters(teacher_name):
                        all_samples = self._run_eval_inference_batches(ts_name, merged_eval, eval_seed)
                        gathered_rewards = self._gather_eval_rewards()
                        gathered_tags = self._gather_eval_tags(all_samples)
                        if self.accelerator.is_main_process:
                            self._log_eval_reward_metrics(
                                gathered_rewards, f"teacher/{teacher_name}/{ts_name}",
                                all_samples, gathered_tags=gathered_tags,
                            )
                torch.set_autocast_cache_enabled(prev_cache)
                self.accelerator.wait_for_everyone()

        logger.info("Evaluating base model (no LoRA)")
        for ts_name in sorted(self.test_dataloaders.keys()):
            self.eval_reward_buffer = RewardBuffer(
                self._eval_reward_processor_for_test_set(ts_name),
                self.training_args.group_size,
            )
            merged_eval = self._merged_eval_args_for_test_set_name(ts_name)
            eval_seed = merged_eval.seed if merged_eval.seed is not None else self.training_args.seed
            with torch.no_grad(), self.autocast(), self.adapter.use_ref_parameters():
                all_samples = self._run_eval_inference_batches(ts_name, merged_eval, eval_seed)
                gathered_rewards = self._gather_eval_rewards()
                gathered_tags = self._gather_eval_tags(all_samples)
                if self.accelerator.is_main_process:
                    self._log_eval_reward_metrics(
                        gathered_rewards, f"base/{ts_name}",
                        all_samples, gathered_tags=gathered_tags,
                    )
            self.accelerator.wait_for_everyone()

    # =========================================================================
    # Main Training Loop
    # =========================================================================

    @property
    def _train_timestep_indices(self) -> List[int]:
        """Training timestep indices: all steps for ODE, scheduler-selected for SDE."""
        if self.adapter.scheduler.dynamics_type == "ODE":
            return list(range(self.training_args.num_inference_steps))
        return self.adapter.scheduler.train_timesteps

    def start(self):
        """Main training loop with checkpoint/eval/ema bookkeeping."""
        if self.epoch == 0 and self.training_args.eval_baselines_at_start:
            self.evaluate_baselines()

        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)

            if (self.log_args.save_freq > 0
                    and self.epoch % self.log_args.save_freq == 0
                    and self.log_args.save_dir):
                save_dir = os.path.join(
                    self.log_args.save_dir, str(self.log_args.run_name), "checkpoints",
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            if self.eval_args.eval_freq > 0 and self.epoch % self.eval_args.eval_freq == 0:
                self.evaluate()
                self._log_training_rewards()

            samples = self.sample()
            self.prepare_feedback(samples)
            self.optimize(samples)

            self.adapter.ema_step(step=self.epoch)
            self.epoch += 1

    # =========================================================================
    # Sampling
    # =========================================================================

    def sample(self) -> List[BaseSample]:
        """Generate on-policy student trajectories with stored latents."""
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples: List[BaseSample] = []

        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self._train_timestep_indices,
            num_inference_steps=self.training_args.num_inference_steps,
        )

        if self.train_dataloaders_by_source:
            data_iter = interleaved_source_iter(self.train_dataloaders_by_source)
        else:
            data_iter = iter(self.dataloader)

        with torch.no_grad(), self.autocast():
            for _ in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f"Epoch {self.epoch} Sampling",
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    "compute_log_prob": False,
                    "trajectory_indices": trajectory_indices,
                    **batch,
                }
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                stitch_batch_metadata(batch, sample_batch)
                self._maybe_offload_samples_to_cpu(sample_batch)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)

        return samples

    # =========================================================================
    # Feedback (reward logging)
    # =========================================================================

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Finalize rewards on sampled trajectories and log sample images."""
        log_data: Dict[str, Any] = {}

        if self.reward_models:
            rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
            if rewards and self.accelerator.is_main_process:
                for rname, rvals in rewards.items():
                    rvals_np = torch.as_tensor(rvals).cpu().numpy()
                    valid = ~np.isnan(rvals_np)
                    if valid.any():
                        log_data[f"train/reward_{rname}_mean"] = float(np.nanmean(rvals_np))
                    source_groups: Dict[str, List[float]] = defaultdict(list)
                    for i, sample in enumerate(samples):
                        source = sample.extra_kwargs.get("__source__", "default")
                        val = float(rvals_np[i])
                        if not np.isnan(val):
                            source_groups[source].append(val)
                    for source, vals in source_groups.items():
                        if vals:
                            log_data[f"train/{source}/reward_{rname}_mean"] = float(np.mean(vals))

        if self.accelerator.is_main_process:
            log_data["train_samples"] = samples[:30]

        if log_data:
            self.log_data(log_data, step=self.step)

    # =========================================================================
    # Optimization (2-pass: precompute teacher targets, then student gradient)
    # =========================================================================

    def optimize(self, samples: List[BaseSample]) -> None:
        """MSE velocity distillation on stored student trajectories.

        Uses a 2-pass pattern (matching OPD's _precompute + _train_pass):
          Pre-pass  (no_grad): compute and cache all teacher velocities per (batch, timestep)
          Main-pass (grad):    student forward only → MSE against cached v_target → backward

        This avoids calling use_named_parameters (parameter swapping) inside the
        gradient-enabled accumulate() scope, preventing DDP/autocast cache interference.
        """
        device = self.accelerator.device
        per_device_batch_size = self.training_args.per_device_batch_size
        num_batches = (len(samples) + per_device_batch_size - 1) // per_device_batch_size
        train_timestep_indices = self._train_timestep_indices

        self.adapter.train()
        loss_info: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for inner_epoch in range(self.training_args.num_inner_epochs):
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]

            for batch_idx in tqdm(
                range(num_batches),
                desc=f"Epoch {self.epoch} Distill",
                disable=not self.show_progress_bar,
            ):
                start = batch_idx * per_device_batch_size
                batch_samples = [
                    s.to(device)
                    for s in shuffled_samples[start:start + per_device_batch_size]
                ]
                batch = BaseSample.stack(batch_samples)
                latents_index_map = batch["latent_index_map"]

                # ============ Pre-pass: cache teacher targets (no grad) ============
                v_target_by_timestep: List[torch.Tensor] = []
                teacher_velocities_by_timestep: List[torch.Tensor] = []
                weights_by_timestep: List[torch.Tensor] = []
                with torch.no_grad(), self.autocast():
                    for timestep_index in train_timestep_indices:
                        t = batch["timesteps"][:, timestep_index]
                        latents = batch["all_latents"][:, latents_index_map[timestep_index]]
                        forward_kwargs = self._build_forward_kwargs(batch, t, latents)

                        teacher_velocities = self._compute_teacher_velocities(forward_kwargs)
                        weights = self._get_mof_weights(
                            t=t, batch=batch, batch_samples=batch_samples,
                            timestep_index=timestep_index,
                        )
                        v_target = self._combine_weighted(weights, teacher_velocities)
                        v_target_by_timestep.append(v_target)
                        teacher_velocities_by_timestep.append(teacher_velocities)
                        weights_by_timestep.append(weights)

                # ============ Main pass: student forward + loss (with grad) ============
                with self.autocast():
                    for t_idx, timestep_index in enumerate(train_timestep_indices):
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
                            t = batch["timesteps"][:, timestep_index]
                            latents = batch["all_latents"][:, latents_index_map[timestep_index]]
                            forward_kwargs = self._build_forward_kwargs(batch, t, latents)

                            student_out = self.adapter.forward(**forward_kwargs)
                            v_student = student_out.noise_pred

                            v_target = v_target_by_timestep[t_idx]

                            loss = ((v_student.float() - v_target.float()) ** 2).mean()
                            loss_info["loss"].append(loss.detach())

                            # Per-teacher MSE decomposition (detached, for logging only)
                            teacher_vels = teacher_velocities_by_timestep[t_idx]
                            mof_weights = weights_by_timestep[t_idx]  # (K, B)
                            v_student_detached = v_student.detach().float()
                            for k_i in range(self.K):
                                mse_k = ((v_student_detached - teacher_vels[k_i].float()) ** 2).mean()
                                loss_info[f"mse_{self._teacher_names[k_i]}"].append(mse_k)
                                loss_info[f"weight_{self._teacher_names[k_i]}"].append(
                                    mof_weights[k_i].mean().detach()
                                )

                            self.accelerator.backward(loss)

                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.adapter.get_trainable_parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()

                                loss_info_reduced = reduce_loss_info(self.accelerator, loss_info)
                                loss_info_reduced["grad_norm"] = grad_norm
                                self.log_data(
                                    {f"train/{k}": v for k, v in loss_info_reduced.items()},
                                    step=self.step,
                                )
                                self.step += 1
                                loss_info = defaultdict(list)

    # =========================================================================
    # Training Reward Monitoring
    # =========================================================================

    def _log_training_rewards(self) -> None:
        """Generate samples and log per-source reward metrics."""
        if not self.reward_models:
            return

        self.adapter.rollout()
        self.reward_buffer.clear()
        all_samples: List[BaseSample] = []

        if self.train_dataloaders_by_source:
            data_iter = interleaved_source_iter(self.train_dataloaders_by_source)
        else:
            data_iter = iter(self.dataloader)

        num_monitor_batches = min(self.training_args.num_batches_per_epoch, 4)

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

        if all_samples:
            rewards = self.reward_buffer.finalize(store_to_samples=True, split="all")
            if rewards and self.accelerator.is_main_process:
                log_data: Dict[str, Any] = {}
                for rname, rvals in rewards.items():
                    rvals_np = torch.as_tensor(rvals).cpu().numpy()
                    valid = ~np.isnan(rvals_np)
                    if valid.any():
                        log_data[f"train/reward_{rname}_mean"] = float(np.nanmean(rvals_np))
                    source_groups: Dict[str, List[float]] = defaultdict(list)
                    for i, sample in enumerate(all_samples):
                        source = sample.extra_kwargs.get("__source__", "default")
                        val = float(rvals_np[i])
                        if not np.isnan(val):
                            source_groups[source].append(val)
                    for source, vals in source_groups.items():
                        if vals:
                            log_data[f"train/{source}/reward_{rname}_mean"] = float(np.mean(vals))
                log_data["train_samples"] = all_samples[:30]
                self.log_data(log_data, step=self.step)
