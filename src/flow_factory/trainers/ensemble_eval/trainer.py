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

# src/flow_factory/trainers/ensemble_eval/trainer.py
"""
Multi-checkpoint offline ensemble evaluation trainer.

Loads multiple LoRA checkpoints as named-parameter snapshots (OPD-style) and
evaluates on configured test sets. Each denoising step fuses checkpoint
``noise_pred`` outputs (linear weighted blend or PCGrad) before a single
scheduler step.

When ``checkpoint_paths`` is empty, runs standard evaluation on the current
adapter weights (no ensemble forward patch).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Union

import torch

from ..abc import BaseTrainer
from ...hparams import EnsembleEvalTrainingArguments
from ...samples import BaseSample
from ...utils.logger_utils import setup_logger
from .common import (
    MERGED_SNAPSHOT_NAME,
    PCGradStats,
    TIESStats,
    build_merged_lora_snapshot,
    cache_scheduler_step_signature,
    ensemble_forward_step,
    load_checkpoints,
    normalize_checkpoint_weights,
)

logger = setup_logger(__name__)


class EnsembleEvalTrainer(BaseTrainer):
    """Eval-only trainer that ensembles multiple LoRA checkpoints at inference."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.training_args: EnsembleEvalTrainingArguments

        self._checkpoint_names: List[str] = []
        self._weights: List[float] = []
        # Effective per-step blend mode. Equals ensemble_blend_mode except for
        # 'weight_merge', which is a one-time weight-space merge run as a single
        # merged checkpoint via the ordinary 'weighted' path (one forward/step).
        self._effective_blend_mode: str = self.training_args.ensemble_blend_mode
        if self.training_args.checkpoint_paths:
            self._checkpoint_names = load_checkpoints(
                self.adapter,
                list(self.training_args.checkpoint_paths),
                self.training_args.checkpoint_param_device,
            )
            self._weights = normalize_checkpoint_weights(
                self.training_args.checkpoint_weights,
                len(self._checkpoint_names),
            )
            blend_mode = self.training_args.ensemble_blend_mode
            if blend_mode == "weight_merge":
                # Collapse the teachers into one merged LoRA snapshot now; eval then
                # runs single-model inference (single merged checkpoint, weighted w=1).
                build_merged_lora_snapshot(
                    self.adapter,
                    self._checkpoint_names,
                    self._weights,
                    device=self.training_args.checkpoint_param_device,
                )
                self._checkpoint_names = [MERGED_SNAPSHOT_NAME]
                self._weights = [1.0]
                self._effective_blend_mode = "weighted"
                logger.info(
                    "Ensemble eval: weight_merge -> single merged LoRA snapshot "
                    f"{MERGED_SNAPSHOT_NAME!r} (single-model inference)."
                )
            elif blend_mode != "weighted" and len(self._checkpoint_names) == 1:
                logger.info(
                    f"Ensemble eval: {blend_mode!r} blend with one checkpoint is "
                    "equivalent to weighted blend (no conflict pairs)."
                )
            logger.info(
                f"Ensemble eval: {len(self._checkpoint_names)} checkpoint(s), "
                f"weights={self._weights}, blend_mode={blend_mode!r}."
            )
        else:
            logger.info(
                "Ensemble eval: checkpoint_paths is empty; evaluating with current "
                "adapter weights (standard forward, no checkpoint ensemble)."
            )

        self._sched_cache = cache_scheduler_step_signature(self.adapter.scheduler.step)
        self._pcgrad_generator: Optional[torch.Generator] = None
        if self._checkpoint_names and self._effective_blend_mode.startswith("pcgrad"):
            self._pcgrad_generator = torch.Generator().manual_seed(
                int(self.training_args.seed)
            )

    def start(self) -> None:
        """Run a single offline evaluation pass over all configured test sets."""
        self.evaluate()

    def _on_no_test_dataloaders_for_eval(self) -> None:
        logger.warning(
            "No test data configured for ensemble-eval; skipping evaluation. "
            "Set eval.test_sets or ensure dataset_dir/test.jsonl exists (legacy)."
        )

    def _eval_progress_desc(self, test_set_name: str) -> str:
        if not self._checkpoint_names:
            return f"Evaluating [{test_set_name}]"
        return f"Ensemble evaluating [{test_set_name}]"

    @contextmanager
    def _bypass_wrapped_transformer(self) -> Iterator[None]:
        """Run the eval scope against the unwrapped ``pipeline.transformer`` --
        **only under plain DDP**, where it is both necessary and safe.

        Plain DDP: the prepared (DDP-wrapped) transformer's forward reads stale
        replica params in ``no_grad``, so per-checkpoint ``use_named_parameters``
        ``.data.copy_()`` swaps are not observed and every teacher collapses to
        identical ``noise_pred`` (all blend modes degenerate to ``weighted``). We
        therefore run forwards against the unwrapped module for the eval scope,
        using ``adapter.get_component_unwrapped`` (== ``pipeline.transformer``, the
        in-place LoRA-injected base). This mirrors
        ``mof.utils.bypass_ddp_for_weight_swap``.

        We DO NOT bypass for:
          * DeepSpeed (ZeRO-1/2): the swap is reflected through the wrapped forward
            natively -- this is why ff-train's default ``deepspeed_zero2`` ensemble
            eval keeps teachers distinct with no bypass. Bypassing would just run a
            non-canonical path for no benefit.
          * Sharded backends (DeepSpeed ZeRO-3 / FSDP): the module holds only param
            *shards* gathered on-demand via hooks on the wrapper. Forwarding through
            the unwrapped module would read empty/partial weights -> garbage/crash.

        Note: when the bypass IS active, ``pipeline.transformer`` is not a
        ``peft.PeftModel`` but its submodules are PEFT tuner layers, so
        ``use_ref_parameters`` (residual / ties ``v_base``) disables those tuner
        layers directly -- ``v_base`` is the exact base model and no warning is
        emitted. On the DeepSpeed path the active component stays the wrapped
        ``PeftModel``, so ``use_ref_parameters`` takes its normal branch.
        """
        wrapped = self.adapter.get_component("transformer")
        unwrapped = self.adapter.get_component_unwrapped("transformer")
        needs_bypass = (
            unwrapped is not wrapped
            and not self.adapter._is_deepspeed()
            and not self.adapter._is_fsdp()
            and not self.adapter._is_param_sharded()
        )
        if needs_bypass:
            self.adapter.set_component("transformer", unwrapped)
        try:
            yield
        finally:
            if needs_bypass:
                self.adapter.set_component("transformer", wrapped)

    @contextmanager
    def _eval_inference_context(self) -> Iterator[None]:
        if not self._checkpoint_names:
            with super()._eval_inference_context():
                yield
            return

        original_forward = self.adapter.forward

        # Create stats accumulator for conflict-resolution modes (deferred logging).
        # Use the effective mode so 'weight_merge' runs as 'weighted' (single merged
        # checkpoint) rather than re-entering a per-step blend path.
        blend_mode = self._effective_blend_mode
        blend_stats: Optional[Union[PCGradStats, TIESStats]] = None
        if blend_mode.startswith("pcgrad"):
            blend_stats = PCGradStats(
                blend_mode=blend_mode,
                num_checkpoints=len(self._checkpoint_names),
            )
        elif blend_mode == "ties":
            blend_stats = TIESStats(
                blend_mode=blend_mode,
                num_checkpoints=len(self._checkpoint_names),
            )

        def patched_forward(**kwargs: Any) -> Any:
            return ensemble_forward_step(
                self.adapter,
                self._checkpoint_names,
                self._weights,
                kwargs,
                self._sched_cache,
                base_forward=original_forward,
                blend_mode=blend_mode,
                pcgrad_eps=self.training_args.pcgrad_eps,
                pcgrad_generator=self._pcgrad_generator,
                ties_density=self.training_args.ties_density,
                weighting=self.training_args.ensemble_blend_weighting,
                stats=blend_stats,
            )

        self.adapter.forward = patched_forward  # type: ignore[method-assign]

        # Disable autocast weight cache for the ensemble scope.
        # use_named_parameters swaps LoRA weights via .data.copy_() which
        # preserves tensor data_ptr; the autocast cache (keyed by data_ptr)
        # would otherwise serve stale casted weights across checkpoint swaps.
        prev_cache_enabled = torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(False)
        # Under plain DDP, bypass the wrapper for the inference scope so the
        # per-checkpoint use_named_parameters swap (.data.copy_) and base_forward
        # read the unwrapped transformer; otherwise stale DDP replica params under
        # no_grad make every checkpoint produce identical noise_pred (all blend
        # modes collapse to weighted). No-op under DeepSpeed (native) and sharded
        # backends (unsafe). See _bypass_wrapped_transformer + CLAUDE.md.
        with self._bypass_wrapped_transformer():
            try:
                yield
            finally:
                torch.set_autocast_cache_enabled(prev_cache_enabled)
                self.adapter.forward = original_forward
                # Log blend-mode summary after all denoising steps complete
                if blend_stats is not None:
                    blend_stats.log_summary()

    def sample(self) -> List[BaseSample]:
        """No-op: ensemble-eval does not sample for training."""
        return []

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """No-op: ensemble-eval does not compute training feedback."""
        del samples

    def optimize(self, samples: List[BaseSample]) -> None:
        """No-op: ensemble-eval does not update policy weights."""
        del samples
