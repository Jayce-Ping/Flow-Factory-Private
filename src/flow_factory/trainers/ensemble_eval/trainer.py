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
evaluates on configured test sets. Each denoising step blends checkpoint
``noise_pred`` outputs with configurable weights before a single scheduler step.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, List

from ..abc import BaseTrainer
from ...hparams import EnsembleEvalTrainingArguments
from ...samples import BaseSample
from ...utils.logger_utils import setup_logger
from .common import (
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

        self._checkpoint_names: List[str] = load_checkpoints(
            self.adapter,
            list(self.training_args.checkpoint_paths),
            self.training_args.checkpoint_param_device,
        )
        self._weights: List[float] = normalize_checkpoint_weights(
            self.training_args.checkpoint_weights,
            len(self._checkpoint_names),
        )
        self._sched_cache = cache_scheduler_step_signature(self.adapter.scheduler.step)

        logger.info(
            f"Ensemble eval: {len(self._checkpoint_names)} checkpoint(s), "
            f"weights={self._weights}."
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
        return f"Ensemble evaluating [{test_set_name}]"

    @contextmanager
    def _eval_inference_context(self) -> Iterator[None]:
        original_forward = self.adapter.forward

        def patched_forward(**kwargs: Any) -> Any:
            return ensemble_forward_step(
                self.adapter,
                self._checkpoint_names,
                self._weights,
                kwargs,
                self._sched_cache,
            )

        self.adapter.forward = patched_forward  # type: ignore[method-assign]
        try:
            yield
        finally:
            self.adapter.forward = original_forward

    def sample(self) -> List[BaseSample]:
        """No-op: ensemble-eval does not sample for training."""
        return []

    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """No-op: ensemble-eval does not compute training feedback."""
        del samples

    def optimize(self, samples: List[BaseSample]) -> None:
        """No-op: ensemble-eval does not update policy weights."""
        del samples
