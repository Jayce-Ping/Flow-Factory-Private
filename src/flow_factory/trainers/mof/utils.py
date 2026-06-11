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

"""Shared utilities for MoF (Mixture-of-Flow) trainers.

Extracted from common.py and distill.py to eliminate copy-paste duplication.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import torch
import torch.nn.functional as F

from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)

#: Valid weight-normalization modes for the mixing module.
#:   'softmax': weights = softmax(logits / temperature) — convex combination,
#:              Σw=1, w∈[0,1]. Gradient has the mean-subtraction property
#:              (∝ g·(v_k − v̄)), which vanishes for LoRA teachers sharing a
#:              base model (docs/mof/mof_per_set_analysis.tex §13).
#:   'affine':  weights = logits − (Σ logits − 1)/K — hard projection onto the
#:              Σw=1 hyperplane. CFG-style: allows w<0 and w>1, preserving FM
#:              validity under shared-marginal teachers, but the Σw=1
#:              constraint still implies the mean-subtraction gradient.
#:   'none':    weights = logits — free linear combination. Largest gradient
#:              (∝ g·v_k, no mean subtraction) but Σw can drift away from 1,
#:              which rescales the combined velocity field (an effective time
#:              reparameterization). Pair with `weight_sum_penalty` to softly
#:              anchor Σw≈1.
WEIGHT_NORMALIZATION_MODES = ("softmax", "affine", "none")


def apply_weight_normalization(
    logits: torch.Tensor,
    mode: str,
    temperature: float = 1.0,
    dim: int = 0,
) -> torch.Tensor:
    """Map raw mixing logits to mixing weights — single source of truth.

    Used by every producer and consumer of MoF mixing weights (LUT modules,
    routers, stage-2 distill, inference/inspection scripts) so that the
    logits→weights semantics can never silently diverge between them.

    Args:
        logits: Raw logits tensor; the teacher axis is ``dim``.
        mode: One of :data:`WEIGHT_NORMALIZATION_MODES`.
        temperature: Softmax temperature. Only used for mode='softmax';
            'affine' and 'none' treat logits as weights directly.
        dim: Teacher (K) axis of ``logits``.

    Returns:
        Mixing weights with the same shape as ``logits``.
    """
    if mode == "softmax":
        return F.softmax(logits / temperature, dim=dim)
    elif mode == "affine":
        K = logits.shape[dim]
        return logits - (logits.sum(dim=dim, keepdim=True) - 1.0) / K
    elif mode == "none":
        return logits
    raise ValueError(
        f"Invalid weight_normalization mode: {mode!r}. "
        f"Valid: {list(WEIGHT_NORMALIZATION_MODES)}."
    )


def resolve_weight_normalization(
    ckpt_value: Optional[str],
    override: Optional[str],
    context: str = "MoF",
) -> str:
    """Resolve the weight-normalization mode for a loaded MoF checkpoint.

    Precedence: explicit ``override`` (warn if it contradicts a recorded
    checkpoint value) > checkpoint value > legacy default 'softmax' with a
    loud warning (checkpoints written before the mode was persisted).

    Args:
        ckpt_value: ``state.get('weight_normalization')`` from mof_state.pt
            (None for legacy checkpoints).
        override: Explicit user override (e.g. ``mof_weight_normalization``
            config field); None means "use the checkpoint value".
        context: Label for log messages.

    Returns:
        One of :data:`WEIGHT_NORMALIZATION_MODES`.
    """
    for label, value in (("checkpoint", ckpt_value), ("override", override)):
        if value is not None and value not in WEIGHT_NORMALIZATION_MODES:
            raise ValueError(
                f"{context}: invalid weight_normalization from {label}: "
                f"{value!r}. Valid: {list(WEIGHT_NORMALIZATION_MODES)}."
            )
    if override is not None:
        if ckpt_value is not None and ckpt_value != override:
            logger.warning(
                f"{context}: overriding checkpoint weight_normalization="
                f"{ckpt_value!r} with explicit {override!r}. The mixing "
                f"weights will be interpreted differently than at train time "
                f"— only do this for mislabeled checkpoints."
            )
        return override
    if ckpt_value is not None:
        return ckpt_value
    logger.warning(
        f"{context}: checkpoint has no 'weight_normalization' metadata "
        f"(legacy format). Assuming 'softmax'. If this checkpoint was "
        f"trained with normalize_weights=false (e.g. a hard-route LUT), "
        f"set the explicit override to 'none' — otherwise the raw weights "
        f"will be incorrectly softmaxed."
    )
    return "softmax"


@contextmanager
def bypass_ddp_for_weight_swap(adapter):
    """Temporarily replace DDP-wrapped transformer with unwrapped module.

    DDP's internal parameter buffers (used for gradient bucketing and
    find_unused_parameters tracking) are NOT updated by .data.copy_()
    which use_named_parameters relies on. In inference mode (no_grad),
    calling the DDP-wrapped module still reads from these stale buffers,
    causing all teacher forwards to produce identical outputs.

    This context manager temporarily points the adapter's transformer
    component to the raw unwrapped module, bypassing DDP's forward path.
    Safe during inference since DDP's gradient sync is not needed.

    Args:
        adapter: The model adapter that provides get_component_unwrapped/set_component.
    """
    unwrapped = adapter.get_component_unwrapped('transformer')
    wrapped = adapter.get_component('transformer')
    if unwrapped is not wrapped:
        adapter.set_component('transformer', unwrapped)
    try:
        yield
    finally:
        if unwrapped is not wrapped:
            adapter.set_component('transformer', wrapped)


def interleaved_source_iter(
    dataloaders_by_source: Dict[str, Any],
    source_ratio: Optional[Dict[str, float]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Block-cycle iterator over per-source dataloaders.

    Each yielded batch is tagged with ``__source__`` key for downstream
    routing (reward filtering, per-set advantage computation, etc.).

    With ``source_ratio=None`` (default), iterates in sorted source-name order
    with equal 1:1:... weighting (legacy round-robin). When a source's dataloader
    is exhausted, it is automatically re-initialized (infinite cycle).

    With ``source_ratio={name: count, ...}``, builds a deterministic block
    pattern by repeating each source name ``count`` times in sorted source-name
    order. E.g. ``{"geneval": 2, "ocr": 2, "pickscore": 1}`` over sources
    ``[geneval, ocr, pickscore]`` yields the cycle ``G G O O P`` repeating.
    All ratio values must be non-negative integer-valued floats; missing /
    unknown source names raise ``ValueError``.

    Args:
        dataloaders_by_source: Dict mapping source name → DataLoader.
        source_ratio: Optional dict mapping source name → integer-valued
            weight. ``None`` means equal weighting.

    Yields:
        Batch dict with ``__source__`` tag and metadata annotated.
    """
    source_names = sorted(dataloaders_by_source.keys())

    if source_ratio is None:
        pattern = list(source_names)
    else:
        unknown = set(source_ratio) - set(source_names)
        missing = set(source_names) - set(source_ratio)
        if unknown:
            raise ValueError(
                f"source_ratio has unknown sources: {sorted(unknown)} "
                f"(available: {source_names})"
            )
        if missing:
            raise ValueError(
                f"source_ratio missing sources: {sorted(missing)} "
                f"(must specify weight for every source in {source_names})"
            )
        pattern = []
        for name in source_names:
            count = source_ratio[name]
            if not float(count).is_integer() or count < 0:
                raise ValueError(
                    f"source_ratio[{name!r}]={count} must be a "
                    f"non-negative integer-valued float (e.g. 2.0)."
                )
            pattern.extend([name] * int(count))
        if not pattern:
            raise ValueError(
                "sum(source_ratio.values()) == 0 — at least one source "
                "must have weight > 0"
            )

    iters = {name: iter(dl) for name, dl in dataloaders_by_source.items()}

    while True:
        for name in pattern:
            try:
                batch = next(iters[name])
            except StopIteration:
                iters[name] = iter(dataloaders_by_source[name])
                batch = next(iters[name])
            batch["__source__"] = name
            if "metadata" in batch:
                for meta in batch["metadata"]:
                    if isinstance(meta, dict):
                        meta["__source__"] = name
            yield batch


def validate_source_ratio(
    source_ratio: Optional[Dict[str, float]],
    num_batches_per_epoch: int,
    train_dataloaders_by_source: Dict[str, Any],
) -> None:
    """Fail-fast check that ``source_ratio`` aligns with the per-epoch loop budget.

    Format errors (unknown/missing keys, non-integer values) are caught
    lazily by ``interleaved_source_iter`` on first use. This function only
    enforces the divisibility invariant that depends on
    ``num_batches_per_epoch``, plus a zero-sum guard so we fail at trainer
    ``__init__`` rather than after sampling starts.

    Args:
        source_ratio: Dict mapping source name → integer-valued weight, or None.
        num_batches_per_epoch: Total iterator ticks per epoch.
        train_dataloaders_by_source: Dict mapping source name → DataLoader.
            When empty (single-source mode) validation is a no-op.

    Raises:
        ValueError: If ``num_batches_per_epoch`` is not divisible by
            ``int(sum(source_ratio.values()))``.
    """
    if source_ratio is None or not train_dataloaders_by_source:
        return
    period = int(sum(source_ratio.values()))
    if period == 0:
        raise ValueError(
            "source_ratio sum is 0 — at least one source must have weight > 0"
        )
    if num_batches_per_epoch % period != 0:
        raise ValueError(
            f"num_batches_per_epoch ({num_batches_per_epoch}) must be divisible "
            f"by sum(source_ratio.values()) ({period}) for clean per-epoch cycles. "
            f"Adjust unique_sample_num_per_epoch or source_ratio. "
            f"Current source_ratio={source_ratio}."
        )
