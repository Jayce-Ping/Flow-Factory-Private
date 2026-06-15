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

# src/flow_factory/trainers/ensemble_eval/common.py
"""Shared helpers for multi-checkpoint ensemble evaluation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Literal, Optional, Sequence, Tuple, Union, get_args

import torch

from ...utils.logger_utils import setup_logger
from ...utils.lora_loader import load_lora_as_named_parameters
from ..opd.common import cache_forward_signature, filter_forward_kwargs

if TYPE_CHECKING:
    from ...models.abc import BaseAdapter

logger = setup_logger(__name__)

SchedulerStepCache = Tuple[FrozenSet[str], bool]

EnsembleBlendMode = Literal[
    "weighted",
    "pcgrad",
    "pcgrad_residual",
    "pcgrad_channelwise",
    "pcgrad_normalized",
    "pcgrad_residual_normalized",
    "pcgrad_residual_channelwise",
    "ties",
]
ENSEMBLE_BLEND_MODES: Tuple[str, ...] = get_args(EnsembleBlendMode)

# Conflict-projection strategy shared by the full-velocity and residual-delta paths.
PCGradProjection = Literal["global", "channelwise", "normalized"]

# blend_mode -> projection for the two PCGrad families (the remaining modes
# 'weighted' and 'ties' are dispatched explicitly in ensemble_forward_step).
_FULL_PROJECTIONS: Dict[str, "PCGradProjection"] = {
    "pcgrad": "global",
    "pcgrad_channelwise": "channelwise",
    "pcgrad_normalized": "normalized",
}
_RESIDUAL_PROJECTIONS: Dict[str, "PCGradProjection"] = {
    "pcgrad_residual": "global",
    "pcgrad_residual_channelwise": "channelwise",
    "pcgrad_residual_normalized": "normalized",
}


# ---------------------------------------------------------------------------
# PCGrad Statistics Accumulator
# ---------------------------------------------------------------------------


@dataclass
class PCGradStats:
    """Accumulates PCGrad conflict statistics across denoising steps.

    Create one instance per evaluation run, pass it to ``ensemble_forward_step``
    (via the ``stats`` parameter), then call :meth:`log_summary` after evaluation
    completes.
    """

    # Per-step raw counters
    num_steps: int = 0
    total_pairs: int = 0
    conflict_pairs: int = 0
    total_elements: int = 0  # batch elements (global) or group elements (channelwise)
    conflict_elements: int = 0

    # Per-step cosine similarity accumulators (mean/min/max across steps)
    _cosine_means: List[float] = field(default_factory=list)
    _cosine_mins: List[float] = field(default_factory=list)
    _cosine_maxs: List[float] = field(default_factory=list)

    # Metadata (set on first call)
    blend_mode: str = ""
    tensor_shape: Tuple[int, ...] = ()
    num_checkpoints: int = 0

    def record_step(
        self,
        *,
        step_total_pairs: int,
        step_conflict_pairs: int,
        step_total_elements: int,
        step_conflict_elements: int,
        cosine_means: Optional[List[float]] = None,
        cosine_mins: Optional[List[float]] = None,
        cosine_maxs: Optional[List[float]] = None,
    ) -> None:
        """Record statistics from one denoising step."""
        self.num_steps += 1
        self.total_pairs += step_total_pairs
        self.conflict_pairs += step_conflict_pairs
        self.total_elements += step_total_elements
        self.conflict_elements += step_conflict_elements
        if cosine_means:
            self._cosine_means.extend(cosine_means)
        if cosine_mins:
            self._cosine_mins.extend(cosine_mins)
        if cosine_maxs:
            self._cosine_maxs.extend(cosine_maxs)

    def log_summary(self) -> None:
        """Log accumulated statistics as a single summary message."""
        if self.num_steps == 0:
            return

        conflict_rate = (
            self.conflict_elements / self.total_elements
            if self.total_elements > 0
            else 0.0
        )
        pairs_with_conflict_rate = (
            self.conflict_pairs / self.total_pairs
            if self.total_pairs > 0
            else 0.0
        )

        summary_lines = [
            f"PCGrad summary ({self.blend_mode}): "
            f"{self.num_checkpoints} checkpoints, "
            f"{self.num_steps} denoising steps, "
            f"tensor_shape={self.tensor_shape}.",
            f"  Conflict rate: {conflict_rate:.4f} "
            f"({self.conflict_elements}/{self.total_elements} elements with dot<0 "
            f"across all steps).",
            f"  Pairs with ≥1 conflict: {pairs_with_conflict_rate:.4f} "
            f"({self.conflict_pairs}/{self.total_pairs}).",
        ]

        if self._cosine_means:
            avg_cos = sum(self._cosine_means) / len(self._cosine_means)
            min_cos = min(self._cosine_mins) if self._cosine_mins else float("nan")
            max_cos = max(self._cosine_maxs) if self._cosine_maxs else float("nan")
            summary_lines.append(
                f"  Cosine similarity (across all steps): "
                f"avg_mean={avg_cos:.4f}, "
                f"global_min={min_cos:.4f}, "
                f"global_max={max_cos:.4f}."
            )

        if self.conflict_elements == 0:
            summary_lines.append(
                "  WARNING: No conflicts detected across any step. "
                "Result is identical to weighted_sum."
            )

        logger.info("\n".join(summary_lines))


@dataclass
class TIESStats:
    """Accumulates TIES-merging sign-disagreement statistics across steps.

    Create one instance per evaluation run, pass it to ``ensemble_forward_step``
    (via the ``stats`` parameter) when ``ensemble_blend_mode='ties'``, then call
    :meth:`log_summary` after evaluation completes.
    """

    num_steps: int = 0
    total_elements: int = 0  # non-zero (checkpoint, element) task entries
    disagree_elements: int = 0  # entries dropped because sign != elected sign

    blend_mode: str = "ties"
    tensor_shape: Tuple[int, ...] = ()
    num_checkpoints: int = 0

    def record_step(
        self,
        *,
        step_total_elements: int,
        step_disagree_elements: int,
    ) -> None:
        """Record statistics from one denoising step."""
        self.num_steps += 1
        self.total_elements += step_total_elements
        self.disagree_elements += step_disagree_elements

    def log_summary(self) -> None:
        """Log accumulated statistics as a single summary message."""
        if self.num_steps == 0:
            return

        disagree_rate = (
            self.disagree_elements / self.total_elements
            if self.total_elements > 0
            else 0.0
        )
        summary_lines = [
            f"TIES summary ({self.blend_mode}): "
            f"{self.num_checkpoints} checkpoints, "
            f"{self.num_steps} denoising steps, "
            f"tensor_shape={self.tensor_shape}.",
            f"  Sign-disagreement rate: {disagree_rate:.4f} "
            f"({self.disagree_elements}/{self.total_elements} non-zero task "
            f"entries dropped by the sign vote across all steps).",
        ]
        if self.disagree_elements == 0:
            summary_lines.append(
                "  WARNING: No sign disagreement detected across any step. "
                "Result is identical to weighted_sum."
            )
        logger.info("\n".join(summary_lines))


def load_checkpoints(
    adapter: "BaseAdapter",
    checkpoint_paths: List[str],
    checkpoint_param_device: str,
) -> List[str]:
    """Load each LoRA checkpoint into a named-parameter snapshot.

    Args:
        adapter: Active ``BaseAdapter`` in LoRA mode.
        checkpoint_paths: LoRA paths accepted by
            :func:`load_lora_as_named_parameters`.
        checkpoint_param_device: ``'cpu'`` or ``'cuda'`` for snapshot storage.

    Returns:
        Ordered snapshot names ``['eval_ckpt_0', 'eval_ckpt_1', ...]``.

    Raises:
        ValueError: ``checkpoint_paths`` is empty.
    """
    if not checkpoint_paths:
        raise ValueError(
            "ensemble-eval requires at least one checkpoint path; "
            f"got checkpoint_paths={checkpoint_paths!r}."
        )

    checkpoint_names: List[str] = []
    for i, path in enumerate(checkpoint_paths):
        name = f"eval_ckpt_{i}"
        load_lora_as_named_parameters(
            adapter=adapter,
            name=name,
            lora_path=path,
            device=checkpoint_param_device,
        )
        checkpoint_names.append(name)
    logger.info(
        f"Loaded {len(checkpoint_names)} ensemble checkpoint(s): {checkpoint_names} "
        f"(device={checkpoint_param_device!r})."
    )
    return checkpoint_names


def normalize_checkpoint_weights(
    weights: Optional[Sequence[float]],
    num_checkpoints: int,
) -> List[float]:
    """Return normalized blend weights that sum to 1.

    Args:
        weights: Optional per-checkpoint weights. When ``None``, uses uniform
            weights ``1 / num_checkpoints``.
        num_checkpoints: Number of loaded checkpoints.

    Returns:
        Normalized weight list of length ``num_checkpoints``.

    Raises:
        ValueError: Invalid ``weights`` length, negative entries, or zero sum.
    """
    if num_checkpoints < 1:
        raise ValueError(
            f"num_checkpoints must be >= 1 for weight normalization, got {num_checkpoints}."
        )
    if weights is None:
        return [1.0 / num_checkpoints] * num_checkpoints

    weight_list = list(weights)
    if len(weight_list) != num_checkpoints:
        raise ValueError(
            f"checkpoint_weights length must match checkpoint_paths ({num_checkpoints}), "
            f"got len(checkpoint_weights)={len(weight_list)}."
        )
    if any(w < 0 for w in weight_list):
        raise ValueError(
            f"All checkpoint_weights must be >= 0, got checkpoint_weights={weight_list!r}."
        )
    total = sum(weight_list)
    if total <= 0:
        raise ValueError(f"checkpoint_weights must sum to a positive value, got {weight_list!r}.")
    return [w / total for w in weight_list]


def cache_scheduler_step_signature(
    scheduler_step_fn: Callable[..., Any],
) -> SchedulerStepCache:
    """Cache ``scheduler.step`` parameter names for cheap per-step filtering."""
    return cache_forward_signature(scheduler_step_fn)


def _build_scheduler_step_kwargs(
    forward_kwargs: Dict[str, Any],
    combined_noise_pred: torch.Tensor,
    sched_cache: SchedulerStepCache,
) -> Dict[str, Any]:
    """Map adapter ``forward`` kwargs to ``scheduler.step`` kwargs."""
    param_names, accepts_var_kwargs = sched_cache
    return_kwargs = forward_kwargs.get("return_kwargs")
    if return_kwargs is None:
        return_kwargs = [
            "noise_pred",
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
        ]

    full_scheduler_kwargs: Dict[str, Any] = {
        "noise_pred": combined_noise_pred,
        "timestep": forward_kwargs.get("t"),
        "latents": forward_kwargs.get("latents"),
        "timestep_next": forward_kwargs.get("t_next"),
        "next_latents": forward_kwargs.get("next_latents"),
        "generator": forward_kwargs.get("generator"),
        "noise_level": forward_kwargs.get("noise_level"),
        "compute_log_prob": forward_kwargs.get("compute_log_prob", False),
        "log_prob_reduction": forward_kwargs.get("log_prob_reduction", "mean"),
        "return_dict": True,
        "return_kwargs": return_kwargs,
        "dynamics_type": forward_kwargs.get("dynamics_type"),
        "sigma_max": forward_kwargs.get("sigma_max"),
    }
    return filter_forward_kwargs(full_scheduler_kwargs, param_names, accepts_var_kwargs)


def _batchwise_broadcast_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    """Shape ``(B, 1, 1, ...)`` for per-batch scalars broadcast over ``tensor``."""
    return (tensor.shape[0],) + (1,) * (tensor.ndim - 1)


def _shuffled_other_indices(
    num_checkpoints: int,
    exclude: int,
    generator: Optional[torch.Generator],
) -> List[int]:
    """Return checkpoint indices ``j != exclude``, in random order."""
    indices = [j for j in range(num_checkpoints) if j != exclude]
    if len(indices) <= 1:
        return indices
    if generator is not None:
        perm = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[p] for p in perm]
    random.shuffle(indices)
    return indices


def pcgrad_blend_noise_preds(
    scaled_preds: Sequence[torch.Tensor],
    *,
    eps: float = 1e-8,
    generator: Optional[torch.Generator] = None,
    stats: Optional[PCGradStats] = None,
) -> torch.Tensor:
    """Blend checkpoint ``noise_pred`` tensors with PCGrad conflict projection.

    Each entry in ``scaled_preds`` is typically ``weight_k * noise_pred_k`` with
    weights normalized to sum to 1. For each pair ``(i, j)`` with negative
    per-batch dot product, the component of ``pc[i]`` along ``scaled_preds[j]``
    is removed (using the **original** ``scaled_preds[j]``, per the PCGrad paper).
    The result is ``sum_k pc[k]``.

    Args:
        scaled_preds: Per-checkpoint velocity tensors, same shape.
        eps: Minimum value for ``||v_j||^2`` when dividing.
        generator: Optional RNG for shuffling inner-loop task order.
        stats: Optional accumulator for deferred summary logging.

    Returns:
        Combined ``noise_pred`` tensor.

    Raises:
        ValueError: Empty sequence, shape mismatch, or invalid ``eps``.
        TypeError: Non-tensor entries in ``scaled_preds``.
    """
    if not scaled_preds:
        raise ValueError(
            "pcgrad_blend_noise_preds requires at least one tensor, got empty sequence."
        )
    if eps <= 0:
        raise ValueError(f"pcgrad_eps must be > 0, got {eps}.")

    ref = scaled_preds[0]
    if not isinstance(ref, torch.Tensor):
        raise TypeError(
            f"pcgrad_blend_noise_preds expected torch.Tensor, got {type(ref).__name__}."
        )
    if ref.ndim < 1:
        raise ValueError(
            f"pcgrad_blend_noise_preds expected batch dimension (ndim >= 1), got ndim={ref.ndim}."
        )
    ref_shape = ref.shape
    for idx, pred in enumerate(scaled_preds):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(
                f"pcgrad_blend_noise_preds expected torch.Tensor at index {idx}, "
                f"got {type(pred).__name__}."
            )
        if pred.shape != ref_shape:
            raise ValueError(
                f"pcgrad_blend_noise_preds expected all tensors to share shape {tuple(ref_shape)}, "
                f"got index {idx} shape {tuple(pred.shape)}."
            )

    if len(scaled_preds) == 1:
        return ref

    batch = ref_shape[0]
    broadcast_shape = _batchwise_broadcast_shape(ref)
    flat_orig = [pred.reshape(batch, -1) for pred in scaled_preds]
    norm_sq_orig = [
        (flat_j * flat_j).sum(dim=1).clamp_min(eps).view(broadcast_shape)
        for flat_j in flat_orig
    ]

    num_checkpoints = len(scaled_preds)
    pc = [pred.clone() for pred in scaled_preds]

    total_pairs = 0
    conflict_pairs = 0
    conflict_batches = 0
    total_batches = 0

    for i in range(num_checkpoints):
        for j in _shuffled_other_indices(num_checkpoints, i, generator):
            flat_pc_i = pc[i].reshape(batch, -1)
            dot = (flat_pc_i * flat_orig[j]).sum(dim=1).view(broadcast_shape)
            coeff = dot / norm_sq_orig[j]
            proj = coeff * scaled_preds[j]
            conflict_mask = dot < 0
            pc[i] = torch.where(conflict_mask, pc[i] - proj, pc[i])

            if stats is not None:
                n_conflicts = int(conflict_mask.sum().item())
                total_pairs += 1
                total_batches += batch
                conflict_batches += n_conflicts
                if n_conflicts > 0:
                    conflict_pairs += 1

    # Record statistics (deferred logging via stats.log_summary())
    if stats is not None and total_pairs > 0:
        if stats.num_steps == 0:
            stats.tensor_shape = tuple(ref_shape)
        # Reuse pre-computed norm_sq_orig for cosine similarity (avoid redundant .norm())
        norm_orig = [
            ns.view(batch, -1).squeeze(1).sqrt()  # (B,)
            for ns in norm_sq_orig
        ]
        cosine_means: List[float] = []
        cosine_mins: List[float] = []
        cosine_maxs: List[float] = []
        for i in range(num_checkpoints):
            for j in range(num_checkpoints):
                if i == j:
                    continue
                cosine_sim = (
                    (flat_orig[i] * flat_orig[j]).sum(dim=1)
                    / (norm_orig[i] * norm_orig[j]).clamp_min(eps)
                )
                cosine_means.append(cosine_sim.mean().item())
                cosine_mins.append(cosine_sim.min().item())
                cosine_maxs.append(cosine_sim.max().item())
        stats.record_step(
            step_total_pairs=total_pairs,
            step_conflict_pairs=conflict_pairs,
            step_total_elements=total_batches,
            step_conflict_elements=conflict_batches,
            cosine_means=cosine_means,
            cosine_mins=cosine_mins,
            cosine_maxs=cosine_maxs,
        )

    result = pc[0]
    for k in range(1, num_checkpoints):
        result = result + pc[k]
    return result


def pcgrad_blend_noise_preds_channelwise(
    scaled_preds: Sequence[torch.Tensor],
    *,
    eps: float = 1e-8,
    generator: Optional[torch.Generator] = None,
    stats: Optional[PCGradStats] = None,
) -> torch.Tensor:
    """Blend checkpoint ``noise_pred`` tensors with per-channel/per-token PCGrad.

    Unlike :func:`pcgrad_blend_noise_preds` which computes a single dot product
    over all spatial+channel dimensions per batch element, this function computes
    dot products at a finer granularity:

    - **4D tensors** ``(B, C, H, W)``: per-channel conflict detection. Each
      channel independently decides whether to project (dot over ``H*W``).
    - **3D tensors** ``(B, seq_len, feat)``: per-token conflict detection. Each
      spatial token independently decides whether to project (dot over ``feat``).

    General rule for ``ndim >= 3``: group dimension is ``dim=1``, feature
    dimensions are ``dim=2..end``.

    Args:
        scaled_preds: Per-checkpoint velocity tensors (weighted), same shape.
        eps: Minimum squared norm when dividing in projection.
        generator: Optional RNG for shuffling inner-loop task order.
        stats: Optional accumulator for deferred summary logging.

    Returns:
        Combined ``noise_pred`` tensor.

    Raises:
        ValueError: Empty sequence, shape mismatch, ndim < 3, or invalid ``eps``.
        TypeError: Non-tensor entries.
    """
    if not scaled_preds:
        raise ValueError(
            "pcgrad_blend_noise_preds_channelwise requires at least one tensor, "
            "got empty sequence."
        )
    if eps <= 0:
        raise ValueError(f"pcgrad_eps must be > 0, got {eps}.")

    ref = scaled_preds[0]
    if not isinstance(ref, torch.Tensor):
        raise TypeError(
            f"pcgrad_blend_noise_preds_channelwise expected torch.Tensor, "
            f"got {type(ref).__name__}."
        )
    if ref.ndim < 3:
        raise ValueError(
            f"pcgrad_blend_noise_preds_channelwise requires ndim >= 3 for "
            f"channel grouping, got ndim={ref.ndim}. Use pcgrad_blend_noise_preds "
            f"(global mode) for 1D/2D tensors."
        )
    ref_shape = ref.shape
    for idx, pred in enumerate(scaled_preds):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(
                f"pcgrad_blend_noise_preds_channelwise expected torch.Tensor at "
                f"index {idx}, got {type(pred).__name__}."
            )
        if pred.shape != ref_shape:
            raise ValueError(
                f"pcgrad_blend_noise_preds_channelwise expected all tensors to "
                f"share shape {tuple(ref_shape)}, got index {idx} shape "
                f"{tuple(pred.shape)}."
            )

    if len(scaled_preds) == 1:
        return ref

    batch = ref_shape[0]
    group_dim_size = ref_shape[1]  # C for 4D, seq_len for 3D
    group_batch = batch * group_dim_size
    # broadcast_shape: (B, group_dim, 1, 1, ...) with (ndim-2) trailing 1s
    broadcast_shape = (batch, group_dim_size) + (1,) * (ref.ndim - 2)

    # Flatten: (B*group_dim, feature_dims_product)
    flat_orig = [pred.reshape(group_batch, -1) for pred in scaled_preds]
    norm_sq_orig = [
        (flat_j * flat_j).sum(dim=1).clamp_min(eps).view(broadcast_shape)
        for flat_j in flat_orig
    ]

    num_checkpoints = len(scaled_preds)
    pc = [pred.clone() for pred in scaled_preds]

    total_pairs = 0
    conflict_pairs = 0
    conflict_groups = 0
    total_groups = 0

    for i in range(num_checkpoints):
        for j in _shuffled_other_indices(num_checkpoints, i, generator):
            flat_pc_i = pc[i].reshape(group_batch, -1)
            dot = (flat_pc_i * flat_orig[j]).sum(dim=1).view(broadcast_shape)
            coeff = dot / norm_sq_orig[j]
            proj = coeff * scaled_preds[j]
            conflict_mask = dot < 0
            pc[i] = torch.where(conflict_mask, pc[i] - proj, pc[i])

            if stats is not None:
                n_conflicts = int(conflict_mask.sum().item())
                total_pairs += 1
                total_groups += group_batch
                conflict_groups += n_conflicts
                if n_conflicts > 0:
                    conflict_pairs += 1

    # Record statistics (deferred logging via stats.log_summary())
    if stats is not None and total_pairs > 0:
        if stats.num_steps == 0:
            stats.tensor_shape = tuple(ref_shape)
        stats.record_step(
            step_total_pairs=total_pairs,
            step_conflict_pairs=conflict_pairs,
            step_total_elements=total_groups,
            step_conflict_elements=conflict_groups,
        )

    result = pc[0]
    for k in range(1, num_checkpoints):
        result = result + pc[k]
    return result


def pcgrad_blend_noise_preds_normalized(
    vectors: Sequence[torch.Tensor],
    weights: Sequence[float],
    *,
    eps: float = 1e-8,
    generator: Optional[torch.Generator] = None,
    stats: Optional[PCGradStats] = None,
) -> torch.Tensor:
    """Magnitude-normalized global PCGrad blend.

    Splits each ``vectors[i]`` into a per-sample magnitude ``n_i = ||vectors[i]||``
    and unit direction ``u_i = vectors[i] / n_i``. The PCGrad pairwise projection
    runs on the **unit directions** so the conflict geometry is magnitude-invariant
    (a high-norm checkpoint cannot dominate the projection coefficient
    ``<u_i, u_j>``). The projected unit directions ``p_i`` are recombined as
    ``sum_i w_i * n_i * p_i``, restoring each checkpoint's original magnitude and
    blend weight.

    No conflicts reduce to the weighted blend ``sum_i w_i * vectors[i]``; a single
    vector returns ``w_0 * vectors[0]``.

    Args:
        vectors: Per-checkpoint velocity tensors, same shape.
        weights: Per-checkpoint blend weights (same length as ``vectors``).
        eps: Minimum value for ``||v||^2`` when normalizing / dividing.
        generator: Optional RNG for shuffling inner-loop task order.
        stats: Optional accumulator for deferred summary logging.

    Returns:
        Combined ``noise_pred`` tensor.

    Raises:
        ValueError: Empty sequence, shape mismatch, length mismatch, or invalid ``eps``.
        TypeError: Non-tensor entries in ``vectors``.
    """
    if not vectors:
        raise ValueError(
            "pcgrad_blend_noise_preds_normalized requires at least one tensor, "
            "got empty sequence."
        )
    if eps <= 0:
        raise ValueError(f"pcgrad_eps must be > 0, got {eps}.")
    if len(vectors) != len(weights):
        raise ValueError(
            "pcgrad_blend_noise_preds_normalized expected len(vectors) == "
            f"len(weights), got len(vectors)={len(vectors)}, "
            f"len(weights)={len(weights)}."
        )

    ref = vectors[0]
    if not isinstance(ref, torch.Tensor):
        raise TypeError(
            f"pcgrad_blend_noise_preds_normalized expected torch.Tensor, "
            f"got {type(ref).__name__}."
        )
    if ref.ndim < 1:
        raise ValueError(
            "pcgrad_blend_noise_preds_normalized expected batch dimension "
            f"(ndim >= 1), got ndim={ref.ndim}."
        )
    ref_shape = ref.shape
    for idx, vec in enumerate(vectors):
        if not isinstance(vec, torch.Tensor):
            raise TypeError(
                "pcgrad_blend_noise_preds_normalized expected torch.Tensor at "
                f"index {idx}, got {type(vec).__name__}."
            )
        if vec.shape != ref_shape:
            raise ValueError(
                "pcgrad_blend_noise_preds_normalized expected all tensors to "
                f"share shape {tuple(ref_shape)}, got index {idx} shape "
                f"{tuple(vec.shape)}."
            )

    if len(vectors) == 1:
        return weights[0] * vectors[0]

    batch = ref_shape[0]
    broadcast_shape = _batchwise_broadcast_shape(ref)

    # Per-sample magnitude (B, 1, 1, ...) and unit direction.
    norm = [
        vec.reshape(batch, -1).pow(2).sum(dim=1).clamp_min(eps).sqrt().view(broadcast_shape)
        for vec in vectors
    ]
    unit = [vec / n for vec, n in zip(vectors, norm, strict=True)]

    flat_unit = [u.reshape(batch, -1) for u in unit]
    norm_sq_unit = [
        (fu * fu).sum(dim=1).clamp_min(eps).view(broadcast_shape) for fu in flat_unit
    ]

    num_checkpoints = len(vectors)
    pc = [u.clone() for u in unit]

    total_pairs = 0
    conflict_pairs = 0
    conflict_batches = 0
    total_batches = 0

    for i in range(num_checkpoints):
        for j in _shuffled_other_indices(num_checkpoints, i, generator):
            flat_pc_i = pc[i].reshape(batch, -1)
            dot = (flat_pc_i * flat_unit[j]).sum(dim=1).view(broadcast_shape)
            coeff = dot / norm_sq_unit[j]
            proj = coeff * unit[j]
            conflict_mask = dot < 0
            pc[i] = torch.where(conflict_mask, pc[i] - proj, pc[i])

            if stats is not None:
                n_conflicts = int(conflict_mask.sum().item())
                total_pairs += 1
                total_batches += batch
                conflict_batches += n_conflicts
                if n_conflicts > 0:
                    conflict_pairs += 1

    if stats is not None and total_pairs > 0:
        if stats.num_steps == 0:
            stats.tensor_shape = tuple(ref_shape)
        cosine_means: List[float] = []
        cosine_mins: List[float] = []
        cosine_maxs: List[float] = []
        for i in range(num_checkpoints):
            for j in range(num_checkpoints):
                if i == j:
                    continue
                # Unit directions: dot product is already the cosine similarity.
                cosine_sim = (flat_unit[i] * flat_unit[j]).sum(dim=1)
                cosine_means.append(cosine_sim.mean().item())
                cosine_mins.append(cosine_sim.min().item())
                cosine_maxs.append(cosine_sim.max().item())
        stats.record_step(
            step_total_pairs=total_pairs,
            step_conflict_pairs=conflict_pairs,
            step_total_elements=total_batches,
            step_conflict_elements=conflict_batches,
            cosine_means=cosine_means,
            cosine_mins=cosine_mins,
            cosine_maxs=cosine_maxs,
        )

    result = weights[0] * norm[0] * pc[0]
    for k in range(1, num_checkpoints):
        result = result + weights[k] * norm[k] * pc[k]
    return result


def _blend_velocity_set(
    vectors: Sequence[torch.Tensor],
    weights: Sequence[float],
    *,
    projection: "PCGradProjection",
    eps: float = 1e-8,
    generator: Optional[torch.Generator] = None,
    stats: Optional[PCGradStats] = None,
) -> torch.Tensor:
    """Blend velocity tensors with the requested PCGrad projection.

    Shared by the full-velocity and residual-delta paths so each projection
    works in both spaces:

    - ``global``: PCGrad on ``w_i * vectors[i]`` with one dot product per sample.
    - ``channelwise``: PCGrad on ``w_i * vectors[i]`` with per-channel (4D) or
      per-token (3D) dot products.
    - ``normalized``: magnitude-normalized PCGrad on unit directions, recombined
      as ``sum_i w_i * n_i * p_i``.

    Raises:
        ValueError: Unknown ``projection``.
    """
    if projection == "normalized":
        return pcgrad_blend_noise_preds_normalized(
            vectors, weights, eps=eps, generator=generator, stats=stats
        )

    scaled_preds = [vec * weight for vec, weight in zip(vectors, weights, strict=True)]
    if projection == "channelwise":
        return pcgrad_blend_noise_preds_channelwise(
            scaled_preds, eps=eps, generator=generator, stats=stats
        )
    if projection == "global":
        return pcgrad_blend_noise_preds(
            scaled_preds, eps=eps, generator=generator, stats=stats
        )
    raise ValueError(
        "_blend_velocity_set expected projection in "
        f"('global', 'channelwise', 'normalized'), got projection={projection!r}."
    )


def _pcgrad_residual_blend(
    adapter: "BaseAdapter",
    checkpoint_names: Sequence[str],
    weights: Sequence[float],
    noise_only_kwargs: Dict[str, Any],
    base_forward: Callable[..., Any],
    pcgrad_eps: float,
    pcgrad_generator: Optional[torch.Generator],
    projection: "PCGradProjection" = "global",
    stats: Optional[PCGradStats] = None,
) -> torch.Tensor:
    """Compute PCGrad on deltas from the pretrained model noise_pred.

    Steps:
        1. Run ``base_forward`` with all LoRA adapters disabled to get the
           pretrained (reference) model's ``noise_pred`` (``v_b``).
        2. For each checkpoint, compute ``tau_i = noise_pred_i - v_b``.
        3. Blend the task-specific deltas with the requested ``projection``
           (``global`` / ``channelwise`` / ``normalized``) via
           :func:`_blend_velocity_set` (blend weights applied inside). These
           corrections are much more likely to conflict than the full predictions.
        4. Return ``v_b + combined_delta``.

    Note:
        This adds one extra forward pass per denoising step for the pretrained model.
    """
    # 1. Get pretrained (reference) noise_pred with adapters disabled
    with torch.no_grad(), adapter.use_ref_parameters():
        ref_out = base_forward(**noise_only_kwargs)
    if ref_out.noise_pred is None:
        raise RuntimeError(
            "Pretrained model forward did not return `noise_pred` in residual "
            "PCGrad mode; check that the adapter supports "
            "return_kwargs=['noise_pred']."
        )
    ref_noise_pred = ref_out.noise_pred

    # 2. Compute raw task-specific deltas from the pretrained baseline.
    taus: List[torch.Tensor] = []
    for name in checkpoint_names:
        with adapter.use_named_parameters(name):
            out = base_forward(**noise_only_kwargs)
        if out.noise_pred is None:
            raise RuntimeError(
                f"Checkpoint '{name}' forward did not return `noise_pred`; "
                "check that the adapter supports return_kwargs=['noise_pred']."
            )
        taus.append(out.noise_pred - ref_noise_pred)

    # 3. Blend deltas with the requested projection (weights applied inside).
    combined_delta = _blend_velocity_set(
        taus,
        weights,
        projection=projection,
        eps=pcgrad_eps,
        generator=pcgrad_generator,
        stats=stats,
    )

    # 4. Add back pretrained baseline.
    return ref_noise_pred + combined_delta


def ties_blend_deltas(
    taus: Sequence[torch.Tensor],
    weights: Sequence[float],
    *,
    density: float = 1.0,
    stats: Optional[TIESStats] = None,
) -> torch.Tensor:
    """Merge task-specific deltas with TIES-merging sign election.

    Implements the elect-sign + disjoint-merge steps of TIES-merging on
    per-checkpoint task vectors ``tau_i`` (typically ``noise_pred_i - v_b``):

    1. (Optional) Trim each ``tau_i`` to its top-``density`` fraction of entries
       by magnitude (per sample), zeroing the rest.
    2. Elect a per-element sign ``gamma = sign(sum_i w_i * tau_i)``.
    3. Disjoint weighted mean over sign-agreeing checkpoints:
       ``tau_merged = sum_i w_i*tau_i*1[sign(tau_i)=gamma]
                      / sum_i w_i*1[sign(tau_i)=gamma]`` (0 where no agreement).

    No sign disagreement reduces to the weighted blend ``sum_i w_i * tau_i``
    (weights normalized); a single delta returns ``tau_0``.

    Args:
        taus: Per-checkpoint task-vector tensors, same shape.
        weights: Per-checkpoint blend weights (same length as ``taus``).
        density: Fraction of largest-magnitude entries to keep per task
            (``1.0`` = no trim). Must be in ``(0, 1]``.
        stats: Optional accumulator for deferred summary logging.

    Returns:
        Merged delta tensor.

    Raises:
        ValueError: Empty sequence, shape mismatch, length mismatch, or invalid ``density``.
        TypeError: Non-tensor entries in ``taus``.
    """
    if not taus:
        raise ValueError("ties_blend_deltas requires at least one tensor, got empty sequence.")
    if len(taus) != len(weights):
        raise ValueError(
            "ties_blend_deltas expected len(taus) == len(weights), got "
            f"len(taus)={len(taus)}, len(weights)={len(weights)}."
        )
    if not (0.0 < density <= 1.0):
        raise ValueError(f"ties_density must be in (0, 1], got density={density}.")

    ref = taus[0]
    if not isinstance(ref, torch.Tensor):
        raise TypeError(f"ties_blend_deltas expected torch.Tensor, got {type(ref).__name__}.")
    if ref.ndim < 1:
        raise ValueError(
            f"ties_blend_deltas expected batch dimension (ndim >= 1), got ndim={ref.ndim}."
        )
    ref_shape = ref.shape
    for idx, tau in enumerate(taus):
        if not isinstance(tau, torch.Tensor):
            raise TypeError(
                f"ties_blend_deltas expected torch.Tensor at index {idx}, "
                f"got {type(tau).__name__}."
            )
        if tau.shape != ref_shape:
            raise ValueError(
                f"ties_blend_deltas expected all tensors to share shape "
                f"{tuple(ref_shape)}, got index {idx} shape {tuple(tau.shape)}."
            )

    batch = ref_shape[0]
    task_list = list(taus)

    # 1. Optional magnitude trim (per sample, per checkpoint).
    if density < 1.0:
        trimmed: List[torch.Tensor] = []
        for tau in task_list:
            flat = tau.reshape(batch, -1)
            num_elems = flat.shape[1]
            keep = max(1, int(math.ceil(density * num_elems)))
            if keep >= num_elems:
                trimmed.append(tau)
                continue
            kth = torch.topk(flat.abs(), keep, dim=1).values[:, -1:]
            mask = flat.abs() >= kth
            trimmed.append((flat * mask).reshape(ref_shape))
        task_list = trimmed

    # 2. Elect a per-element sign from the weighted sum.
    weighted_sum = task_list[0] * weights[0]
    for k in range(1, len(task_list)):
        weighted_sum = weighted_sum + task_list[k] * weights[k]
    gamma = torch.sign(weighted_sum)

    # 3. Disjoint weighted mean over sign-agreeing checkpoints.
    numerator = torch.zeros_like(ref)
    denom = torch.zeros_like(ref)
    total_elements = 0
    disagree_elements = 0
    for tau, weight in zip(task_list, weights, strict=True):
        nonzero = tau != 0
        agree = (torch.sign(tau) == gamma) & nonzero
        agree_f = agree.to(ref.dtype)
        numerator = numerator + weight * tau * agree_f
        denom = denom + weight * agree_f
        if stats is not None:
            total_elements += int(nonzero.sum().item())
            disagree_elements += int((nonzero & ~agree).sum().item())

    tiny = torch.finfo(ref.dtype).tiny
    tau_merged = torch.where(
        denom > 0, numerator / denom.clamp_min(tiny), torch.zeros_like(ref)
    )

    if stats is not None:
        if stats.num_steps == 0:
            stats.tensor_shape = tuple(ref_shape)
        stats.record_step(
            step_total_elements=total_elements,
            step_disagree_elements=disagree_elements,
        )

    return tau_merged


def _ties_blend(
    adapter: "BaseAdapter",
    checkpoint_names: Sequence[str],
    weights: Sequence[float],
    noise_only_kwargs: Dict[str, Any],
    base_forward: Callable[..., Any],
    density: float,
    stats: Optional[TIESStats] = None,
) -> torch.Tensor:
    """TIES-merging fusion of checkpoint noise_pred via base-anchored sign vote.

    Runs the pretrained baseline forward (``v_b``) and each checkpoint forward,
    forms task vectors ``tau_i = noise_pred_i - v_b``, merges them with
    :func:`ties_blend_deltas` (sign election + disjoint weighted mean), and
    returns ``v_b + tau_merged``.

    Note:
        This adds one extra forward pass per denoising step for the pretrained model.
    """
    with torch.no_grad(), adapter.use_ref_parameters():
        ref_out = base_forward(**noise_only_kwargs)
    if ref_out.noise_pred is None:
        raise RuntimeError(
            "Pretrained model forward did not return `noise_pred` in TIES mode; "
            "check that the adapter supports return_kwargs=['noise_pred']."
        )
    ref_noise_pred = ref_out.noise_pred

    taus: List[torch.Tensor] = []
    for name in checkpoint_names:
        with adapter.use_named_parameters(name):
            out = base_forward(**noise_only_kwargs)
        if out.noise_pred is None:
            raise RuntimeError(
                f"Checkpoint '{name}' forward did not return `noise_pred`; "
                "check that the adapter supports return_kwargs=['noise_pred']."
            )
        taus.append(out.noise_pred - ref_noise_pred)

    tau_merged = ties_blend_deltas(taus, weights, density=density, stats=stats)
    return ref_noise_pred + tau_merged


def ensemble_forward_step(
    adapter: "BaseAdapter",
    checkpoint_names: Sequence[str],
    weights: Sequence[float],
    forward_kwargs: Dict[str, Any],
    sched_cache: SchedulerStepCache,
    base_forward: Callable[..., Any],
    blend_mode: EnsembleBlendMode = "weighted",
    pcgrad_eps: float = 1e-8,
    pcgrad_generator: Optional[torch.Generator] = None,
    ties_density: float = 1.0,
    stats: Optional[Union[PCGradStats, TIESStats]] = None,
) -> Any:
    """Blend per-checkpoint ``noise_pred`` tensors, then run one scheduler step.

    For each snapshot, calls ``base_forward`` (the unpatched ``adapter.forward``)
    under :meth:`BaseAdapter.use_named_parameters` with
    ``return_kwargs=['noise_pred']``. The blended prediction is passed to a single
    ``adapter.scheduler.step`` call.

    Args:
        adapter: Model adapter whose ``scheduler`` is used for the final step.
        checkpoint_names: Snapshot names from :func:`load_checkpoints`.
        weights: Normalized weights (same length as ``checkpoint_names``).
        forward_kwargs: Keyword arguments passed to ``base_forward``.
        sched_cache: Cached signature from :func:`cache_scheduler_step_signature`.
        base_forward: Original ``adapter.forward`` before any ensemble patch; must
            not re-enter :func:`ensemble_forward_step`.
        blend_mode: Fusion strategy. The PCGrad family is the cross product of
            ``vector in {full velocity, residual delta from base}`` and
            ``projection in {global, channelwise, normalized}``:
            ``'weighted'``: linear blend ``sum_i w_i * noise_pred_i``.
            ``'pcgrad'`` / ``'pcgrad_channelwise'`` / ``'pcgrad_normalized'``:
            full-velocity PCGrad with global / per-channel / magnitude-normalized
            projection.
            ``'pcgrad_residual'`` / ``'pcgrad_residual_channelwise'`` /
            ``'pcgrad_residual_normalized'``: same three projections on the
            task-specific deltas from the pretrained model (one extra forward
            per step).
            ``'ties'``: TIES-merging base-anchored per-element sign vote (one
            extra forward per step).
        pcgrad_eps: Epsilon for PCGrad denominator (any pcgrad mode).
        pcgrad_generator: Optional RNG for PCGrad inner-loop shuffle.
        ties_density: Fraction of largest-magnitude entries kept per task in
            ``'ties'`` mode (``1.0`` = no trim).
        stats: Optional :class:`PCGradStats` (pcgrad modes) or :class:`TIESStats`
            (``'ties'`` mode) accumulator for deferred logging.

    Returns:
        Scheduler step output (same type as ``adapter.forward``).

    Raises:
        ValueError: Mismatched lengths, invalid ``blend_mode``, or empty checkpoints.
        RuntimeError: A checkpoint forward did not return ``noise_pred``.
    """
    if blend_mode not in ENSEMBLE_BLEND_MODES:
        raise ValueError(
            f"ensemble_forward_step expected blend_mode in {ENSEMBLE_BLEND_MODES}, "
            f"got blend_mode={blend_mode!r}."
        )
    if len(checkpoint_names) != len(weights):
        raise ValueError(
            f"checkpoint_names and weights must have the same length, got "
            f"len(checkpoint_names)={len(checkpoint_names)}, len(weights)={len(weights)}."
        )
    if not checkpoint_names:
        raise ValueError("ensemble_forward_step requires at least one checkpoint.")

    noise_only_kwargs = dict(forward_kwargs)
    noise_only_kwargs["return_kwargs"] = ["noise_pred"]

    if blend_mode in _RESIDUAL_PROJECTIONS:
        combined_noise_pred = _pcgrad_residual_blend(
            adapter=adapter,
            checkpoint_names=checkpoint_names,
            weights=weights,
            noise_only_kwargs=noise_only_kwargs,
            base_forward=base_forward,
            pcgrad_eps=pcgrad_eps,
            pcgrad_generator=pcgrad_generator,
            projection=_RESIDUAL_PROJECTIONS[blend_mode],
            stats=stats,
        )
    elif blend_mode == "ties":
        combined_noise_pred = _ties_blend(
            adapter=adapter,
            checkpoint_names=checkpoint_names,
            weights=weights,
            noise_only_kwargs=noise_only_kwargs,
            base_forward=base_forward,
            density=ties_density,
            stats=stats,
        )
    else:
        # Full-velocity blends: weighted, pcgrad, pcgrad_channelwise, pcgrad_normalized.
        raw_preds: List[torch.Tensor] = []
        for name in checkpoint_names:
            with adapter.use_named_parameters(name):
                out = base_forward(**noise_only_kwargs)
            if out.noise_pred is None:
                raise RuntimeError(
                    f"Checkpoint '{name}' forward did not return `noise_pred`; "
                    "check that the adapter supports return_kwargs=['noise_pred']."
                )
            raw_preds.append(out.noise_pred)

        if blend_mode == "weighted":
            combined_noise_pred = torch.stack(
                [pred * weight for pred, weight in zip(raw_preds, weights, strict=True)],
                dim=0,
            ).sum(dim=0)
        else:
            combined_noise_pred = _blend_velocity_set(
                raw_preds,
                weights,
                projection=_FULL_PROJECTIONS[blend_mode],
                eps=pcgrad_eps,
                generator=pcgrad_generator,
                stats=stats,
            )

    scheduler_kwargs = _build_scheduler_step_kwargs(
        forward_kwargs, combined_noise_pred, sched_cache
    )
    return adapter.scheduler.step(**scheduler_kwargs)
