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

import random
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Literal, Optional, Sequence, Tuple, get_args

import torch

from ...utils.logger_utils import setup_logger
from ...utils.lora_loader import load_lora_as_named_parameters
from ..opd.common import cache_forward_signature, filter_forward_kwargs

if TYPE_CHECKING:
    from ...models.abc import BaseAdapter

logger = setup_logger(__name__)

SchedulerStepCache = Tuple[FrozenSet[str], bool]

EnsembleBlendMode = Literal["weighted", "pcgrad"]
ENSEMBLE_BLEND_MODES: Tuple[str, ...] = get_args(EnsembleBlendMode)


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

    for i in range(num_checkpoints):
        for j in _shuffled_other_indices(num_checkpoints, i, generator):
            flat_pc_i = pc[i].reshape(batch, -1)
            dot = (flat_pc_i * flat_orig[j]).sum(dim=1).view(broadcast_shape)
            coeff = dot / norm_sq_orig[j]
            proj = coeff * scaled_preds[j]
            pc[i] = torch.where(dot < 0, pc[i] - proj, pc[i])

    return torch.stack(pc, dim=0).sum(dim=0)


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
        blend_mode: ``'weighted'`` for linear blend, ``'pcgrad'`` for PCGrad fusion.
        pcgrad_eps: Epsilon for PCGrad denominator when ``blend_mode='pcgrad'``.
        pcgrad_generator: Optional RNG for PCGrad inner-loop shuffle.

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

    raw_preds: List[torch.Tensor] = []
    scaled_preds: List[torch.Tensor] = []
    for name, weight in zip(checkpoint_names, weights, strict=True):
        with adapter.use_named_parameters(name):
            out = base_forward(**noise_only_kwargs)
        if out.noise_pred is None:
            raise RuntimeError(
                f"Checkpoint '{name}' forward did not return `noise_pred`; "
                "check that the adapter supports return_kwargs=['noise_pred']."
            )
        raw_preds.append(out.noise_pred)
        scaled_preds.append(out.noise_pred * weight)

    # Sanity check: if all raw noise_preds are identical, the parameter swap
    # is likely not taking effect (common cause: load_adapter silently failed
    # or parameter objects diverged from model forward path).
    if len(raw_preds) > 1:
        ref = raw_preds[0]
        all_same = all(torch.equal(ref, p) for p in raw_preds[1:])
        if all_same:
            logger.warning(
                "ensemble_forward_step: ALL checkpoint noise_preds are numerically "
                "identical! This means use_named_parameters is NOT effectively "
                "swapping model weights. Check that load_lora_as_named_parameters "
                "actually loaded distinct weights for each checkpoint."
            )

    if blend_mode == "weighted":
        combined_noise_pred = torch.stack(scaled_preds, dim=0).sum(dim=0)
    else:
        combined_noise_pred = pcgrad_blend_noise_preds(
            scaled_preds,
            eps=pcgrad_eps,
            generator=pcgrad_generator,
        )

    scheduler_kwargs = _build_scheduler_step_kwargs(
        forward_kwargs, combined_noise_pred, sched_cache
    )
    return adapter.scheduler.step(**scheduler_kwargs)
