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

from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch

from ...utils.logger_utils import setup_logger
from ...utils.lora_loader import load_lora_as_named_parameters
from ..opd.common import cache_forward_signature, filter_forward_kwargs

if TYPE_CHECKING:
    from ...models.abc import BaseAdapter

logger = setup_logger(__name__)

SchedulerStepCache = Tuple[FrozenSet[str], bool]


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


def ensemble_forward_step(
    adapter: "BaseAdapter",
    checkpoint_names: Sequence[str],
    weights: Sequence[float],
    forward_kwargs: Dict[str, Any],
    sched_cache: SchedulerStepCache,
) -> Any:
    """Blend per-checkpoint ``noise_pred`` tensors, then run one scheduler step.

    For each snapshot, calls ``adapter.forward`` under
    :meth:`BaseAdapter.use_named_parameters` with ``return_kwargs=['noise_pred']``.
    The blended prediction is passed to a single ``adapter.scheduler.step`` call.

    Args:
        adapter: Model adapter whose ``forward`` / ``scheduler`` are used.
        checkpoint_names: Snapshot names from :func:`load_checkpoints`.
        weights: Normalized weights (same length as ``checkpoint_names``).
        forward_kwargs: Keyword arguments passed to ``adapter.forward``.
        sched_cache: Cached signature from :func:`cache_scheduler_step_signature`.

    Returns:
        Scheduler step output (same type as ``adapter.forward``).

    Raises:
        ValueError: Mismatched ``checkpoint_names`` / ``weights`` lengths.
        RuntimeError: A checkpoint forward did not return ``noise_pred``.
    """
    if len(checkpoint_names) != len(weights):
        raise ValueError(
            f"checkpoint_names and weights must have the same length, got "
            f"len(checkpoint_names)={len(checkpoint_names)}, len(weights)={len(weights)}."
        )
    if not checkpoint_names:
        raise ValueError("ensemble_forward_step requires at least one checkpoint.")

    noise_only_kwargs = dict(forward_kwargs)
    noise_only_kwargs["return_kwargs"] = ["noise_pred"]

    combined_noise_pred: Optional[torch.Tensor] = None
    for name, weight in zip(checkpoint_names, weights, strict=True):
        with adapter.use_named_parameters(name):
            out = adapter.forward(**noise_only_kwargs)
        if out.noise_pred is None:
            raise RuntimeError(
                f"Checkpoint '{name}' forward did not return `noise_pred`; "
                "check that the adapter supports return_kwargs=['noise_pred']."
            )
        term = out.noise_pred * weight
        combined_noise_pred = term if combined_noise_pred is None else combined_noise_pred + term

    if combined_noise_pred is None:
        raise RuntimeError("ensemble_forward_step produced no blended noise_pred.")

    scheduler_kwargs = _build_scheduler_step_kwargs(
        forward_kwargs, combined_noise_pred, sched_cache
    )
    return adapter.scheduler.step(**scheduler_kwargs)
