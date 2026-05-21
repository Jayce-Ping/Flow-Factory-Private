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

# src/flow_factory/trainers/opd/common.py
"""Shared helpers for the OPD trainer family.

Module-level pure functions used by :class:`OPDTrainer` (SDE/ODE) and
:class:`DiffusionOPDTrainer`. Handles teacher administration, forward-kwarg
plumbing, and timestep preparation.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Tuple

import torch

from ...scheduler import set_scheduler_timesteps
from ...utils.logger_utils import setup_logger
from ...utils.lora_loader import load_lora_as_named_parameters

if TYPE_CHECKING:
    from ...models.abc import BaseAdapter

logger = setup_logger(__name__)


def load_teachers(
    adapter: "BaseAdapter",
    teacher_paths: List[str],
    teacher_param_device: str,
) -> List[str]:
    """Load each teacher LoRA checkpoint into a named-parameter snapshot.

    Args:
        adapter: Active ``BaseAdapter`` instance (must already be in LoRA
            mode with the student adapter attached).
        teacher_paths: List of teacher LoRA checkpoint paths or HF Hub repo
            ids (whatever :func:`load_lora_as_named_parameters` accepts).
            Must contain at least one entry.
        teacher_param_device: ``'cpu'`` or ``'cuda'``; passed verbatim to
            :func:`load_lora_as_named_parameters`. ``'cuda'`` keeps each
            snapshot on-device for fast swaps at the cost of LoRA-sized VRAM
            per teacher.

    Returns:
        Ordered list of snapshot names ``['opd_teacher_0', 'opd_teacher_1', ...]``
        in the same order as ``teacher_paths``. Use as the lookup keys for
        :meth:`BaseAdapter.use_named_parameters`.

    Raises:
        ValueError: ``teacher_paths`` is empty.
    """
    if not teacher_paths:
        raise ValueError(
            "OPD requires at least one teacher LoRA path; " f"got teacher_paths={teacher_paths!r}."
        )

    teacher_names: List[str] = []
    for i, path in enumerate(teacher_paths):
        name = f"opd_teacher_{i}"
        load_lora_as_named_parameters(
            adapter=adapter,
            name=name,
            lora_path=path,
            device=teacher_param_device,
        )
        teacher_names.append(name)
    logger.info(
        f"Loaded {len(teacher_names)} OPD teacher(s): {teacher_names} "
        f"(device={teacher_param_device!r})."
    )
    return teacher_names


def teacher_indices_for_batch(
    *,
    teacher_aggregation: str,
    num_teachers: int,
    epoch: int,
    inner_epoch: int,
    batch_idx: int,
    num_inner: int,
    num_batches: int,
) -> List[int]:
    """Pick which teacher(s) to evaluate for one micro-batch.

    Four strategies:

    - ``'round_robin'``: a single teacher, cycling across ``(epoch,
      inner_epoch, batch_idx)``. The schedule folds ``inner_epoch`` in so
      different inner epochs of the same outer epoch use different
      teachers (otherwise inner-epoch 0 and inner-epoch 1 would always
      pick the same teacher for the same ``batch_idx``).
    - ``'average'``: all teachers; the caller is expected to forward each
      and average their predictions.
    - ``'sum'``: all teachers; compute per-teacher losses and sum gradients
      directly (no conflict resolution; PCGrad ablation baseline).
    - ``'pcgrad'``: all teachers; compute per-teacher losses and apply
      PCGrad (Projected Gradient Descent) to resolve conflicts.

    Args:
        teacher_aggregation: ``'round_robin'``, ``'average'``, ``'sum'``, or ``'pcgrad'``.
        num_teachers: ``len(self._teacher_names)`` -- pre-computed.
        epoch: Current outer epoch (``self.epoch``).
        inner_epoch: Current inner-epoch index within :meth:`optimize`.
        batch_idx: Current micro-batch index within the inner epoch.
        num_inner: ``training_args.num_inner_epochs``.
        num_batches: ``training_args.num_batches_per_epoch``.

    Returns:
        Sorted list of teacher indices to use this micro-batch.

    Raises:
        ValueError: unknown ``teacher_aggregation`` value.
    """
    if teacher_aggregation == "round_robin":
        global_batch = (epoch * num_inner + inner_epoch) * num_batches + batch_idx
        return [global_batch % num_teachers]
    if teacher_aggregation == "average":
        return list(range(num_teachers))
    if teacher_aggregation in ("sum", "pcgrad"):
        return list(range(num_teachers))
    raise ValueError(
        f"Unknown teacher_aggregation={teacher_aggregation!r}; "
        "expected 'round_robin', 'average', 'sum', or 'pcgrad'."
    )


def cache_forward_signature(
    forward_fn: Callable[..., Any],
) -> Tuple[FrozenSet[str], bool]:
    """Snapshot ``inspect.signature(forward_fn)`` so per-timestep filtering is cheap.

    The per-step ``adapter.forward`` call runs O(num_train_timesteps *
    num_batches * num_inner_epochs) times per epoch -- naive
    :func:`filter_kwargs` would re-introspect the signature every time.
    Cache the parameter-name allow-list (or the var-kwargs flag) once at
    trainer ``__init__`` and pass to :func:`filter_forward_kwargs` later.

    Args:
        forward_fn: Bound method, typically ``self.adapter.forward``.

    Returns:
        ``(param_names, accepts_var_kwargs)``: the set of declared parameter
        names (excluding ``self``, which is not in the bound signature) and
        a flag that is ``True`` when ``forward_fn`` accepts ``**kwargs``.
    """
    sig = inspect.signature(forward_fn)
    param_names: FrozenSet[str] = frozenset(sig.parameters.keys())
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    return param_names, accepts_var_kwargs


def filter_forward_kwargs(
    full_kwargs: Dict[str, Any],
    param_names: FrozenSet[str],
    accepts_var_kwargs: bool,
) -> Dict[str, Any]:
    """Cheap per-call kwarg filter using the cache from :func:`cache_forward_signature`.

    When ``forward_fn`` accepts ``**kwargs`` the full dict is returned as-is
    (no work); otherwise a fresh dict is built containing only entries whose
    keys are in ``param_names``. Does NOT mutate ``full_kwargs``.

    Args:
        full_kwargs: Raw kwarg dict to filter.
        param_names: Allow-list from :func:`cache_forward_signature`.
        accepts_var_kwargs: Bypass flag from :func:`cache_forward_signature`.

    Returns:
        Filtered dict; same instance as ``full_kwargs`` iff
        ``accepts_var_kwargs`` is ``True``.
    """
    if accepts_var_kwargs:
        return full_kwargs
    return {k: v for k, v in full_kwargs.items() if k in param_names}


def pcgrad_project_gradients(
    per_teacher_grads: List[List[torch.Tensor]],
    eps: float = 1e-8,
) -> List[torch.Tensor]:
    """Apply PCGrad (Projected Gradient Descent) to per-teacher gradient lists.

    For each pair of teachers (i, j): if grad_i · grad_j < 0 (conflicting),
    subtract the projection of grad_i onto grad_j from grad_i. This ensures
    grad_i never moves against grad_j's direction.

    Args:
        per_teacher_grads: K lists, each containing one gradient tensor per
            trainable parameter (same order across all K teachers).
            Shape: [K][num_params], where K = num_teachers and num_params
            is the count of trainable parameters.
        eps: Minimum squared norm for projection denominator clamping.
            Prevents division by near-zero gradients. Default 1e-8.

    Returns:
        Single list of tensors (one per parameter) = sum of all K projected
        gradient sets. Shape: [num_params], same structure as per_teacher_grads[0].

    Example:
        >>> grad_teacher_0 = [g0_p0, g0_p1, ...]
        >>> grad_teacher_1 = [g1_p0, g1_p1, ...]
        >>> projected = pcgrad_project_gradients([grad_teacher_0, grad_teacher_1])
        >>> # projected[0] = projected_g0_p0 + projected_g1_p0
    """
    K = len(per_teacher_grads)
    if K == 1:
        # Single teacher: no conflicts to resolve
        return per_teacher_grads[0]

    # Flatten each teacher's gradient list into a single vector for dot products
    flat_grads = [torch.cat([g.flatten() for g in grads]) for grads in per_teacher_grads]

    # Pre-cache squared norms to avoid redundant O(K²) recomputation
    norm_sq_cache = [(g * g).sum().clamp(min=eps) for g in flat_grads]

    # PCGrad projection: iteratively remove conflicts
    pc = [g.clone() for g in flat_grads]
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            dot = (pc[i] * flat_grads[j]).sum()
            if dot < 0:
                pc[i] = pc[i] - (dot / norm_sq_cache[j]) * flat_grads[j]

    # Sum projected gradients across all teachers
    combined_flat = sum(pc)

    # Unflatten back to per-parameter tensor structure
    result = []
    offset = 0
    for g in per_teacher_grads[0]:  # Use first teacher's shapes as template
        numel = g.numel()
        result.append(combined_flat[offset : offset + numel].view_as(g))
        offset += numel

    return result


def prepare_train_timesteps(
    scheduler,
    *,
    num_inference_steps: int,
    height: int,
    width: int,
    patch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """SD3.5-compatible noise schedule for ODE Euler training.

    Mirrors ``SD3_5Adapter.inference`` step 5: ``image_seq_len`` from latent
    spatial size, then ``set_scheduler_timesteps`` (shift + retrieve_timesteps).
    Returns scheduler-scale timesteps in inference order (noisy → clean).
    """
    latent_h = height // 8
    latent_w = width // 8
    image_seq_len = (latent_h // patch_size) * (latent_w // patch_size)
    return set_scheduler_timesteps(
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
        seq_len=image_seq_len,
        device=device,
    )
