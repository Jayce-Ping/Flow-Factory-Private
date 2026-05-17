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
"""Shared helpers for the OPD trainer family (``sde.py`` and ``ode.py``).

These are module-level pure functions to keep the two trainers DRY without
introducing class-inheritance coupling (constraint #11 -- flat trainer
hierarchy). Both :class:`flow_factory.trainers.opd.sde.OPDTrainer` (SDE) and
:class:`flow_factory.trainers.opd.ode.OPDODETrainer` (ODE) own their own
algorithm-specific machinery and only delegate teacher administration and
``adapter.forward`` kwarg plumbing here.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Tuple

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

    Two strategies:

    - ``'round_robin'``: a single teacher, cycling across ``(epoch,
      inner_epoch, batch_idx)``. The schedule folds ``inner_epoch`` in so
      different inner epochs of the same outer epoch use different
      teachers (otherwise inner-epoch 0 and inner-epoch 1 would always
      pick the same teacher for the same ``batch_idx``).
    - ``'average'``: all teachers; the caller is expected to forward each
      and average their predictions.

    Args:
        teacher_aggregation: ``'round_robin'`` or ``'average'``.
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
    raise ValueError(
        f"Unknown teacher_aggregation={teacher_aggregation!r}; "
        "expected 'round_robin' or 'average'."
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
