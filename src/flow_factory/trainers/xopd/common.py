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

# src/flow_factory/trainers/xopd/common.py
"""Pure, stateless helpers for the Cross-OPD (XOPD) trainer.

XOPD is a standalone trainer (decoupled from the OPD / MoF trainer families).
These functions are copied/adapted from :mod:`flow_factory.trainers.opd` so the
XOPD package does not import OPD internals; see the OPD originals for the full
derivations (Flow-OPD paper, Appendix B for the Gaussian transition KL).
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Generator, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class POPDResponsibility:
    """Detached Gaussian-mixture responsibility statistics for one transition batch."""

    log_ratio_sum: torch.Tensor
    log_ratio_per_dim: torch.Tensor
    tempered_log_ratio: torch.Tensor
    gate_logit: torch.Tensor
    teacher_responsibility: torch.Tensor
    teacher_old_kl_joint: torch.Tensor
    teacher_old_kl_per_dim: torch.Tensor
    event_dim: int
    alpha: float
    temperature: float


@dataclass(frozen=True)
class POPDBehaviorTransition:
    """Cached behavior-policy statistics aligned to one rollout timestep."""

    mu_old: torch.Tensor
    std_dev_t: torch.Tensor
    dt: torch.Tensor


def extract_popd_behavior_transition(
    batch: Dict[str, Any],
    *,
    timestep_index: int,
) -> POPDBehaviorTransition:
    """Extract cached behavior mean and scheduler scales for one trajectory step.

    Args:
        batch: Stacked rollout sample containing callback tensors and their index map.
        timestep_index: Original denoising-loop step index.

    Returns:
        Detached behavior transition statistics aligned to ``timestep_index``.
    """
    if not isinstance(batch, dict):
        raise TypeError(f"expected dict for batch, got {type(batch).__name__}: {batch!r}")
    if not isinstance(timestep_index, int) or timestep_index < 0:
        raise ValueError(
            f"expected non-negative int timestep_index, got timestep_index={timestep_index!r}."
        )
    required = ("next_latents_mean", "std_dev_t", "dt", "callback_index_map")
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(
            "P-OPD rollout is missing required behavior callback fields "
            f"{missing!r}; available keys={sorted(batch.keys())!r}. "
            "Ensure sample() requests next_latents_mean, std_dev_t, and dt."
        )

    callback_map = batch["callback_index_map"]
    if not isinstance(callback_map, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor for callback_index_map, "
            f"got {type(callback_map).__name__}: {callback_map!r}"
        )
    if callback_map.ndim == 2:
        if callback_map.shape[0] < 1:
            raise ValueError(
                "expected callback_index_map with a non-empty batch dimension, "
                f"got shape={tuple(callback_map.shape)}."
            )
        first_map = callback_map[0]
        if not torch.equal(callback_map, first_map.unsqueeze(0).expand_as(callback_map)):
            raise ValueError(
                "expected every sample to share the same callback_index_map, "
                f"got shape={tuple(callback_map.shape)} and values={callback_map.cpu().tolist()}."
            )
        callback_map = first_map
    elif callback_map.ndim != 1:
        raise ValueError(
            "expected callback_index_map shape (T,) or (B,T), "
            f"got shape={tuple(callback_map.shape)}."
        )
    if timestep_index >= callback_map.shape[0]:
        raise ValueError(
            f"timestep_index={timestep_index} exceeds callback_index_map length "
            f"{callback_map.shape[0]}."
        )
    compact_index = int(callback_map[timestep_index].item())
    if compact_index < 0:
        raise ValueError(
            f"P-OPD behavior callbacks were not collected for timestep_index={timestep_index}; "
            f"callback_index_map value={compact_index}."
        )

    selected: Dict[str, torch.Tensor] = {}
    batch_size: Optional[int] = None
    for name in ("next_latents_mean", "std_dev_t", "dt"):
        value = batch[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"expected torch.Tensor for cached {name}, got {type(value).__name__}: {value!r}"
            )
        if value.ndim < 2:
            raise ValueError(
                f"expected cached {name} shape (B,T,...), got shape={tuple(value.shape)}."
            )
        if batch_size is None:
            batch_size = int(value.shape[0])
        elif value.shape[0] != batch_size:
            raise ValueError(
                f"expected cached {name} batch size {batch_size}, got shape={tuple(value.shape)}."
            )
        if compact_index >= value.shape[1]:
            raise ValueError(
                f"callback compact index {compact_index} exceeds cached {name} timestep "
                f"dimension {value.shape[1]} for shape={tuple(value.shape)}."
            )
        selected[name] = value[:, compact_index].detach()

    return POPDBehaviorTransition(
        mu_old=selected["next_latents_mean"],
        std_dev_t=selected["std_dev_t"],
        dt=selected["dt"],
    )


def compute_popd_quantiles(values: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute detached P-OPD p01/p10/p50/p90/p99 statistics.

    Args:
        values: Non-empty finite tensor containing globally gathered sample values.

    Returns:
        Mapping from percentile name to detached scalar tensor.
    """
    if not isinstance(values, torch.Tensor):
        raise TypeError(
            f"expected torch.Tensor for P-OPD quantiles, got {type(values).__name__}: {values!r}"
        )
    flattened = values.detach().float().flatten()
    if flattened.numel() < 1:
        raise ValueError(
            f"expected at least one value for P-OPD quantiles, got shape={tuple(values.shape)}."
        )
    if not torch.isfinite(flattened).all():
        raise ValueError(
            f"expected finite values for P-OPD quantiles, got values={flattened.cpu().tolist()}."
        )
    probabilities = torch.tensor(
        [0.01, 0.10, 0.50, 0.90, 0.99],
        device=flattened.device,
        dtype=flattened.dtype,
    )
    quantiles = torch.quantile(flattened, probabilities)
    return {
        name: value.detach() for name, value in zip(("p01", "p10", "p50", "p90", "p99"), quantiles)
    }


def validate_popd_configuration(
    *,
    target_mode: str,
    dynamics_type: str,
    noise_level: float,
    xopd_dk_space: str,
    normalize_d_k: bool,
    is_cross_vae: bool,
    pixel_loss: bool,
) -> None:
    """Validate that an XOPD configuration satisfies the P-OPD probability assumptions.

    Args:
        target_mode: XOPD target mode, either ``"direct"`` or ``"p_opd"``.
        dynamics_type: Scheduler dynamics used for rollout and replay.
        noise_level: Positive scheduler noise scale used by stochastic transitions.
        xopd_dk_space: Distillation loss space.
        normalize_d_k: Whether transition-mean error is covariance-normalized.
        is_cross_vae: Whether teacher and student use different latent spaces.
        pixel_loss: Whether the L1 target is decoded to pixel space.
    """
    if target_mode == "direct":
        return
    if target_mode != "p_opd":
        raise ValueError(
            f"P-OPD target mode must be 'direct' or 'p_opd', got target_mode={target_mode!r}."
        )
    supported_dynamics = ("Flow-SDE", "Dance-SDE", "CPS")
    if dynamics_type not in supported_dynamics:
        raise ValueError(
            "P-OPD requires a stochastic Gaussian transition with dynamics_type in "
            f"{supported_dynamics!r}, got dynamics_type={dynamics_type!r}."
        )
    if (
        isinstance(noise_level, bool)
        or not isinstance(noise_level, (int, float))
        or not math.isfinite(float(noise_level))
        or float(noise_level) <= 0.0
    ):
        raise ValueError(
            "P-OPD requires a finite scheduler noise_level > 0, "
            f"got noise_level={noise_level!r}, dynamics_type={dynamics_type!r}."
        )
    if xopd_dk_space != "xt":
        raise ValueError(
            "P-OPD local mixture-KL surrogate requires xopd_dk_space='xt', "
            f"got xopd_dk_space={xopd_dk_space!r}."
        )
    if normalize_d_k is not True:
        raise ValueError(
            "P-OPD local mixture-KL surrogate requires normalize_d_k=True, "
            f"got normalize_d_k={normalize_d_k!r}."
        )
    if is_cross_vae:
        raise ValueError(
            "P-OPD currently requires a shared latent space and identity transport, "
            f"got is_cross_vae={is_cross_vae!r}."
        )
    if pixel_loss:
        raise ValueError(
            "P-OPD is defined on Gaussian latent transitions and does not support "
            f"pixel loss, got pixel_loss={pixel_loss!r}."
        )


def _batch_scalar(
    value: torch.Tensor,
    *,
    name: str,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Convert a scheduler batch-scalar tensor to shape ``(B,)`` with strict validation."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected torch.Tensor for {name}, got {type(value).__name__}: {value!r}")
    if value.ndim < 1:
        raise ValueError(
            f"expected {name} to include a batch dimension, got shape={tuple(value.shape)}"
        )
    actual_batch_size = int(value.shape[0])
    if batch_size is not None and actual_batch_size != batch_size:
        raise ValueError(
            f"expected {name} batch size {batch_size}, got shape={tuple(value.shape)}."
        )
    flattened = value.float().reshape(actual_batch_size, -1)
    if flattened.shape[1] != 1:
        raise ValueError(
            f"expected one scalar per sample for {name}, got shape={tuple(value.shape)} "
            f"with {flattened.shape[1]} values per sample."
        )
    flattened = flattened[:, 0]
    if not torch.isfinite(flattened).all():
        raise ValueError(
            f"expected finite values for {name}, got shape={tuple(value.shape)} "
            f"and values={flattened.detach().cpu().tolist()}."
        )
    return flattened


def _broadcast_batch_scalar(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast a validated ``(B,)`` tensor over a reference tensor's event dimensions."""
    return value.reshape(value.shape[0], *([1] * (reference.ndim - 1)))


def _validate_same_event_shape(
    reference: torch.Tensor,
    tensors: Dict[str, torch.Tensor],
    *,
    reference_name: str,
) -> None:
    """Validate floating event tensors share a non-empty ``(B, ...)`` shape."""
    if not isinstance(reference, torch.Tensor):
        raise TypeError(
            f"expected torch.Tensor for {reference_name}, "
            f"got {type(reference).__name__}: {reference!r}"
        )
    if reference.ndim < 2 or reference.shape[0] < 1:
        raise ValueError(
            f"expected {reference_name} shape (B, event...) with B >= 1, "
            f"got shape={tuple(reference.shape)}."
        )
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"expected torch.Tensor for {name}, got {type(tensor).__name__}: {tensor!r}"
            )
        if tensor.shape != reference.shape:
            raise ValueError(
                f"expected {name}.shape={tuple(reference.shape)} to match {reference_name}, "
                f"got {name}.shape={tuple(tensor.shape)}."
            )
        if not torch.isfinite(tensor.detach().float()).all():
            raise ValueError(
                f"expected finite {name}, got shape={tuple(tensor.shape)} "
                f"with abs_max={tensor.detach().float().abs().max().item()!r}."
            )
    if not torch.isfinite(reference.detach().float()).all():
        raise ValueError(
            f"expected finite {reference_name}, got shape={tuple(reference.shape)} "
            f"with abs_max={reference.detach().float().abs().max().item()!r}."
        )


def compute_transition_variance(
    std_dev_t: torch.Tensor,
    dt: torch.Tensor,
    dynamics_type: str,
) -> torch.Tensor:
    """Compute the actual scalar Gaussian transition variance for each sample.

    Args:
        std_dev_t: Scheduler diffusion scale, one scalar per sample.
        dt: Scheduler timestep delta, one strictly negative scalar per sample.
        dynamics_type: One of ``Flow-SDE``, ``Dance-SDE``, or ``CPS``.

    Returns:
        A float32 tensor of shape ``(B,)`` containing strictly positive variances.
    """
    std = _batch_scalar(std_dev_t, name="std_dev_t")
    delta = _batch_scalar(dt, name="dt", batch_size=std.shape[0])
    if not (delta < 0).all():
        raise ValueError(
            "expected strictly negative dt for a denoising SDE transition, "
            f"got dynamics_type={dynamics_type!r}, dt={delta.detach().cpu().tolist()}."
        )
    if dynamics_type in ("Flow-SDE", "Dance-SDE"):
        variance = std.square() * (-delta)
    elif dynamics_type == "CPS":
        variance = std.square()
    elif dynamics_type == "ODE":
        raise ValueError(
            "P-OPD requires positive Gaussian transition variance, but "
            "dynamics_type='ODE' is deterministic."
        )
    else:
        raise ValueError(
            "expected dynamics_type in ('Flow-SDE', 'Dance-SDE', 'CPS'), "
            f"got dynamics_type={dynamics_type!r}."
        )
    if not torch.isfinite(variance).all() or not (variance > 0).all():
        raise ValueError(
            "expected finite positive transition variance for P-OPD, "
            f"got dynamics_type={dynamics_type!r}, variance={variance.detach().cpu().tolist()}, "
            f"std_dev_t={std.detach().cpu().tolist()}, dt={delta.detach().cpu().tolist()}."
        )
    return variance


def compute_popd_responsibility(
    *,
    next_latents: torch.Tensor,
    mu_old: torch.Tensor,
    mu_teacher: torch.Tensor,
    transition_variance: torch.Tensor,
    alpha: float,
    temperature: float,
) -> POPDResponsibility:
    """Compute detached P-OPD teacher responsibility from a behavior transition.

    Args:
        next_latents: Observed transition sampled by the behavior student.
        mu_old: Mean of that behavior-student transition.
        mu_teacher: Teacher transition mean at the same state.
        transition_variance: Shared scalar covariance value per sample.
        alpha: Prior teacher mixture probability in ``(0, 1)``.
        temperature: Positive temperature applied after the exact event-dimension sum.

    Returns:
        Detached density-ratio, posterior-gate, and teacher-gap statistics.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(f"expected numeric alpha in (0, 1), got {type(alpha).__name__}: {alpha!r}")
    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"expected alpha in (0, 1), got alpha={alpha!r}.")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError(
            "expected positive numeric temperature, "
            f"got {type(temperature).__name__}: {temperature!r}"
        )
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError(f"expected finite temperature > 0, got temperature={temperature!r}.")

    _validate_same_event_shape(
        next_latents,
        {"mu_old": mu_old, "mu_teacher": mu_teacher},
        reference_name="next_latents",
    )
    batch_size = int(next_latents.shape[0])
    variance = _batch_scalar(
        transition_variance,
        name="transition_variance",
        batch_size=batch_size,
    )
    if not (variance > 0).all():
        raise ValueError(
            "expected strictly positive transition_variance for P-OPD, "
            f"got values={variance.detach().cpu().tolist()}."
        )
    variance_b = _broadcast_batch_scalar(variance, next_latents)
    event_dim = math.prod(next_latents.shape[1:])
    with torch.no_grad():
        observed = next_latents.detach().float()
        old = mu_old.detach().float()
        teacher = mu_teacher.detach().float()
        old_sq = (observed - old).square()
        teacher_sq = (observed - teacher).square()
        log_ratio_sum = ((old_sq - teacher_sq) / (2.0 * variance_b)).flatten(1).sum(dim=1)
        teacher_old_kl_joint = ((teacher - old).square() / (2.0 * variance_b)).flatten(1).sum(dim=1)
        log_ratio_per_dim = log_ratio_sum / float(event_dim)
        tempered_log_ratio = log_ratio_sum / float(temperature)
        prior_logit = math.log(float(alpha)) - math.log1p(-float(alpha))
        gate_logit = tempered_log_ratio + prior_logit
        teacher_responsibility = torch.sigmoid(gate_logit)

    finite_outputs = {
        "log_ratio_sum": log_ratio_sum,
        "teacher_old_kl_joint": teacher_old_kl_joint,
        "gate_logit": gate_logit,
        "teacher_responsibility": teacher_responsibility,
    }
    for name, value in finite_outputs.items():
        if not torch.isfinite(value).all():
            raise ValueError(
                f"expected finite {name} for P-OPD, got values={value.detach().cpu().tolist()}, "
                f"event_dim={event_dim}, alpha={alpha!r}, temperature={temperature!r}, "
                f"variance_min={variance.min().item()!r}."
            )

    return POPDResponsibility(
        log_ratio_sum=log_ratio_sum,
        log_ratio_per_dim=log_ratio_per_dim,
        tempered_log_ratio=tempered_log_ratio,
        gate_logit=gate_logit,
        teacher_responsibility=teacher_responsibility,
        teacher_old_kl_joint=teacher_old_kl_joint,
        teacher_old_kl_per_dim=teacher_old_kl_joint / float(event_dim),
        event_dim=event_dim,
        alpha=float(alpha),
        temperature=float(temperature),
    )


def compute_popd_gaussian_mean_kl(
    mu_student: torch.Tensor,
    mu_teacher: torch.Tensor,
    transition_variance: torch.Tensor,
) -> torch.Tensor:
    """Compute per-dimension Gaussian mean KL with gradients only through the student.

    Args:
        mu_student: Current student transition mean.
        mu_teacher: Detached teacher transition mean.
        transition_variance: Shared scalar covariance value per sample.

    Returns:
        Per-sample mean KL tensor of shape ``(B,)``.
    """
    _validate_same_event_shape(
        mu_student,
        {"mu_teacher": mu_teacher},
        reference_name="mu_student",
    )
    variance = _batch_scalar(
        transition_variance,
        name="transition_variance",
        batch_size=mu_student.shape[0],
    )
    if not (variance > 0).all():
        raise ValueError(
            "expected strictly positive transition_variance for Gaussian mean KL, "
            f"got values={variance.detach().cpu().tolist()}."
        )
    variance_b = _broadcast_batch_scalar(variance, mu_student)
    diff = mu_student.float() - mu_teacher.detach().float()
    mean_kl = (diff.square() / (2.0 * variance_b)).flatten(1).mean(dim=1)
    if not torch.isfinite(mean_kl.detach()).all():
        raise ValueError(
            "expected finite P-OPD Gaussian mean KL, "
            f"got values={mean_kl.detach().cpu().tolist()}, "
            f"student_shape={tuple(mu_student.shape)}, variance_min={variance.min().item()!r}."
        )
    return mean_kl


def _event_rms(value: torch.Tensor) -> torch.Tensor:
    """Compute per-sample event RMS."""
    return value.float().square().flatten(1).mean(dim=1).sqrt()


def _event_l2(value: torch.Tensor) -> torch.Tensor:
    """Compute per-sample event L2 norm."""
    return value.float().square().flatten(1).sum(dim=1).sqrt()


def compute_popd_diagnostics(
    *,
    next_latents: torch.Tensor,
    mu_old: torch.Tensor,
    mu_teacher: torch.Tensor,
    mu_student: torch.Tensor,
    transition_variance: torch.Tensor,
    dt: torch.Tensor,
    responsibility: POPDResponsibility,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute detached per-sample P-OPD scale and sampler diagnostics.

    Every returned key is a per-sample tensor, which the reducer expands into four statistics
    and logs once globally and once per trained timestep. Thirty keys therefore became several
    hundred metrics per epoch, which is unreadable, so the default set is the ten that are
    actually load-bearing:

    * ``old_innovation_rms``, ``behavior_drift_rms`` -- the two self-validating checks. Cheap
      insurance that keeps a broken covariance from being mistaken for a saturated gate.
    * ``teacher_old_kl_joint``, ``teacher_old_gap_whitened_rms``, ``log_rho_sum`` -- the joint KL
      that drives the gate, its scale-free per-dimension form, and the ratio itself.
    * ``gamma``, ``gamma_lt_001``, ``gamma_gt_099`` -- the gate and how often it is pinned.
    * ``ungated_mean_kl``, ``gated_mean_kl`` -- the objective with and without the gate.

    What was dropped and why: constants already in the config (``alpha``, ``temperature``,
    ``event_dim``, ``temperature_over_event_dim``); exact restatements of another key
    (``teacher_old_kl_per_dim`` is the joint over D, ``effective_gate`` is ``gamma``,
    ``tempered_log_ratio`` and ``gate_logit`` are affine in ``log_rho_sum``, ``log_rho_per_dim``
    is ``log_rho_sum`` over D); scheduler quantities that never move within a run
    (``transition_std``, ``transition_variance``, ``abs_dt``); un-whitened or absolute magnitudes
    superseded by their whitened form (``teacher_old_gap_rms``, ``teacher_old_gap_l2``,
    ``next_latent_rms``, ``mu_*_rms``, ``teacher_innovation_rms``, ``teacher_pull_rms``); and
    ``student_teacher_gap_rms`` / ``student_teacher_gap_whitened_rms``, which are numerically
    IDENTICAL to their ``teacher_old`` counterparts here -- the diagnostics run during the replay
    pass at ``theta = theta_old``, where the student mean IS the behavior mean, so those keys can
    never show the student closing on the teacher and reading them as progress is a trap.

    Set ``verbose`` to restore the full set when debugging the sampler or the gate itself.

    Args:
        next_latents: Observed behavior transition.
        mu_old: Cached behavior transition mean.
        mu_teacher: Teacher transition mean.
        mu_student: Current gradient-pass student transition mean.
        transition_variance: Shared scalar covariance value per sample.
        dt: Scheduler timestep delta per sample.
        responsibility: Detached output from :func:`compute_popd_responsibility`.
        verbose: Emit every diagnostic instead of the essential subset.

    Returns:
        Mapping of metric names to detached finite tensors of shape ``(B,)``.
    """
    _validate_same_event_shape(
        next_latents,
        {
            "mu_old": mu_old,
            "mu_teacher": mu_teacher,
            "mu_student": mu_student,
        },
        reference_name="next_latents",
    )
    batch_size = int(next_latents.shape[0])
    variance = _batch_scalar(
        transition_variance,
        name="transition_variance",
        batch_size=batch_size,
    )
    if not (variance > 0).all():
        raise ValueError(
            "expected strictly positive transition_variance for P-OPD diagnostics, "
            f"got values={variance.detach().cpu().tolist()}."
        )
    delta = _batch_scalar(dt, name="dt", batch_size=batch_size)
    if responsibility.teacher_responsibility.shape != (batch_size,):
        raise ValueError(
            "expected responsibility.teacher_responsibility shape "
            f"({batch_size},), got {tuple(responsibility.teacher_responsibility.shape)}."
        )
    if responsibility.event_dim != math.prod(next_latents.shape[1:]):
        raise ValueError(
            "P-OPD responsibility event dimension does not match diagnostic tensors: "
            f"responsibility.event_dim={responsibility.event_dim}, "
            f"tensor_event_dim={math.prod(next_latents.shape[1:])}, "
            f"shape={tuple(next_latents.shape)}."
        )

    with torch.no_grad():
        observed = next_latents.detach().float()
        old = mu_old.detach().float()
        teacher = mu_teacher.detach().float()
        student = mu_student.detach().float()
        variance_b = _broadcast_batch_scalar(variance, observed)
        transition_std = variance.sqrt()
        std_b = _broadcast_batch_scalar(transition_std, observed)
        gamma = responsibility.teacher_responsibility.detach()
        gamma_b = _broadcast_batch_scalar(gamma, observed)

        teacher_old_gap = teacher - old
        student_teacher_gap = student - teacher
        behavior_drift = student - old
        ungated_mean_kl = (student_teacher_gap.square() / (2.0 * variance_b)).flatten(1).mean(dim=1)
        metrics = {
            "old_innovation_rms": _event_rms((observed - old) / std_b),
            "behavior_drift_rms": _event_rms(behavior_drift / std_b),
            "teacher_old_gap_whitened_rms": _event_rms(teacher_old_gap / std_b),
            "teacher_old_kl_joint": responsibility.teacher_old_kl_joint,
            "log_rho_sum": responsibility.log_ratio_sum,
            "gamma": gamma,
            "gamma_lt_001": (gamma < 0.01).float(),
            "gamma_gt_099": (gamma > 0.99).float(),
            "ungated_mean_kl": ungated_mean_kl,
            "gated_mean_kl": gamma * ungated_mean_kl,
            # Constant within a run, but kept because the calibration identity K = (D/2) w^2 is
            # unreadable without it and D cannot be recovered from a wandb run's config alone.
            "event_dim": torch.full_like(gamma, float(responsibility.event_dim)),
        }
        if verbose:
            metrics.update(
                {
                    "transition_std": transition_std,
                    "transition_variance": variance,
                    "abs_dt": delta.abs(),
                    "next_latent_rms": _event_rms(observed),
                    "mu_old_rms": _event_rms(old),
                    "mu_teacher_rms": _event_rms(teacher),
                    "mu_student_rms": _event_rms(student),
                    "teacher_innovation_rms": _event_rms((observed - teacher) / std_b),
                    "teacher_old_gap_rms": _event_rms(teacher_old_gap),
                    "teacher_old_gap_l2": _event_l2(teacher_old_gap),
                    "teacher_old_kl_per_dim": responsibility.teacher_old_kl_per_dim,
                    "log_rho_per_dim": responsibility.log_ratio_per_dim,
                    "tempered_log_ratio": responsibility.tempered_log_ratio,
                    "gate_logit": responsibility.gate_logit,
                    "gate_entropy": F.softplus(responsibility.gate_logit)
                    - gamma * responsibility.gate_logit,
                    "student_teacher_gap_rms": _event_rms(student_teacher_gap),
                    "student_teacher_gap_whitened_rms": _event_rms(student_teacher_gap / std_b),
                    "effective_gate": gamma,
                    "teacher_pull_rms": _event_rms(gamma_b * (old - teacher) / variance_b),
                    "event_dim": torch.full_like(gamma, float(responsibility.event_dim)),
                    "alpha": torch.full_like(gamma, responsibility.alpha),
                    "temperature": torch.full_like(gamma, responsibility.temperature),
                    "temperature_over_event_dim": torch.full_like(
                        gamma,
                        responsibility.temperature / float(responsibility.event_dim),
                    ),
                }
            )

    for name, value in metrics.items():
        value = value.detach()
        if value.shape != (batch_size,):
            raise ValueError(
                f"expected P-OPD diagnostic {name!r} shape ({batch_size},), "
                f"got shape={tuple(value.shape)}."
            )
        if not torch.isfinite(value).all():
            raise ValueError(
                f"expected finite P-OPD diagnostic {name!r}, "
                f"got values={value.cpu().tolist()}, "
                f"variance_min={variance.min().item()!r}, event_dim={responsibility.event_dim}."
            )
        metrics[name] = value
    return metrics


def cache_forward_signature(
    forward_fn: Callable[..., Any],
) -> Tuple[FrozenSet[str], bool]:
    """Snapshot ``inspect.signature(forward_fn)`` for cheap per-step filtering.

    Returns ``(param_names, accepts_var_kwargs)``: the declared parameter names
    and whether the callable accepts ``**kwargs``.
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
    """Filter ``full_kwargs`` to keys accepted by a cached forward signature."""
    if accepts_var_kwargs:
        return full_kwargs
    return {k: v for k, v in full_kwargs.items() if k in param_names}


def build_forward_kwargs(
    *,
    training_args: Any,
    batch: Dict[str, Any],
    t: torch.Tensor,
    t_next: torch.Tensor,
    latents: torch.Tensor,
    next_latents: torch.Tensor,
    compute_log_prob: bool,
    noise_level: Any,
    return_kwargs: List[str],
    param_names: FrozenSet[str],
    accepts_var_kwargs: bool,
) -> Dict[str, Any]:
    """Assemble the per-timestep ``adapter.forward`` kwargs (student or teacher).

    Mirrors ``OPDTrainer._build_forward_kwargs``: spreads ``training_args`` (a
    mapping-like dataclass) and ``batch`` (preprocessed text/latent fields),
    overlays the per-step tensors, then filters to the forward signature.
    """
    full_kwargs = {
        **training_args,
        "t": t,
        "t_next": t_next,
        "latents": latents,
        "next_latents": next_latents,
        "compute_log_prob": compute_log_prob,
        "noise_level": noise_level,
        **batch,
    }
    forward_kwargs = filter_forward_kwargs(full_kwargs, param_names, accepts_var_kwargs)
    forward_kwargs["return_kwargs"] = return_kwargs
    return forward_kwargs


def _to_clean_x0(
    mu: torch.Tensor,
    x_t: torch.Tensor,
    sigma: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Clean latent ``x0 = x_t - sigma * v`` from the ODE Euler transition mean.

    Recovers the velocity from ``mu = x_t + v*dt`` (ODE-exact): ``v = (mu - x_t)/dt``,
    then ``x0 = x_t - sigma*v``. ``sigma`` and ``dt`` are per-sample; broadcast over the
    trailing (non-batch) dims of ``mu``. See docs/xopd/per_timestep_loss_dominance_theory.tex.
    """
    sig = sigma.float()
    d = dt.float()
    while sig.dim() < mu.dim():
        sig = sig.unsqueeze(-1)
    while d.dim() < mu.dim():
        d = d.unsqueeze(-1)
    v = (mu.float() - x_t.float()) / d
    return x_t.float() - sig * v


def compute_per_step_kl(
    mu_student: torch.Tensor,
    mu_teacher: torch.Tensor,
    std_dev_t: torch.Tensor,
    dt: torch.Tensor,
    *,
    normalize: bool,
    space: str = "xt",
    latents: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-sample distillation d_k between student and teacher, in one of three spaces.

    All are plain per-sample MSE with a ``mean`` spatial reduction over non-batch dims. Under
    ODE (``mu = x_t + v*dt`` with x_t shared, so it cancels in the student-teacher diff) they are
    per-timestep reweightings of the same ``dv = v_s - v_t``: ``MSE(v):MSE(xt):MSE(x0) = 1:dt^2:sigma^2``.

    ``space="v"``: raw velocity MSE ``mean(||v_s - v_t||^2)`` with ``v = (mu - x_t)/dt``. Because
        x_t cancels, this is ``mean(||(mu_s - mu_t)/dt||^2)`` -- only ``dt`` is needed. ``normalize``
        is ignored. ODE-only (the ``mu = x_t + v*dt`` identity).
    ``space="x0"``: clean-latent MSE ``mean(||x0_s - x0_t||^2)`` with ``x0 = x_t - sigma*v``
        (requires ``latents`` (= x_t) and ``sigma``). ``normalize`` is ignored. ODE-only.
    ``space="x0_norm"``: DiffusionNFT/DMD self-normalized x0 regression --- the x0-MSE above divided
        by ``sg(mean|x0_s - x0_t|)`` (detached per-sample scale) + eps. Scale-invariant per step
        (equalizes per-step gradient magnitude, keeps the high-noise tilt). Same requirements as x0.
    ``space="xt"`` (default): the transition-mean / next-latent term (the DiffusionOPD default).
        ``normalize=True``: ``mean(||mu_s - mu_t||^2) / (2 * sigma_bar^2)`` with
        ``sigma_bar^2 = std_dev_t^2 * (-dt)``. ``normalize=False``: plain ``mean(||mu_s - mu_t||^2)``.
        Under ODE (``std_dev_t ~ 0``) falls back to plain MSE. Works under any dynamics.
    """
    if mu_student.shape != mu_teacher.shape:
        raise ValueError(
            "mu_student and mu_teacher must have the same shape, "
            f"got mu_student.shape={tuple(mu_student.shape)} vs "
            f"mu_teacher.shape={tuple(mu_teacher.shape)}."
        )

    if space == "v":
        # Raw velocity MSE. v_s - v_t = (mu_s - mu_t)/dt (ODE-exact; x_t cancels), so only dt is
        # needed -- no latents/sigma. Equivalent to MSE(xt)/dt^2.
        d = dt.float()
        while d.dim() < mu_student.dim():
            d = d.unsqueeze(-1)
        diff = (mu_student.float() - mu_teacher.float()) / d
        return (diff**2).mean(dim=tuple(range(1, diff.ndim)))

    if space == "x0":
        if latents is None or sigma is None:
            raise ValueError(
                "compute_per_step_kl(space='x0') requires `latents` (x_t) and `sigma`; "
                f"got latents={'None' if latents is None else 'set'}, "
                f"sigma={'None' if sigma is None else 'set'}."
            )
        x0_s = _to_clean_x0(mu_student, latents, sigma, dt)
        x0_t = _to_clean_x0(mu_teacher, latents, sigma, dt)
        return ((x0_s - x0_t) ** 2).mean(dim=tuple(range(1, x0_s.ndim)))

    if space == "x0_norm":
        # DiffusionNFT / DMD self-normalized x0 regression: the per-sample x0-MSE divided by the
        # DETACHED per-sample mean-abs x0 error. Scale-invariant -> equalizes each step's gradient
        # magnitude (removing the early-large/late-small tilt) while keeping the implied high-noise
        # weighting because the normalization is done in x0 space (see
        # docs/xopd/per_timestep_loss_dominance_theory.tex, sec "Adaptive self-normalized reweighting").
        # ``normalize`` is ignored. ODE-only (x0 = x_t - sigma*v).
        if latents is None or sigma is None:
            raise ValueError(
                "compute_per_step_kl(space='x0_norm') requires `latents` (x_t) and `sigma`; "
                f"got latents={'None' if latents is None else 'set'}, "
                f"sigma={'None' if sigma is None else 'set'}."
            )
        x0_s = _to_clean_x0(mu_student, latents, sigma, dt)
        x0_t = _to_clean_x0(mu_teacher, latents, sigma, dt)
        err = x0_s - x0_t
        spatial = tuple(range(1, err.ndim))
        num = (err ** 2).mean(dim=spatial)                    # (B,) per-sample x0 MSE (numerator)
        scale = err.abs().mean(dim=spatial).detach()          # (B,) sg(mean|x0_s - x0_t|)
        return num / (scale + 1e-8)                           # eps floors the denominator as gap -> 0

    if space != "xt":
        raise ValueError(f"space must be 'v', 'xt', 'x0' or 'x0_norm', got {space!r}.")

    diff_sq = (mu_student.float() - mu_teacher.float()) ** 2
    diff_sq = diff_sq.mean(dim=tuple(range(1, diff_sq.ndim)))  # (B,)

    if not normalize:
        return diff_sq

    sigma_bar_sq = ((std_dev_t.float() ** 2) * (-dt.float())).flatten()

    # Under ODE, std_dev_t is zero -> sigma_bar^2 = 0; use the sigma^2=1
    # convention (plain MSE) to avoid division by zero.
    if sigma_bar_sq.abs().max() < 1e-10:
        return diff_sq

    sigma_bar_sq = sigma_bar_sq.clamp(min=1e-12)
    return diff_sq / (2.0 * sigma_bar_sq)


def l0_loss_weight(
    sigma: torch.Tensor,
    scheme: Literal["min_snr", "snr", "uniform"] = "min_snr",
    gamma: float = 5.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Per-sample weight ``w(t)`` for L0 velocity regression.

    ``sigma`` is the flow-matching noise fraction in ``[0, 1]`` (``sigma = t/1000``;
    ``x_t = (1-sigma) x0 + sigma eps``). Schemes:

    - ``'uniform'``: ``w = 1``.
    - ``'snr'``: ``w = (1-sigma)/sigma`` (proportional to ``1/g_t^2``, the
      Girsanov velocity weight; ``~ sqrt(SNR)``).
    - ``'min_snr'``: ``w = min((1-sigma)/sigma, gamma)`` (clamp to avoid the
      ``sigma -> 0`` blow-up; the recommended default).

    Returns a tensor of the same shape as ``sigma``.
    """
    sigma = sigma.float().clamp(min=eps, max=1.0 - eps)
    if scheme == "uniform":
        return torch.ones_like(sigma)
    ratio = (1.0 - sigma) / sigma
    if scheme == "snr":
        return ratio
    if scheme == "min_snr":
        return ratio.clamp(max=gamma)
    raise ValueError(
        f"Unknown l0 weighting scheme={scheme!r}; expected 'min_snr', 'snr', or 'uniform'."
    )


def validate_l1_one_step_per_epoch(
    *,
    num_batches_per_epoch: int,
    num_train_timesteps: int,
    gradient_accumulation_steps: int,
    num_inner_epochs: int,
) -> None:
    """Fail fast unless the L1 stage performs exactly one optimizer step per epoch.

    L1 enters ``accelerator.accumulate`` ``num_batches_per_epoch * num_train_timesteps``
    times per epoch (one per training timestep, per micro-batch). For a single
    on-policy optimizer step per epoch, GAS must equal that product and
    ``num_inner_epochs`` must be 1 (no rollout reuse).
    """
    if num_inner_epochs != 1:
        raise ValueError(
            f"XOPD requires num_inner_epochs == 1 (one on-policy optimizer step per "
            f"epoch), got num_inner_epochs={num_inner_epochs}."
        )
    expected = num_batches_per_epoch * num_train_timesteps
    if gradient_accumulation_steps != expected:
        raise ValueError(
            "XOPD L1 requires exactly one optimizer step per epoch: "
            f"gradient_accumulation_steps ({gradient_accumulation_steps}) must equal "
            f"num_batches_per_epoch ({num_batches_per_epoch}) * num_train_timesteps "
            f"({num_train_timesteps}) = {expected}. Fix: set gradient_step_per_epoch=1 "
            "with gradient_accumulation_steps='auto', or set "
            f"gradient_accumulation_steps={expected} explicitly."
        )


def align_l0_inner_steps(
    num_batches_per_epoch: int,
    gradient_accumulation_steps: int,
    l0_inner_steps: int,
) -> int:
    """Round ``l0_inner_steps`` UP so each L0 epoch ends on a gradient-sync boundary.

    The no-leakage condition is ``(num_batches_per_epoch * l0_inner_steps) % GAS == 0``.
    The minimal value ``>= l0_inner_steps`` satisfying it is the next multiple of
    ``d = GAS // gcd(num_batches_per_epoch, GAS)``. Under the L1 invariant
    ``GAS == num_batches_per_epoch * num_train_timesteps`` this reduces to "round up
    to a multiple of ``num_train_timesteps``". L0 then runs ``l0_inner_steps // d``
    clean optimizer steps per epoch.
    """
    if num_batches_per_epoch <= 0 or gradient_accumulation_steps <= 0 or l0_inner_steps <= 0:
        raise ValueError(
            "align_l0_inner_steps requires positive inputs, got "
            f"num_batches_per_epoch={num_batches_per_epoch}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}, "
            f"l0_inner_steps={l0_inner_steps}."
        )
    d = gradient_accumulation_steps // math.gcd(num_batches_per_epoch, gradient_accumulation_steps)
    return math.ceil(l0_inner_steps / d) * d


# Optional I2I condition keys that may be present on preprocessed batches.
# Missing / None is normal for T2I samples in mixed T2I+I2I training (bsz=1).
I2I_LATENT_KEYS: Tuple[str, ...] = ("image_latents", "image_latent_ids")
I2I_PIXEL_KEYS: Tuple[str, ...] = ("images", "condition_images")


def extract_i2i_condition_kwargs(
    batch: Dict[str, Any],
    *,
    prefer_latents: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Collect optional image-condition kwargs from a batch (T2I-safe).

    Skips missing / ``None`` keys so mixed T2I+I2I training with
    ``per_device_batch_size=1`` does not crash on text-only batches.

    Args:
        batch: Preprocessed dataloader batch (may lack image fields).
        prefer_latents: If True (shared-VAE / student path), forward cached
            ``image_latents`` / ``image_latent_ids`` when present, plus
            ``condition_images`` for sample metadata / rewards. If False
            (independent teacher with a possibly different VAE), pass pixel
            condition as ``images`` so the callee can re-encode, and omit
            student ``image_latents``.
        device: Optional device to move tensor values onto.

    Returns:
        Dict of non-None I2I kwargs suitable for ``filter_kwargs`` / inference /
        ``predict_velocity``. Empty when the batch is T2I-only.
    """
    if not isinstance(batch, dict):
        raise TypeError(
            f"expected dict for batch, got {type(batch).__name__}: {batch!r}"
        )

    out: Dict[str, Any] = {}

    def _maybe_to_device(value: Any) -> Any:
        if device is None or not torch.is_tensor(value):
            return value
        return value.to(device)

    if prefer_latents:
        for key in (*I2I_LATENT_KEYS, "condition_images"):
            if key not in batch:
                continue
            value = batch[key]
            if value is None:
                continue
            out[key] = _maybe_to_device(value)
        return out

    # Cross-VAE / independent teacher: prefer pixel inputs so the teacher VAE
    # re-encodes. Do not forward student image_latents (wrong latent space).
    if batch.get("images") is not None:
        out["images"] = batch["images"]
    elif batch.get("condition_images") is not None:
        # Preprocess caches resized tensors as condition_images; reuse as the
        # encode_image input (Flux2 adapters accept tensor batches).
        out["images"] = batch["condition_images"]
    return out


def interleaved_source_iter(
    dataloaders_by_source: Dict[str, Any],
    source_ratio: Optional[Dict[str, float]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Block-cycle iterator over per-source dataloaders (copied from mof/utils.py).

    Each yielded batch is tagged with a ``__source__`` key (and per-row
    ``metadata[*]["__source__"]``) for downstream routing. XOPD uses a single
    teacher, so the tag only feeds eval/reward metadata; the teacher forward is
    source-agnostic.

    With ``source_ratio=None`` (default), iterates in sorted source-name order
    with equal 1:1:... weighting. When a source's dataloader is exhausted it is
    re-initialized (infinite cycle).

    With ``source_ratio={name: count, ...}``, builds a deterministic block
    pattern by repeating each source name ``count`` times in sorted source-name
    order. E.g. ``{"geneval": 2, "ocr": 2, "pickscore": 1}`` over sources
    ``[geneval, ocr, pickscore]`` yields the cycle ``G G O O P`` repeating. All
    ratio values must be non-negative integer-valued floats; missing / unknown
    source names raise ``ValueError``.

    Args:
        dataloaders_by_source: Dict mapping source name -> DataLoader.
        source_ratio: Optional dict mapping source name -> integer-valued
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
                "sum(source_ratio.values()) == 0 — at least one source " "must have weight > 0"
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

    Copied from mof/utils.py. Format errors (unknown/missing keys, non-integer
    values) are caught lazily by ``interleaved_source_iter`` on first use; this
    function only enforces the divisibility invariant that depends on
    ``num_batches_per_epoch``, plus a zero-sum guard, so XOPD fails at trainer
    ``__init__`` rather than after sampling starts.

    Args:
        source_ratio: Dict mapping source name -> integer-valued weight, or None.
        num_batches_per_epoch: Total iterator ticks per epoch.
        train_dataloaders_by_source: Dict mapping source name -> DataLoader.
            When empty (single-source mode) validation is a no-op.

    Raises:
        ValueError: If ``num_batches_per_epoch`` is not divisible by
            ``int(sum(source_ratio.values()))``.
    """
    if source_ratio is None or not train_dataloaders_by_source:
        return
    period = int(sum(source_ratio.values()))
    if period == 0:
        raise ValueError("source_ratio sum is 0 — at least one source must have weight > 0")
    if num_batches_per_epoch % period != 0:
        raise ValueError(
            f"num_batches_per_epoch ({num_batches_per_epoch}) must be divisible "
            f"by sum(source_ratio.values()) ({period}) for clean per-epoch cycles. "
            f"Adjust unique_sample_num_per_epoch or source_ratio. "
            f"Current source_ratio={source_ratio}."
        )
