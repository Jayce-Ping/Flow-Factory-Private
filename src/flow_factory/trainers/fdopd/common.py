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

"""Pure target, trust, and validation helpers for Flow Direct-OPD."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence

import torch


@dataclass(frozen=True)
class FDOPDTarget:
    """Detached Flow Direct-OPD target and per-sample trust statistics."""

    target: torch.Tensor
    donor_delta: torch.Tensor
    lambda_eff: torch.Tensor
    clipped: torch.Tensor
    delta_rms: torch.Tensor
    delta_l2: torch.Tensor
    relative_delta_rms: torch.Tensor
    target_shift_rms: torch.Tensor
    unit_kl_per_dim: Optional[torch.Tensor]


def _event_axes(tensor: torch.Tensor, *, name: str) -> tuple[int, ...]:
    if tensor.ndim < 2:
        raise ValueError(
            f"{name} must include batch and event dimensions, got shape={tuple(tensor.shape)}."
        )
    return tuple(range(1, tensor.ndim))


def _require_finite(tensor: torch.Tensor, *, name: str) -> None:
    if not torch.isfinite(tensor).all():
        finite_fraction = torch.isfinite(tensor).float().mean().item()
        raise ValueError(
            f"{name} must be finite, got shape={tuple(tensor.shape)} and "
            f"finite_fraction={finite_fraction:.6f}."
        )


def _broadcast_batch_scalar(
    value: torch.Tensor, reference: torch.Tensor, *, name: str
) -> torch.Tensor:
    result = value
    while result.ndim < reference.ndim:
        result = result.unsqueeze(-1)
    try:
        return torch.broadcast_to(result, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} must broadcast to reference shape={tuple(reference.shape)}, "
            f"got shape={tuple(value.shape)}."
        ) from exc


def compose_fdopd_target(
    *,
    recipient_base: torch.Tensor,
    donor_base: torch.Tensor,
    donor_rl: torch.Tensor,
    transfer_strength: float,
    compute_delta_fp32: bool = True,
    max_relative_delta_rms: Optional[float] = None,
    trust_kl_per_dim: Optional[float] = None,
    transition_variance: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> FDOPDTarget:
    """Compose a detached recipient-base plus donor-shift target.

    Args:
        recipient_base: Recipient reference output with shape ``(B, *event)``.
        donor_base: Donor output before RL, with the same shape.
        donor_rl: Donor output after RL, with the same shape.
        transfer_strength: Nominal non-negative policy-shift multiplier.
        compute_delta_fp32: Compute subtraction and target composition in float32.
        max_relative_delta_rms: Optional cap on target-shift RMS divided by
            recipient-base RMS.
        trust_kl_per_dim: Optional target-to-recipient-base KL budget per event
            dimension.
        transition_variance: Positive transition variance broadcastable to the
            output shape; required with ``trust_kl_per_dim``.
        eps: Positive denominator floor for relative-scale diagnostics.

    Returns:
        Detached target tensors and per-sample trust statistics.
    """
    if recipient_base.shape != donor_base.shape or recipient_base.shape != donor_rl.shape:
        raise ValueError(
            "recipient_base, donor_base, and donor_rl must have identical shapes, got "
            f"recipient_base={tuple(recipient_base.shape)}, "
            f"donor_base={tuple(donor_base.shape)}, donor_rl={tuple(donor_rl.shape)}."
        )
    axes = _event_axes(recipient_base, name="recipient_base")
    if (
        isinstance(transfer_strength, bool)
        or not isinstance(transfer_strength, (int, float))
        or not math.isfinite(float(transfer_strength))
        or float(transfer_strength) < 0.0
    ):
        raise ValueError(
            "transfer_strength must be a finite number >= 0, got " f"{transfer_strength!r}."
        )
    if max_relative_delta_rms is not None and trust_kl_per_dim is not None:
        raise ValueError(
            "compose_fdopd_target accepts at most one trust budget, got both "
            f"max_relative_delta_rms={max_relative_delta_rms!r} and "
            f"trust_kl_per_dim={trust_kl_per_dim!r}."
        )
    if eps <= 0.0 or not math.isfinite(float(eps)):
        raise ValueError(f"eps must be finite and > 0, got eps={eps!r}.")

    dtype = torch.float32 if compute_delta_fp32 else recipient_base.dtype
    recipient = recipient_base.detach().to(dtype)
    base = donor_base.detach().to(dtype)
    rl = donor_rl.detach().to(dtype)
    _require_finite(recipient, name="recipient_base")
    _require_finite(base, name="donor_base")
    _require_finite(rl, name="donor_rl")

    donor_delta = rl - base
    delta_rms = donor_delta.square().mean(dim=axes).sqrt()
    delta_l2 = donor_delta.square().sum(dim=axes).sqrt()
    recipient_rms = recipient.square().mean(dim=axes).sqrt()
    relative_delta_rms = delta_rms / recipient_rms.clamp_min(eps)

    lambda_eff = torch.full_like(delta_rms, float(transfer_strength))
    unit_kl_per_dim: Optional[torch.Tensor] = None

    if max_relative_delta_rms is not None:
        if not math.isfinite(float(max_relative_delta_rms)) or float(max_relative_delta_rms) <= 0.0:
            raise ValueError(
                "max_relative_delta_rms must be finite and > 0, got " f"{max_relative_delta_rms!r}."
            )
        undefined = (recipient_rms <= eps) & (delta_rms > eps)
        if undefined.any():
            indices = undefined.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                "relative-RMS trust requires non-zero recipient-base RMS for samples "
                f"with non-zero donor delta; invalid batch indices={indices}."
            )
        cap = float(max_relative_delta_rms) * recipient_rms / delta_rms.clamp_min(eps)
        lambda_eff = torch.minimum(lambda_eff, cap)

    if trust_kl_per_dim is not None:
        if not math.isfinite(float(trust_kl_per_dim)) or float(trust_kl_per_dim) <= 0.0:
            raise ValueError(
                "trust_kl_per_dim must be finite and > 0, got " f"{trust_kl_per_dim!r}."
            )
        if transition_variance is None:
            raise ValueError(
                "trust_kl_per_dim requires transition_variance, got transition_variance=None."
            )
        variance = transition_variance.detach().to(torch.float32)
        _require_finite(variance, name="transition_variance")
        variance_full = _broadcast_batch_scalar(
            variance,
            donor_delta,
            name="transition_variance",
        )
        if (variance_full <= 0.0).any():
            minimum = variance_full.min().item()
            raise ValueError(
                "transition_variance must be strictly positive for KL trust, "
                f"got minimum={minimum:.8e}."
            )
        unit_kl_per_dim = 0.5 * (donor_delta.square() / variance_full).mean(dim=axes)
        cap = torch.sqrt(
            torch.full_like(unit_kl_per_dim, float(trust_kl_per_dim))
            / unit_kl_per_dim.clamp_min(eps)
        )
        lambda_eff = torch.minimum(lambda_eff, cap)

    lambda_broadcast = lambda_eff
    while lambda_broadcast.ndim < donor_delta.ndim:
        lambda_broadcast = lambda_broadcast.unsqueeze(-1)
    target = recipient + lambda_broadcast * donor_delta
    target_shift_rms = lambda_eff * delta_rms
    clipped = lambda_eff < (float(transfer_strength) - 1e-12)

    return FDOPDTarget(
        target=target.detach(),
        donor_delta=donor_delta.detach(),
        lambda_eff=lambda_eff.detach(),
        clipped=clipped.detach(),
        delta_rms=delta_rms.detach(),
        delta_l2=delta_l2.detach(),
        relative_delta_rms=relative_delta_rms.detach(),
        target_shift_rms=target_shift_rms.detach(),
        unit_kl_per_dim=None if unit_kl_per_dim is None else unit_kl_per_dim.detach(),
    )


def fdopd_target_diagnostics(
    result: FDOPDTarget,
    *,
    recipient_base: torch.Tensor,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """Return detached per-sample target diagnostics.

    Every key here is a per-sample tensor that the reducer expands into four statistics, so the
    default set is kept to the quantities that cannot be derived from one another:

    * ``delta_rms`` and ``recipient_base_rms`` -- the two magnitudes everything else is built from;
    * ``relative_delta_rms`` -- how large the donor's RL shift is against the recipient's own field,
      the scale-free number that decides whether the trust cap binds;
    * ``relative_target_shift_rms`` -- what the recipient is actually asked to move, and the
      quantity ``fdopd_max_relative_delta_rms`` bounds directly;
    * ``lambda_eff`` and ``trust_clipped`` -- the realized transfer strength and how often the cap
      is active.

    Dropped from the default set: ``delta_l2``, which is ``delta_rms`` times a constant
    sqrt(event_dim) and so carries no extra information while being incomparable across latent
    shapes, and ``target_shift_rms``, whose relative form is the one the trust budget is expressed
    in. Both return under ``verbose``.
    """
    axes = _event_axes(recipient_base, name="recipient_base")
    recipient_rms = recipient_base.detach().float().square().mean(dim=axes).sqrt()
    diagnostics = {
        "delta_rms": result.delta_rms.detach(),
        "relative_delta_rms": result.relative_delta_rms.detach(),
        "recipient_base_rms": recipient_rms.detach(),
        "relative_target_shift_rms": (
            result.target_shift_rms / recipient_rms.clamp_min(1e-8)
        ).detach(),
        "lambda_eff": result.lambda_eff.detach(),
        "trust_clipped": result.clipped.detach().float(),
    }
    if result.unit_kl_per_dim is not None:
        diagnostics["unit_kl_per_dim"] = result.unit_kl_per_dim.detach()
    if verbose:
        diagnostics["delta_l2"] = result.delta_l2.detach()
        diagnostics["target_shift_rms"] = result.target_shift_rms.detach()
    return diagnostics


def validate_fdopd_transition_stats(
    *,
    recipient_std: torch.Tensor,
    donor_std: torch.Tensor,
    recipient_dt: torch.Tensor,
    donor_dt: torch.Tensor,
    context: str,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Fail when donor and recipient transition geometry differs."""
    pairs = (
        ("transition std", recipient_std, donor_std),
        ("dt", recipient_dt, donor_dt),
    )
    for label, recipient, donor in pairs:
        recipient_value = recipient.detach().float()
        donor_value = donor.detach().float()
        _require_finite(recipient_value, name=f"recipient {label}")
        _require_finite(donor_value, name=f"donor {label}")
        try:
            recipient_value, donor_value = torch.broadcast_tensors(recipient_value, donor_value)
        except RuntimeError as exc:
            raise ValueError(
                f"Flow Direct-OPD {label} shapes must broadcast in {context}, got "
                f"recipient={tuple(recipient.shape)} and donor={tuple(donor.shape)}."
            ) from exc
        if not torch.allclose(recipient_value, donor_value, atol=atol, rtol=rtol):
            maximum = (recipient_value - donor_value).abs().max().item()
            raise ValueError(
                f"Flow Direct-OPD {label} mismatch in {context}: expected donor and "
                f"recipient values to match within atol={atol}, rtol={rtol}; "
                f"max_abs_diff={maximum:.8e}."
            )


def synchronize_fdopd_scheduler_state(recipient, donor) -> None:
    """Copy the exact recipient timestep/sigma grid into the donor scheduler."""

    def copy_value(name: str, *, required: bool) -> None:
        if not hasattr(recipient, name):
            if required:
                raise AttributeError(
                    f"Recipient scheduler must expose {name!r} for Flow Direct-OPD."
                )
            return
        value = getattr(recipient, name)
        if torch.is_tensor(value):
            donor_current = getattr(donor, name, None)
            device = donor_current.device if torch.is_tensor(donor_current) else value.device
            copied = value.detach().to(device=device).clone()
        elif isinstance(value, list):
            copied = list(value)
        else:
            copied = value
        setattr(donor, name, copied)

    copy_value("timesteps", required=True)
    copy_value("sigmas", required=True)
    copy_value("num_inference_steps", required=False)
    copy_value("_sde_steps", required=False)
    copy_value("_num_sde_steps", required=False)
    copy_value("seed", required=False)
    copy_value("_is_eval", required=False)
    copy_value("noise_level", required=False)
    copy_value("dynamics_type", required=False)
    if hasattr(donor, "_step_index"):
        donor._step_index = None
    if hasattr(donor, "_begin_index"):
        donor._begin_index = None
    if hasattr(recipient, "train_timesteps") and hasattr(donor, "train_timesteps"):
        recipient_steps = torch.as_tensor(recipient.train_timesteps).detach().cpu()
        donor_steps = torch.as_tensor(donor.train_timesteps).detach().cpu()
        if not torch.equal(recipient_steps, donor_steps):
            raise ValueError(
                "Flow Direct-OPD derived donor train_timesteps do not match the recipient "
                f"after scheduler synchronization: recipient={recipient_steps.tolist()}, "
                f"donor={donor_steps.tolist()}."
            )


def validate_fdopd_runtime(
    *,
    dynamics_type: str,
    loss_space: str,
    normalize_d_k: bool,
    trust_kl_per_dim: Optional[float],
    max_relative_delta_rms: Optional[float],
) -> None:
    """Validate dynamics-dependent Flow Direct-OPD choices."""
    if dynamics_type == "ODE":
        if trust_kl_per_dim is not None:
            raise ValueError(
                "Flow Direct-OPD cannot apply transition-KL trust under a deterministic ODE; "
                "use fdopd_max_relative_delta_rms instead."
            )
        if loss_space != "v":
            raise ValueError(
                "Flow Direct-OPD ODE requires fdopd_loss_space='v', got "
                f"fdopd_loss_space={loss_space!r}."
            )
        return
    if dynamics_type not in ("Flow-SDE", "Dance-SDE", "CPS"):
        raise ValueError(
            "Flow Direct-OPD supports dynamics_type in "
            "('ODE', 'Flow-SDE', 'Dance-SDE', 'CPS'), got "
            f"dynamics_type={dynamics_type!r}."
        )
    if loss_space != "xt":
        raise ValueError(
            "Flow Direct-OPD stochastic dynamics require fdopd_loss_space='xt', got "
            f"fdopd_loss_space={loss_space!r}."
        )
    if trust_kl_per_dim is not None and not normalize_d_k:
        raise ValueError(
            "fdopd_trust_kl_per_dim requires normalize_d_k=True so optimization and "
            "trust use the same transition covariance."
        )
    if max_relative_delta_rms is not None:
        raise ValueError(
            "fdopd_max_relative_delta_rms is the ODE velocity-space trust; use "
            "fdopd_trust_kl_per_dim for stochastic dynamics."
        )


def select_fdopd_steps(
    *,
    pool: Sequence[int],
    num_steps: int,
    strategy: Literal["uniform", "stratified"],
    seed: int,
) -> List[int]:
    """Select a deterministic fixed-size subset of rollout-step indices."""
    values = list(pool)
    if not values:
        raise ValueError("pool must contain at least one rollout-step index.")
    if len(set(values)) != len(values):
        raise ValueError(f"pool must contain unique indices, got pool={values!r}.")
    if not isinstance(num_steps, int) or num_steps < 1 or num_steps > len(values):
        raise ValueError(
            "num_steps must be an integer in [1, len(pool)], got "
            f"num_steps={num_steps!r}, len(pool)={len(values)}."
        )
    if strategy not in ("uniform", "stratified"):
        raise ValueError(
            "strategy must be 'uniform' or 'stratified', got " f"strategy={strategy!r}."
        )
    generator = torch.Generator().manual_seed(int(seed))
    if strategy == "uniform":
        selected_positions = sorted(
            torch.randperm(len(values), generator=generator)[:num_steps].tolist()
        )
    else:
        bounds = [round(i * len(values) / num_steps) for i in range(num_steps + 1)]
        selected_positions = []
        for segment in range(num_steps):
            lower, upper = bounds[segment], bounds[segment + 1]
            if upper <= lower:
                raise ValueError(
                    "stratified step sampling produced an empty segment, got "
                    f"segment={segment}, bounds=({lower}, {upper}), "
                    f"pool_size={len(values)}, num_steps={num_steps}."
                )
            selected_positions.append(
                int(torch.randint(lower, upper, (1,), generator=generator).item())
            )
    return [values[position] for position in selected_positions]
