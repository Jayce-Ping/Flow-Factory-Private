# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Pure helpers for trajectory distribution matching (Approach B / TDM).

Design: docs/xopd/approach_b_trajectory_dm_design.md,
docs/xopd/tdm_cross_model_design.md,
docs/xopd/tdm_opd_dm_gradient_relation.tex.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist

from ...utils.noise_schedule import TIMESTEP_MAX, flow_match_sigma

__all__ = [
    "TIMESTEP_MAX",
    "flow_match_sigma",
    "broadcast_int",
    "uniform_scheduler_t",
    "x0_from_velocity",
    "add_noise_rf",
    "self_normalized_dm_grad",
    "dm_stopgrad_loss",
    "pseudo_huber_from_residual",
    "pseudo_huber_c_default",
    "ode_euler_next",
]


def broadcast_int(value: int, device: torch.device, world_size: int, src: int = 0) -> int:
    """Broadcast a Python int from ``src`` so all ranks share the same segment index."""
    if world_size <= 1 or not dist.is_initialized():
        return int(value)
    t = torch.tensor([int(value)], device=device, dtype=torch.long)
    dist.broadcast(t, src=src)
    return int(t.item())


def uniform_scheduler_t(
    batch_size: int,
    device: torch.device,
    lo_frac: float,
    hi_frac: float,
) -> torch.Tensor:
    """Per-sample uniform scheduler timesteps in ``[lo_frac, hi_frac] * TIMESTEP_MAX``."""
    if not (0.0 <= lo_frac <= hi_frac <= 1.0):
        raise ValueError(
            f"expected 0 <= lo_frac <= hi_frac <= 1 for uniform_scheduler_t, "
            f"got lo_frac={lo_frac!r}, hi_frac={hi_frac!r}"
        )
    if batch_size < 1:
        raise ValueError(f"expected batch_size >= 1, got {batch_size!r}")
    return (
        torch.rand(batch_size, device=device) * ((hi_frac - lo_frac) * TIMESTEP_MAX)
        + lo_frac * TIMESTEP_MAX
    )


def x0_from_velocity(
    latents: torch.Tensor, velocity: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Rectified-flow clean estimate: ``x0 = x_t - σ(t) * v``."""
    if latents.shape != velocity.shape:
        raise ValueError(
            f"latents and velocity shapes must match, got {tuple(latents.shape)} vs "
            f"{tuple(velocity.shape)}"
        )
    sig = flow_match_sigma(t).reshape(-1, *([1] * (latents.ndim - 1)))
    return latents - sig * velocity


def add_noise_rf(x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """``x_t = (1-σ) x0 + σ ε``. Returns ``(x_t, eps)``."""
    if eps is None:
        eps = torch.randn_like(x0)
    elif eps.shape != x0.shape:
        raise ValueError(
            f"eps shape must match x0, got {tuple(eps.shape)} vs {tuple(x0.shape)}"
        )
    sig = flow_match_sigma(t).reshape(-1, *([1] * (x0.ndim - 1)))
    return (1.0 - sig) * x0 + sig * eps, eps


def self_normalized_dm_grad(
    p_real: torch.Tensor, p_fake: torch.Tensor
) -> torch.Tensor:
    """DMD2 self-normalized direction ``(p_real - p_fake) / mean|p_real|`` (nan_to_num)."""
    if p_real.shape != p_fake.shape:
        raise ValueError(
            f"p_real and p_fake shapes must match, got {tuple(p_real.shape)} vs "
            f"{tuple(p_fake.shape)}"
        )
    spatial = tuple(range(1, p_real.ndim))
    norm = p_real.float().abs().mean(dim=spatial, keepdim=True)
    return torch.nan_to_num((p_real.float() - p_fake.float()) / norm)


def dm_stopgrad_loss(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Stop-grad identity ``0.5 * MSE(x, sg(x - grad))`` so ``∂loss/∂x = grad`` (mean over batch)."""
    if x.shape != grad.shape:
        raise ValueError(
            f"x and grad shapes must match for dm_stopgrad_loss, got "
            f"{tuple(x.shape)} vs {tuple(grad.shape)}"
        )
    spatial = tuple(range(1, x.ndim))
    target = (x.float() - grad.float()).detach()
    return 0.5 * (x.float() - target).pow(2).mean(dim=spatial).mean()


def pseudo_huber_c_default(num_elements: int) -> float:
    """iCT / TDM default ``c = 0.00054 * sqrt(d)`` for data with ``d`` dimensions."""
    if num_elements < 1:
        raise ValueError(f"expected num_elements >= 1, got {num_elements!r}")
    return 0.00054 * float(num_elements) ** 0.5


def pseudo_huber_from_residual(
    residual: torch.Tensor, c: float
) -> torch.Tensor:
    """``sqrt(||r||^2 + c^2) - c`` averaged over batch (per-sample spatial mean of r^2)."""
    if c <= 0.0:
        raise ValueError(f"expected c > 0 for Pseudo-Huber, got {c!r}")
    spatial = tuple(range(1, residual.ndim))
    sq = residual.float().pow(2).mean(dim=spatial)
    return (sq + c * c).sqrt().mean() - c


def ode_euler_next(
    latents: torch.Tensor, velocity: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor
) -> torch.Tensor:
    """Deterministic RF Euler step: ``x_{t_next} = x_t + v * (σ_next - σ)`` with ``σ = t/1000``.

    For FLUX flow-matching, scheduler timesteps decrease (noise→data); ``dt_sigma = σ_next - σ``
    is typically negative when ``v`` points toward data.
    """
    if latents.shape != velocity.shape:
        raise ValueError(
            f"latents and velocity shapes must match, got {tuple(latents.shape)} vs "
            f"{tuple(velocity.shape)}"
        )
    sig = flow_match_sigma(t).reshape(-1, *([1] * (latents.ndim - 1)))
    sig_next = flow_match_sigma(t_next).reshape(-1, *([1] * (latents.ndim - 1)))
    return latents + velocity * (sig_next - sig)
