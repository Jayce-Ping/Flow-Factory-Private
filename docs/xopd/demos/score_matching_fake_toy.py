#!/usr/bin/env python3
# Copyright 2026 Jayce-Ping
"""1D toy: score matching on external data vs sampling-defined p_θ.

Companion note: docs/xopd/score_matching_fake_network_primer.md (§2 and Appendix A).

Run from repo root:
  .venv/bin/python docs/xopd/demos/score_matching_fake_toy.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sample_x0(n: int, device: torch.device) -> torch.Tensor:
    mix = torch.randint(0, 2, (n,), device=device)
    means = torch.tensor([-2.0, 2.0], device=device)
    return means[mix] + 0.3 * torch.randn(n, device=device)


class ScoreNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.net(torch.stack([x_t, sigma], dim=-1)).squeeze(-1)


def train_on_distribution(
    sample_fn,
    steps: int = 1500,
    batch: int = 256,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> ScoreNet:
    """FM-style velocity matching: v* = ε - x0 on x_t = (1-σ)x0 + σ ε."""
    device = device or torch.device("cpu")
    net = ScoreNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        x0 = sample_fn(batch)
        sigma = torch.rand(batch, device=device).clamp(0.05, 0.95)
        eps = torch.randn(batch, device=device)
        x_t = (1.0 - sigma) * x0 + sigma * eps
        v_target = eps - x0
        loss = ((net(x_t, sigma) - v_target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return net


def main() -> None:
    device = torch.device("cpu")
    torch.manual_seed(0)

    # --- Teacher: SM/FM on *external* data ---
    teacher = train_on_distribution(lambda n: sample_x0(n, device), device=device)

    # --- Student sampler defines p_θ = Law(θ * Z), Z~N(0,1) ---
    # Gold for *sampling*; the scalar θ is NOT a score network.
    theta = 1.5

    def sample_student(n: int) -> torch.Tensor:
        return theta * torch.randn(n, device=device)

    # --- Fake: SM/FM on *student* samples (tracks ∇ log p_θ) ---
    fake = train_on_distribution(sample_student, device=device)

    # Probe: at a noisy point from student samples, teacher vs fake disagree
    # (different distributions) — illustrating two different score fields.
    x0_s = sample_student(4096)
    sigma = torch.full((4096,), 0.5, device=device)
    eps = torch.randn(4096, device=device)
    x_t = (1.0 - sigma) * x0_s + sigma * eps
    with torch.no_grad():
        v_t = teacher(x_t, sigma)
        v_f = fake(x_t, sigma)
    gap = (v_t - v_f).abs().mean().item()
    print("1D toy OK.")
    print(f"  student scale θ = {theta}")
    print(f"  mean |v_teacher - v_fake| on student-noised points @ σ=0.5: {gap:.4f}")
    print("  (gap > 0 expected: teacher scores p_data, fake scores p_θ)")
    print("See docs/xopd/score_matching_fake_network_primer.md")


if __name__ == "__main__":
    main()
